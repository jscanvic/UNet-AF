// Metal implementation of circular_filtered_lrelu for Apple Silicon (PyTorch MPS).
//
// Semantics match `_circular_filtered_lrelu_ref`: everything is periodic, so the
// intermediate (upsampled) signal has period L = in_size * up along each axis and
// every index is wrapped modulo the relevant period.
//
// With P_ZERO_PAD set, the same kernels implement StyleGAN3's zero-padded
// `filtered_lrelu` instead: out-of-range input samples read as zero, the
// intermediate spans [0, LZ) without wrapping, and the output size is explicit.
//
// Forward for one axis (the 2D op is the tensor product of two such axes):
//   z[k] = scale * sum_t wu[t] * u[k - OU + t],   u[m] = x[m / up] + b if m % up == 0 else 0
//   a[k] = clamp(lrelu(z[k], slope), -clamp, clamp)
//   y[i] = gain_out * sum_t wd[t] * a[down * i - OD + t]
// where wu/wd are correlation weights (the host flips the user filters when
// flip_filter=False) and OU/OD are the left padding amounts.
//
// Backward reuses the same kernels with up/down swapped, reversed weights, the
// right padding amounts as offsets, and mode=2, which multiplies by the stored
// activation derivative instead of applying the activation.
//
// Integer and float parameter layouts are shared with circular_filtered_lrelu_mps.py.

#include <metal_stdlib>
using namespace metal;

enum : int {
  P_W = 0,  // input width
  P_H,      // input height
  P_UP,     // upsampling factor
  P_DOWN,   // downsampling factor
  P_FUX,    // upsampling filter taps along x
  P_FUY,    // upsampling filter taps along y
  P_OUX,    // upsampling filter offset (left padding) along x
  P_OUY,
  P_FDX,    // downsampling filter taps along x
  P_FDY,
  P_ODX,    // downsampling filter offset along x
  P_ODY,
  P_TOX,    // output tile width (fused kernel only)
  P_TOY,
  P_ZNX,    // intermediate tile width (fused kernel only)
  P_ZNY,
  P_INX,    // input tile width (fused kernel only)
  P_INY,
  P_C,      // number of channels (for bias indexing)
  P_HAS_BIAS,
  P_MODE,   // 0: forward, 1: forward + write mask, 2: backward (read mask)
  P_WO,     // output width
  P_HO,     // output height
  P_LZX,    // intermediate extent along x (mask layout; the period when circular)
  P_LZY,
  P_ZERO_PAD,  // 0: circular boundary, 1: zero padding
  P_COUNT
};

enum : int { F_SCALE = 0, F_GAIN_OUT, F_SLOPE, F_CLAMP, F_COUNT };

enum : int { MODE_FWD = 0, MODE_FWD_MASK = 1, MODE_BWD = 2 };

// Mask codes: derivative of the activation w.r.t. its (scaled) input.
enum : uchar { CODE_PASS = 0, CODE_SLOPE = 1, CODE_ZERO = 2 };

// Fused kernel threadgroup shape and threadgroup buffer capacities. Keep in sync with the host.
constant constexpr int TG_DIM = 16;
constant constexpr int TG_SIZE = TG_DIM * TG_DIM;
constant constexpr int BUF0_CAP = 1856;  // floats
constant constexpr int BUF1_CAP = 1280;  // floats
constant constexpr int AXIS_CAP = 48;    // max input / intermediate tile extent per axis

inline int pmod(int a, int n) {
  int r = a % n;
  return r < 0 ? r + n : r;
}

inline int floordiv(int a, int b) {
  return a >= 0 ? a / b : -((-a + b - 1) / b);
}

inline int ceildiv(int a, int b) {
  return -floordiv(-a, b);
}

// Forward: applies leaky ReLU and clamp, and optionally records the derivative code.
// Backward: gates the incoming gradient with the recorded derivative code.
inline float act_or_gate(
    float v,
    int mode,
    float slope,
    float clampv,
    device uchar* mask,
    ulong midx,
    bool write) {
  if (mode == MODE_BWD) {
    const uchar code = mask[midx];
    return code == CODE_PASS ? v : (code == CODE_SLOPE ? v * slope : 0.0f);
  }
  const bool neg = !(v > 0.0f);  // matches torch leaky_relu backward at 0
  float a = neg ? v * slope : v;
  const bool clamped = fabs(a) > clampv;
  if (clamped) {
    a = copysign(clampv, a);
  }
  if (mode == MODE_FWD_MASK && write) {
    mask[midx] = clamped ? CODE_ZERO : (neg ? CODE_SLOPE : CODE_PASS);
  }
  return a;
}

// ---------------------------------------------------------------------------
// Fused kernel for separable filters.
//
// One 16x16 threadgroup computes a TOY x TOX output tile of one (n, c) plane:
//   0. per-tile index tables and filters          -> threadgroup memory
//   1. load the wrapped input tile                -> buf0 [INY][INX]
//   2. upsample + filter along x                  -> buf1 [INY][ZNX]
//   3. upsample + filter along y, activation      -> buf0 [ZNY][ZNX]
//   4. filter + downsample along x                -> buf1 [ZNY][TOX]
//   5. filter + downsample along y                -> y
// Filters are packed as [wu_x (FUX), wu_y (FUY), wd_x (FDX), wd_y (FDY)].
// The tables hoist every modulo and division out of the inner loops.
// ---------------------------------------------------------------------------

template <typename T, int MODE>
kernel void cflrelu_sep(
    device const T* x [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    constant float* filt [[buffer(2)]],
    device uchar* mask [[buffer(3)]],
    device T* y [[buffer(4)]],
    constant int* ip [[buffer(5)]],
    constant float* fp [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup float buf0[BUF0_CAP];
  threadgroup float buf1[BUF1_CAP];
  // Threadgroup memory use caps occupancy, which bounds this kernel's speed.
  // Wrapped input row/column for the tile load.
  threadgroup int in_gx[AXIS_CAP], in_gy[AXIS_CAP];
  // Polyphase first tap and first local input index for each intermediate row/column.
  threadgroup int zx_t0[AXIS_CAP], zx_li[AXIS_CAP];
  threadgroup int zy_t0[AXIS_CAP], zy_li[AXIS_CAP];
  // Global intermediate coordinates (wrapped, or -1 when out of range), for reading the
  // mask in the backward pass.
  threadgroup int zx_g[MODE == MODE_BWD ? AXIS_CAP : 1], zy_g[MODE == MODE_BWD ? AXIS_CAP : 1];

  const int W = ip[P_W], H = ip[P_H], U = ip[P_UP], D = ip[P_DOWN];
  const int Wo = ip[P_WO], Ho = ip[P_HO];
  const int LZx = ip[P_LZX], LZy = ip[P_LZY];
  const bool zero_pad = ip[P_ZERO_PAD] != 0;
  const int FUx = ip[P_FUX], FUy = ip[P_FUY], OUx = ip[P_OUX], OUy = ip[P_OUY];
  const int FDx = ip[P_FDX], FDy = ip[P_FDY], ODx = ip[P_ODX], ODy = ip[P_ODY];
  const int TOx = ip[P_TOX], TOy = ip[P_TOY];
  const int ZNx = ip[P_ZNX], ZNy = ip[P_ZNY];
  const int INx = ip[P_INX], INy = ip[P_INY];
  const float scale = fp[F_SCALE], gain_out = fp[F_GAIN_OUT];
  const float slope = fp[F_SLOPE], clampv = fp[F_CLAMP];

  const int tx = int(lid.x), ty = int(lid.y), tid = ty * TG_DIM + tx;
  const int nc = int(tg.z);
  const int o0x = int(tg.x) * TOx, o0y = int(tg.y) * TOy;
  const int z0x = D * o0x - ODx, z0y = D * o0y - ODy;
  const int i0x = ceildiv(z0x - OUx, U), i0y = ceildiv(z0y - OUy, U);
  const float bval = ip[P_HAS_BIAS] ? float(bias[nc % ip[P_C]]) : 0.0f;

  // 0. Tables.
  for (int i = tid; i < INx; i += TG_SIZE) {
    const int g = i0x + i;
    in_gx[i] = zero_pad ? ((g >= 0 && g < W) ? g : -1) : pmod(g, W);
  }
  for (int i = tid; i < INy; i += TG_SIZE) {
    const int g = i0y + i;
    in_gy[i] = zero_pad ? ((g >= 0 && g < H) ? g : -1) : pmod(g, H);
  }
  for (int i = tid; i < ZNx; i += TG_SIZE) {
    const int m = z0x + i - OUx;
    const int t = pmod(-m, U);
    zx_t0[i] = t;
    zx_li[i] = (m + t) / U - i0x;
    if (MODE == MODE_BWD) {
      const int k = z0x + i;
      zx_g[i] = zero_pad ? ((k >= 0 && k < LZx) ? k : -1) : pmod(k, LZx);
    }
  }
  for (int i = tid; i < ZNy; i += TG_SIZE) {
    const int m = z0y + i - OUy;
    const int t = pmod(-m, U);
    zy_t0[i] = t;
    zy_li[i] = (m + t) / U - i0y;
    if (MODE == MODE_BWD) {
      const int k = z0y + i;
      zy_g[i] = zero_pad ? ((k >= 0 && k < LZy) ? k : -1) : pmod(k, LZy);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constant float* wux = filt;
  constant float* wuy = wux + FUx;
  constant float* wdx = wuy + FUy;
  constant float* wdy = wdx + FDx;

  // 1. Load the input tile, wrapping circularly or reading zeros outside the image.
  device const T* xp = x + ulong(nc) * ulong(H) * ulong(W);
  for (int ly = ty; ly < INy; ly += TG_DIM) {
    const int gy = in_gy[ly];
    threadgroup float* row = buf0 + ly * INx;
    for (int lx = tx; lx < INx; lx += TG_DIM) {
      const int gx = in_gx[lx];
      row[lx] = (gx >= 0 && gy >= 0) ? float(xp[gy * W + gx]) + bval : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Upsample and filter along x (polyphase: only taps that hit nonzero samples).
  for (int ly = ty; ly < INy; ly += TG_DIM) {
    threadgroup const float* row = buf0 + ly * INx;
    threadgroup float* out = buf1 + ly * ZNx;
    for (int rx = tx; rx < ZNx; rx += TG_DIM) {
      int li = zx_li[rx];
      float acc = 0.0f;
      for (int t = zx_t0[rx]; t < FUx; t += U, ++li) {
        acc = fma(wux[t], row[li], acc);
      }
      out[rx] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Upsample and filter along y, then activation (or gradient gating).
  device uchar* mp = mask + ulong(nc) * ulong(LZy) * ulong(LZx);
  // Each intermediate sample is owned by exactly one tile, which records its mask code:
  // a tile owns [D * o0, D * (o0 + TO)), and the last tile also owns the tail up to LZ,
  // which the zero-padded op reads when D * Wo < LZ. Owned samples lie in [0, LZ), so
  // writes need no wrapping; backward reads do.
  const int own_x1 = o0x + TOx >= Wo ? ZNx : ODx + D * TOx;
  const int own_y1 = o0y + TOy >= Ho ? ZNy : ODy + D * TOy;
  for (int ry = ty; ry < ZNy; ry += TG_DIM) {
    const int t0 = zy_t0[ry], li0 = zy_li[ry];
    const bool own_y = ry >= ODy && ry < own_y1 && z0y + ry < LZy;
    const int gzy = MODE == MODE_BWD ? zy_g[ry] : 0;
    device uchar* mrow_bwd = mp + (gzy >= 0 ? ulong(gzy) * ulong(LZx) : 0);
    device uchar* mrow_fwd = mp + long(z0y + ry) * long(LZx) + z0x;
    threadgroup float* out = buf0 + ry * ZNx;
    for (int rx = tx; rx < ZNx; rx += TG_DIM) {
      int li = li0;
      float acc = 0.0f;
      for (int t = t0; t < FUy; t += U, ++li) {
        acc = fma(wuy[t], buf1[li * ZNx + rx], acc);
      }
      float v = scale * acc;
      if (MODE == MODE_BWD) {
        // Outside [0, LZ) (zero padding only) the incoming gradient is exactly zero.
        const int gzx = zx_g[rx];
        const uchar code = (gzy >= 0 && gzx >= 0) ? mrow_bwd[gzx] : CODE_ZERO;
        v = code == CODE_PASS ? v : (code == CODE_SLOPE ? v * slope : 0.0f);
      } else {
        const bool neg = !(v > 0.0f);  // matches torch leaky_relu backward at 0
        v = neg ? v * slope : v;
        const bool clamped = fabs(v) > clampv;
        v = clamped ? copysign(clampv, v) : v;
        if (MODE == MODE_FWD_MASK && own_y && rx >= ODx && rx < own_x1 && z0x + rx < LZx) {
          mrow_fwd[rx] = clamped ? CODE_ZERO : (neg ? CODE_SLOPE : CODE_PASS);
        }
      }
      out[rx] = v;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 4. Filter and downsample along x.
  for (int ry = ty; ry < ZNy; ry += TG_DIM) {
    threadgroup const float* row = buf0 + ry * ZNx;
    threadgroup float* out = buf1 + ry * TOx;
    for (int ox = tx; ox < TOx; ox += TG_DIM) {
      threadgroup const float* src = row + D * ox;
      float acc = 0.0f;
      for (int t = 0; t < FDx; ++t) {
        acc = fma(wdx[t], src[t], acc);
      }
      out[ox] = acc;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 5. Filter and downsample along y, write output.
  device T* yp = y + ulong(nc) * ulong(Ho) * ulong(Wo);
  for (int oy = ty; oy < TOy; oy += TG_DIM) {
    const int gy = o0y + oy;
    if (gy >= Ho) {
      break;
    }
    for (int ox = tx; ox < TOx; ox += TG_DIM) {
      const int gx = o0x + ox;
      if (gx >= Wo) {
        break;
      }
      float acc = 0.0f;
      for (int t = 0; t < FDy; ++t) {
        acc = fma(wdy[t], buf1[(D * oy + t) * TOx + ox], acc);
      }
      yp[gy * Wo + gx] = static_cast<T>(gain_out * acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Generic two-pass path for non-separable (2D) filters.
// Filters are packed as [wu (FUY x FUX), wd (FDY x FDX)], row-major.
// ---------------------------------------------------------------------------

// Pass 1: bias, upsample, filter with wu, activation. Writes the float intermediate z.
template <typename T>
kernel void cflrelu_up2d(
    device const T* x [[buffer(0)]],
    device const T* bias [[buffer(1)]],
    constant float* filt [[buffer(2)]],
    device uchar* mask [[buffer(3)]],
    device float* z [[buffer(4)]],
    constant int* ip [[buffer(5)]],
    constant float* fp [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int W = ip[P_W], H = ip[P_H], U = ip[P_UP];
  const int LZx = ip[P_LZX], LZy = ip[P_LZY];
  const bool zero_pad = ip[P_ZERO_PAD] != 0;
  const int kx = int(gid.x), ky = int(gid.y), nc = int(gid.z);
  if (kx >= LZx || ky >= LZy) {
    return;
  }
  const int FUx = ip[P_FUX], FUy = ip[P_FUY], OUx = ip[P_OUX], OUy = ip[P_OUY];
  const int mode = ip[P_MODE];
  const float bval = ip[P_HAS_BIAS] ? float(bias[nc % ip[P_C]]) : 0.0f;

  device const T* xp = x + ulong(nc) * ulong(H) * ulong(W);
  const int my = ky - OUy, mx = kx - OUx;
  const int ty0 = pmod(-my, U), tx0 = pmod(-mx, U);
  // my + ty0 and mx + tx0 are multiples of U, so these divisions are exact.
  const int iy0 = (my + ty0) / U, ix0 = (mx + tx0) / U;

  float acc = 0.0f;
  int iy = zero_pad ? iy0 : pmod(iy0, H);
  for (int ty = ty0; ty < FUy; ty += U, ++iy) {
    if (!zero_pad && iy == H) {
      iy = 0;
    }
    if (iy < 0 || iy >= H) {
      continue;  // zero padding
    }
    int ix = zero_pad ? ix0 : pmod(ix0, W);
    for (int tx = tx0; tx < FUx; tx += U, ++ix) {
      if (!zero_pad && ix == W) {
        ix = 0;
      }
      if (ix < 0 || ix >= W) {
        continue;
      }
      acc = fma(filt[ty * FUx + tx], float(xp[iy * W + ix]) + bval, acc);
    }
  }

  const ulong plane = ulong(LZy) * ulong(LZx);
  const ulong kidx = ulong(ky) * ulong(LZx) + ulong(kx);
  z[ulong(nc) * plane + kidx] = act_or_gate(
      fp[F_SCALE] * acc, mode, fp[F_SLOPE], fp[F_CLAMP], mask + ulong(nc) * plane, kidx, true);
}

// Pass 2: filter with wd and downsample.
template <typename T>
kernel void cflrelu_down2d(
    device const float* z [[buffer(0)]],
    constant float* filt [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant int* ip [[buffer(3)]],
    constant float* fp [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int D = ip[P_DOWN];
  const int Wo = ip[P_WO], Ho = ip[P_HO];
  const int LZx = ip[P_LZX], LZy = ip[P_LZY];
  const bool zero_pad = ip[P_ZERO_PAD] != 0;
  const int ox = int(gid.x), oy = int(gid.y), nc = int(gid.z);
  if (ox >= Wo || oy >= Ho) {
    return;
  }
  const int FUx = ip[P_FUX], FUy = ip[P_FUY];
  const int FDx = ip[P_FDX], FDy = ip[P_FDY], ODx = ip[P_ODX], ODy = ip[P_ODY];
  constant float* wd = filt + FUx * FUy;

  device const float* zp = z + ulong(nc) * ulong(LZy) * ulong(LZx);
  float acc = 0.0f;
  const int zy0 = D * oy - ODy, zx0 = D * ox - ODx;
  int zy = zero_pad ? zy0 : pmod(zy0, LZy);
  for (int ty = 0; ty < FDy; ++ty, ++zy) {
    if (!zero_pad && zy == LZy) {
      zy = 0;
    }
    if (zy < 0 || zy >= LZy) {
      continue;  // zero padding: the intermediate is zero outside [0, LZ)
    }
    int zx = zero_pad ? zx0 : pmod(zx0, LZx);
    for (int tx = 0; tx < FDx; ++tx, ++zx) {
      if (!zero_pad && zx == LZx) {
        zx = 0;
      }
      if (zx < 0 || zx >= LZx) {
        continue;
      }
      acc = fma(wd[ty * FDx + tx], zp[zy * LZx + zx], acc);
    }
  }
  y[ulong(nc) * ulong(Ho) * ulong(Wo) + ulong(oy) * ulong(Wo) + ulong(ox)] =
      static_cast<T>(fp[F_GAIN_OUT] * acc);
}

// ---------------------------------------------------------------------------
// Instantiations.
// ---------------------------------------------------------------------------

#define INSTANTIATE_SEP(T, NAME, MODE)                                  \
  template [[host_name("cflrelu_sep_" #NAME "_" #MODE)]] kernel void    \
  cflrelu_sep<T, MODE>(                                                 \
      device const T* x [[buffer(0)]],                                  \
      device const T* bias [[buffer(1)]],                               \
      constant float* filt [[buffer(2)]],                               \
      device uchar* mask [[buffer(3)]],                                 \
      device T* y [[buffer(4)]],                                        \
      constant int* ip [[buffer(5)]],                                   \
      constant float* fp [[buffer(6)]],                                 \
      uint3 tg [[threadgroup_position_in_grid]],                        \
      uint3 lid [[thread_position_in_threadgroup]]);

#define INSTANTIATE_CFLRELU(T, NAME)                                    \
  INSTANTIATE_SEP(T, NAME, 0)                                           \
  INSTANTIATE_SEP(T, NAME, 1)                                           \
  INSTANTIATE_SEP(T, NAME, 2)                                           \
  template [[host_name("cflrelu_up2d_" #NAME)]] kernel void             \
  cflrelu_up2d<T>(                                                      \
      device const T* x [[buffer(0)]],                                  \
      device const T* bias [[buffer(1)]],                               \
      constant float* filt [[buffer(2)]],                               \
      device uchar* mask [[buffer(3)]],                                 \
      device float* z [[buffer(4)]],                                    \
      constant int* ip [[buffer(5)]],                                   \
      constant float* fp [[buffer(6)]],                                 \
      uint3 gid [[thread_position_in_grid]]);                           \
  template [[host_name("cflrelu_down2d_" #NAME)]] kernel void           \
  cflrelu_down2d<T>(                                                    \
      device const float* z [[buffer(0)]],                              \
      constant float* filt [[buffer(1)]],                               \
      device T* y [[buffer(2)]],                                        \
      constant int* ip [[buffer(3)]],                                   \
      constant float* fp [[buffer(4)]],                                 \
      uint3 gid [[thread_position_in_grid]]);

INSTANTIATE_CFLRELU(float, float)
INSTANTIATE_CFLRELU(half, half)
#if __METAL_VERSION__ >= 310
INSTANTIATE_CFLRELU(bfloat, bfloat)
#endif
