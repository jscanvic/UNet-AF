// Metal implementation of the upfirdn2d plugin for Apple Silicon (PyTorch MPS).
//
// For each output pixel, directly evaluates
//   y[o] = gain * sum_t w[t] * u[o * down - pad0 + t],   u[m] = x[m / up] if m % up == 0 else 0
// along both axes, where w are correlation weights (the host flips the filter when
// flip=False) and u is zero outside the input. Only taps that hit nonzero samples
// of the upsampled signal are visited (polyphase).
//
// Parameter layout is shared with upfirdn2d_mps.py.

#include <metal_stdlib>
using namespace metal;

enum : int {
  P_W = 0,  // input width
  P_H,      // input height
  P_UPX,
  P_UPY,
  P_DOWNX,
  P_DOWNY,
  P_PADX0,  // left padding of the upsampled image (negative crops)
  P_PADY0,  // top padding
  P_FW,     // filter width
  P_FH,     // filter height
  P_WO,     // output width
  P_HO,     // output height
  P_COUNT
};

enum : int { F_GAIN = 0, F_COUNT };

inline int pmod(int a, int n) {
  int r = a % n;
  return r < 0 ? r + n : r;
}

template <typename T>
kernel void upfirdn2d(
    device const T* x [[buffer(0)]],
    constant float* w [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant int* ip [[buffer(3)]],
    constant float* fp [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int Wo = ip[P_WO], Ho = ip[P_HO];
  const int ox = int(gid.x), oy = int(gid.y), nc = int(gid.z);
  if (ox >= Wo || oy >= Ho) {
    return;
  }
  const int W = ip[P_W], H = ip[P_H];
  const int UX = ip[P_UPX], UY = ip[P_UPY];
  const int FW = ip[P_FW], FH = ip[P_FH];

  // First tap hitting a nonzero sample, and the input index it reads.
  const int my = oy * ip[P_DOWNY] - ip[P_PADY0];
  const int mx = ox * ip[P_DOWNX] - ip[P_PADX0];
  const int ty0 = pmod(-my, UY), tx0 = pmod(-mx, UX);
  const int iy0 = (my + ty0) / UY, ix0 = (mx + tx0) / UX;  // exact divisions

  device const T* xp = x + ulong(nc) * ulong(H) * ulong(W);
  float acc = 0.0f;
  int iy = iy0;
  for (int ty = ty0; ty < FH && iy < H; ty += UY, ++iy) {
    if (iy < 0) {
      continue;
    }
    device const T* xrow = xp + iy * W;
    constant float* wrow = w + ty * FW;
    int ix = ix0;
    for (int tx = tx0; tx < FW && ix < W; tx += UX, ++ix) {
      if (ix >= 0) {
        acc = fma(wrow[tx], float(xrow[ix]), acc);
      }
    }
  }
  y[ulong(nc) * ulong(Ho) * ulong(Wo) + ulong(oy) * ulong(Wo) + ulong(ox)] =
      static_cast<T>(fp[F_GAIN] * acc);
}

#define INSTANTIATE_UPFIRDN2D(T, NAME)                        \
  template [[host_name("upfirdn2d_" #NAME)]] kernel void      \
  upfirdn2d<T>(                                               \
      device const T* x [[buffer(0)]],                        \
      constant float* w [[buffer(1)]],                        \
      device T* y [[buffer(2)]],                              \
      constant int* ip [[buffer(3)]],                         \
      constant float* fp [[buffer(4)]],                       \
      uint3 gid [[thread_position_in_grid]]);

INSTANTIATE_UPFIRDN2D(float, float)
INSTANTIATE_UPFIRDN2D(half, half)
#if __METAL_VERSION__ >= 310
INSTANTIATE_UPFIRDN2D(bfloat, bfloat)
#endif
