// Metal port of bias_act.cu for Apple Silicon (PyTorch MPS).
//
// Elementwise over the raw storage of x, so any non-overlapping dense layout works
// (the bias index uses the stride of `dim`, as in CUDA). grad = 0 evaluates the
// activation, 1 its first derivative and 2 its second derivative, using the same
// xref/yref conventions as the CUDA kernel.
//
// Parameter layout is shared with bias_act_mps.py.

#include <metal_stdlib>
using namespace metal;

enum : int {
  P_SIZEX = 0,
  P_SIZEB,
  P_STEPB,
  P_GRAD,
  P_HAS_B,
  P_HAS_XREF,
  P_HAS_YREF,
  P_HAS_DY,
  P_COUNT
};

enum : int { F_ALPHA = 0, F_GAIN, F_CLAMP, F_COUNT };

template <typename T, int A>
kernel void bias_act(
    device const T* px [[buffer(0)]],
    device const T* pb [[buffer(1)]],
    device const T* pxref [[buffer(2)]],
    device const T* pyref [[buffer(3)]],
    device const T* pdy [[buffer(4)]],
    device T* py [[buffer(5)]],
    constant int* ip [[buffer(6)]],
    constant float* fp [[buffer(7)]],
    uint xi [[thread_position_in_grid]]) {
  if (int(xi) >= ip[P_SIZEX]) {
    return;
  }
  const int G = ip[P_GRAD];
  const float alpha = fp[F_ALPHA];
  const float gain = fp[F_GAIN];
  const float clampv = fp[F_CLAMP];
  const float one = 1.0f;
  const float two = 2.0f;
  const float expRange = 80.0f;
  const float halfExpRange = 40.0f;
  const float seluScale = 1.0507009873554804934193349852946f;
  const float seluAlpha = 1.6732632423543772848170429916717f;

  // Load.
  float x = float(px[xi]);
  const float b = ip[P_HAS_B] ? float(pb[(int(xi) / ip[P_STEPB]) % ip[P_SIZEB]]) : 0.0f;
  float xref = ip[P_HAS_XREF] ? float(pxref[xi]) : 0.0f;
  float yref = ip[P_HAS_YREF] ? float(pyref[xi]) : 0.0f;
  const float dy = ip[P_HAS_DY] ? float(pdy[xi]) : one;
  const float yy = (gain != 0.0f) ? yref / gain : 0.0f;
  float y = 0.0f;

  // Apply bias.
  if (G == 0) {
    x += b;
  } else {
    xref += b;
  }

  if (A == 1) {  // linear
    if (G == 0) y = x;
    if (G == 1) y = x;
  }
  if (A == 2) {  // relu
    if (G == 0) y = (x > 0) ? x : 0;
    if (G == 1) y = (yy > 0) ? x : 0;
  }
  if (A == 3) {  // lrelu
    if (G == 0) y = (x > 0) ? x : x * alpha;
    if (G == 1) y = (yy > 0) ? x : x * alpha;
  }
  if (A == 4) {  // tanh
    if (G == 0) {
      const float c = exp(x);
      const float d = one / c;
      y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d);
    }
    if (G == 1) y = x * (one - yy * yy);
    if (G == 2) y = x * (one - yy * yy) * (-two * yy);
  }
  if (A == 5) {  // sigmoid
    if (G == 0) y = (x < -expRange) ? 0 : one / (exp(-x) + one);
    if (G == 1) y = x * yy * (one - yy);
    if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
  }
  if (A == 6) {  // elu
    if (G == 0) y = (x >= 0) ? x : exp(x) - one;
    if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
    if (G == 2) y = (yy >= 0) ? 0 : x * (yy + one);
  }
  if (A == 7) {  // selu
    if (G == 0) y = (x >= 0) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
    if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
    if (G == 2) y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
  }
  if (A == 8) {  // softplus
    if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
    if (G == 1) y = x * (one - exp(-yy));
    if (G == 2) {
      const float c = exp(-yy);
      y = x * c * (one - c);
    }
  }
  if (A == 9) {  // swish
    if (G == 0) {
      y = (x < -expRange) ? 0 : x / (exp(-x) + one);
    } else {
      const float c = exp(xref);
      const float d = c + one;
      if (G == 1) {
        y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
      } else {
        y = (xref > halfExpRange) ? 0 : x * c * (xref * (two - d) + two * d) / (d * d * d);
      }
      yref = (xref < -expRange) ? 0 : xref / (exp(-xref) + one) * gain;
    }
  }

  // Apply gain.
  y *= gain * dy;

  // Clamp.
  if (clampv >= 0) {
    if (G == 0) {
      y = (y > -clampv && y < clampv) ? y : (y >= 0) ? clampv : -clampv;
    } else {
      y = (yref > -clampv && yref < clampv) ? y : 0;
    }
  }

  py[xi] = static_cast<T>(y);
}

#define INSTANTIATE_BIAS_ACT_ONE(T, NAME, A)                  \
  template [[host_name("bias_act_" #NAME "_" #A)]] kernel void \
  bias_act<T, A>(                                              \
      device const T* px [[buffer(0)]],                        \
      device const T* pb [[buffer(1)]],                        \
      device const T* pxref [[buffer(2)]],                     \
      device const T* pyref [[buffer(3)]],                     \
      device const T* pdy [[buffer(4)]],                       \
      device T* py [[buffer(5)]],                              \
      constant int* ip [[buffer(6)]],                          \
      constant float* fp [[buffer(7)]],                        \
      uint xi [[thread_position_in_grid]]);

#define INSTANTIATE_BIAS_ACT(T, NAME)    \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 1)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 2)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 3)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 4)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 5)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 6)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 7)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 8)   \
  INSTANTIATE_BIAS_ACT_ONE(T, NAME, 9)

INSTANTIATE_BIAS_ACT(float, float)
INSTANTIATE_BIAS_ACT(half, half)
#if __METAL_VERSION__ >= 310
INSTANTIATE_BIAS_ACT(bfloat, bfloat)
#endif
