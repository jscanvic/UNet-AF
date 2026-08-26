"""EQ metric adapted to use ContinuousTranslation instead of generic transforms.

This module provides EQ which is a Metric wrapper that applies translations to the model
by symmetrizing it with respect to translations and averages a base Metric over samples.
"""

import deepinv as dinv
import torch
import torchvision.transforms.functional as TF

from .continuous_translation import ContinuousTranslation


class SincLowPassFilter:
    """Separable sinc/Dirichlet low-pass filter"""

    def __init__(self, cutoff: float, critical: bool = True):
        if cutoff < 0 or cutoff > 0.5:
            raise ValueError(f"Invalid cutoff value {cutoff}, should be in [0, 0.5]")
        self.cutoff = float(cutoff)
        self.critical = bool(critical)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input of shape (B, C, H, W); got {tuple(x.shape)}"
            )
        s = x.shape[-2:]
        H, W = s
        freq_h = torch.fft.fftfreq(H, d=1.0, device=x.device)
        freq_w = torch.fft.rfftfreq(W, d=1.0, device=x.device)
        fh, fw = torch.meshgrid(freq_h, freq_w, indexing="ij")
        radius = torch.maximum(fh.abs(), fw.abs())
        passband = radius <= self.cutoff if self.critical else radius < self.cutoff

        spectrum = torch.fft.rfft2(x, dim=(-2, -1))
        spectrum = passband * spectrum
        return torch.fft.irfft2(spectrum, s=s, dim=(-2, -1))


class Residue(dinv.models.Reconstructor):
    def __init__(
        self,
        backbone: dinv.models.Reconstructor,
        residual_mode: str,
    ):
        super().__init__()
        self.backbone = backbone
        self._residual_mode = residual_mode

    def forward(self, y, physics, **kwargs):
        mode = self._residual_mode
        if mode == "adjoint":
            x0 = physics.A_adjoint(y)
        elif mode == "pinv":
            x0 = physics.A_dagger(y)
        elif mode == "direct":
            x0 = y
        else:
            raise ValueError(f"Unsupported backbone mode '{mode}' for Residue")
        return self.backbone(y, physics, **kwargs) - x0


class EQUIV(dinv.loss.Metric):
    def __init__(
        self,
        *,
        translation: ContinuousTranslation,
        metric: dinv.loss.Metric,
        n_samples: int = 5,
        db: bool = False,
        n_trans: int = 1,
        ignore_nyquist: bool = False,
        residue: bool = False,
        residual_mode: str | None = None,
        translation_mode: str = "circular",
        crop_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._translation = translation
        self._metric_base = metric
        self._n_samples = int(n_samples)
        self._db = db
        # The number of translations per sample; keep compatibility with previous Transform.n_trans
        self.n_trans = int(n_trans)
        self._ignore_nyquist = bool(ignore_nyquist)
        self._nyquist_filter = SincLowPassFilter(cutoff=0.5, critical=False)
        self._residue = bool(residue)
        self._residual_mode = residual_mode
        self._translation_mode = translation_mode
        self._crop_size = crop_size

    def metric(self, *args, **kwargs):
        # delegate to the static implementation to keep compatibility with base signature
        return self._metric_wrapper(*args, **kwargs)

    def _metric_wrapper(self, x, x_net, y, physics, model, **kwargs):
        return self._metric_fn(
            y=y,
            x_net=x_net,
            model=model,
            translation=self._translation,
            physics=physics,
            metric=self._metric_base,
            n_samples=self._n_samples,
            db=self._db,
            n_trans=self.n_trans,
            ignore_nyquist=self._ignore_nyquist,
            lpf=self._nyquist_filter,
            residue=self._residue,
            residual_mode=self._residual_mode,
            translation_mode=self._translation_mode,
            crop_size=self._crop_size,
        )

    @staticmethod
    def _metric_fn(
        *,
        y: torch.Tensor,
        x_net: torch.Tensor,
        model: dinv.models.Reconstructor,
        translation: ContinuousTranslation,
        physics: dinv.physics.LinearPhysics,
        metric: dinv.loss.Metric,
        n_samples: int,
        db: bool,
        n_trans: int,
        ignore_nyquist: bool,
        lpf: SincLowPassFilter,
        residue: bool,
        residual_mode: str | None,
        translation_mode: str,
        crop_size=None,
    ):
        if n_trans != 1:
            raise ValueError(f"Unsupported value of n_trans, {n_trans}, should be 1")

        if ignore_nyquist:
            y = lpf(y)

        if residue:
            if residual_mode is not None:
                resolved_mode = residual_mode
            elif isinstance(model, dinv.models.ArtifactRemoval):
                resolved_mode = model.mode
            else:
                raise ValueError(
                    "Residue requested without residual_mode requires an ArtifactRemoval backbone; "
                    f"got {type(model)}"
                )
            model = Residue(model, residual_mode=resolved_mode)

        base_out = model(y, physics)

        out = None
        for _ in range(n_samples):
            device = y.device
            max_disp = 8
            dh = int(torch.randint(-max_disp, max_disp + 1, (1,), device=device).item())
            dw = int(torch.randint(-max_disp, max_disp + 1, (1,), device=device).item())
            displacement = (dh, dw)

            if translation_mode == "circular":

                def translate_fn(t):
                    return translation.translate(t, displacement)

            elif translation_mode == "crop_shift":

                def translate_fn(t, displacement=displacement):
                    H_full, W_full = t.shape[-2:]
                    shifted = torch.roll(t, shifts=displacement, dims=(-2, -1))
                    cropped = TF.center_crop(shifted, crop_size)
                    dh, dw = displacement
                    H_crop, W_crop = cropped.shape[-2:]
                    overlap_h = max(0, abs(dh) - (H_full - H_crop) // 2)
                    overlap_w = max(0, abs(dw) - (W_full - W_crop) // 2)
                    if dw > 0:
                        cropped = cropped[..., :, overlap_w:]
                    elif dw < 0:
                        cropped = cropped[..., :, : W_crop - overlap_w]
                    if dh > 0:
                        cropped = cropped[..., overlap_h:, :]
                    elif dh < 0:
                        cropped = cropped[..., : H_crop - overlap_h, :]
                    return cropped

            else:
                raise ValueError(f"Unsupported translation_mode '{translation_mode}'")

            translated_in = translate_fn(y)
            out_from_translated_input = model(translated_in, physics)
            translated_output = translate_fn(base_out)

            # Use a crop-translation to get rid of the boundary pixels
            # for comparison
            if translation_mode == "crop_shift":
                # Even though it looks like a translation, it is actually just a crop
                out_from_translated_input = translate_fn(
                    out_from_translated_input,
                    displacement=(-displacement[0], -displacement[1]),
                )

            # metric(actual, reference): actual is model(translate(y)), reference is translate(model(y))
            values = metric(out_from_translated_input, translated_output)
            out = values if out is None else out + values
        out = out / float(n_samples)
        if db:
            out = -10 * torch.log10(out)
        return out
