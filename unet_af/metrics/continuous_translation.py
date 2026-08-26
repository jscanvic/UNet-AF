"""Continuous subpixel translations using real DFT phase shifts.

Provides ContinuousTranslation which applies a translation to a batch of images using the
real DFT (torch.rfft/torch.fft) phase shift formula. Uses the real part (cosine) of the
phase-shift at Nyquist frequencies for even-sized dimensions so integer shifts are better supported.

The implementation works on (B, C, H, W) tensors and returns translated tensors of the same shape.
"""

import torch


class ContinuousTranslation:
    """Apply continuous (possibly subpixel) translation to images using DFT phase shifts."""

    def translate(
        self, x: torch.Tensor, displacement: tuple[float, float]
    ) -> torch.Tensor:
        """Translate input tensor x by displacement=(disp_h, disp_w).

        Args:
            x: (B, C, H, W) tensor
            displacement: (dh, dw) translation in pixels. Can be fractional and negative.
        Returns:
            translated tensor of same shape and dtype as x
        """
        if x.dim() != 4:
            raise ValueError("x must be a 4D tensor (B, C, H, W)")
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Move to the frequency domain
        s = (H, W)
        x = torch.fft.rfft2(x, dim=(-2, -1))

        # Shift the phase
        fh = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)
        fw = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype)
        fh, fw = torch.meshgrid(
            fh, fw, indexing="ij"
        )  # frequencies in Hz normalized between -0.5 and 0.5
        dh, dw = displacement  #  in px
        x = x * torch.exp(-1j * 2 * torch.pi * (fw * dw + fh * dh))

        # Get back to the spatial domain
        return torch.fft.irfft2(x, dim=(-2, -1), s=s)


# small helper to expose a functional API
def translate(x: torch.Tensor, displacement: tuple[float, float]) -> torch.Tensor:
    return ContinuousTranslation().translate(x, displacement)
