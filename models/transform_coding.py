from typing import Optional, Tuple

import numpy as np
import torch

from models.quantization.mu_law import manual_mu_law_decode, manual_mu_law_encode


class TransformCoding:
    def __init__(self, encoded_dim: int, img_shape: Optional[Tuple[int, int, int]] = None, nbr_bits: int = 32):
        """Simple transform coding using non-linear quantization."""
        if img_shape is None:
            img_shape = (2, 32, 32)
        self.img_shape = img_shape
        self.encoded_dim = np.prod(img_shape) / encoded_dim
        self.nbr_channels, self.nbr_rec, self.nbr_trans = img_shape
        self.quantization_channels = nbr_bits ** 2
        # We need log2(32x32) = 10 bits to encode the position.
        self.coeffs_used = int(self.encoded_dim * nbr_bits // (10 + nbr_bits * 2))
        assert self.nbr_channels == 2, 'We assume two image channels corresponding to real and imaginary parts.'

    def __call__(self, x: torch.Tensor):
        """Note the input is assumed to be in the sparse domain already."""
        x = x.detach().numpy() - 0.5  # De-centralize
        if x.ndim == 4:
            out = []
            for i, xx in enumerate(x):
                idx, yr, yi = self.encoder(xx)
                xx_rec = self.decoder(idx, yr, yi)
                out.append(xx_rec)
            return torch.from_numpy(np.array(out) + 0.5)
        else:
            raise NotImplementedError('Only support for batch processing right now')

    def encoder(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pick the strongest coefficients (in terms of absolute value)."""
        xc = x[0].flatten() + 1j * x[1].flatten()
        idx = np.argpartition(np.abs(xc.flatten()), -self.coeffs_used)[-self.coeffs_used:]
        return idx, manual_mu_law_encode(xc.real[idx], self.quantization_channels), \
            manual_mu_law_encode(xc.imag[idx], self.quantization_channels)

    def decoder(self, idx: np.ndarray, y_real: np.ndarray, y_imag: np.ndarray) -> np.ndarray:
        """Decode signal."""
        s = np.zeros(self.nbr_rec * self.nbr_trans, dtype=np.complex128)
        s[idx] = manual_mu_law_decode(y_real, self.quantization_channels) + \
            1j * manual_mu_law_decode(y_imag, self.quantization_channels)
        return np.dstack(
            (s.real.reshape((self.nbr_rec, self.nbr_trans)), s.imag.reshape((self.nbr_rec, self.nbr_trans)))
        ).transpose((2, 0, 1))

    def eval(self):
        """Dummy function needed for compatibility with other methods."""
        pass

    def to(self, dummy):
        """Dummy function needed for compatibility with other methods."""
        pass
