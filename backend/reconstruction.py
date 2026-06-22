import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

PEAK_Y = 509
PEAK_X = 1535
DX = 50
DY = 50

def reconstruct_from_hologram(im, peak_y=PEAK_Y, peak_x=PEAK_X, dx=DX, dy=DY):

    im = im.astype(np.float32)

    F = fftshift(fft2(im))

    H, W = im.shape
    cy, cx = H // 2, W // 2

    patch = F[peak_y-dy:peak_y+dy, peak_x-dx:peak_x+dx]

    F_crop = np.zeros_like(F)
    F_crop[cy-dy:cy+dy, cx-dx:cx+dx] = patch

    field = ifft2(ifftshift(F_crop))

    return field
