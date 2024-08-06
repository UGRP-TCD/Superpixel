import numpy as np
from skimage import color

def label_image(f, L):
    f = f.astype(np.float64)
    
    L = np.asarray(L).astype(int)
    if L.ndim != 2:
        raise ValueError("Label image L must be a 2D array")

    num_area = np.max(L)
    Num_p = np.zeros(num_area, dtype=int)

    M, N = f.shape[:2]
    fs = np.zeros((M, N, 3) if f.ndim == 3 else (M, N))
    center_p = np.zeros((num_area, 3) if f.ndim == 3 else (num_area,))

    for i in range(1, num_area + 1):
        mask = (L == i)
        if np.any(mask):
            if f.ndim == 3:
                for c in range(3):
                    f2 = f[:,:,c][mask]
                    if f2.size > 0:
                        med = np.median(f2)
                        fs[:,:,c][mask] = med
                        center_p[i-1, c] = med
            else:
                f2 = f[mask]
                if f2.size > 0:
                    med = np.median(f2)
                    fs[mask] = med
                    center_p[i-1] = med
            Num_p[i-1] = np.sum(mask)

    fs = np.clip(fs, 0, 255).astype(np.uint8)
    center_p = np.clip(center_p, 0, 255).astype(np.uint8)

    if center_p.shape[1] == 3:
        center_lab = color.rgb2lab(center_p[np.newaxis, :, :])
        center_lab = center_lab.reshape(-1, 3)
    else:
        center_lab = center_p
    return fs, center_p, Num_p, center_lab