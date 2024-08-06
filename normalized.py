import numpy as np

def g_normalized(a):
    a = a.astype(np.double)
    a_min = np.min(a)
    g = a - a_min
    g_max = np.max(g)
    g = g / g_max
    return g

def normalized(a):
    if a.ndim < 3:
        g = g_normalized(a)
    else:
        f1 = a[:,:,0]
        f2 = a[:,:,1]
        f3 = a[:,:,2]
        ff1 = g_normalized(f1)
        ff2 = g_normalized(f2)
        ff3 = g_normalized(f3)
        g = np.stack([ff1, ff2, ff3], 2)
    return g


