import numpy as np
import cv2 
from colorspace import colorspace 
from normalized import normalized
from skimage import morphology,segmentation
from scipy.ndimage import sobel

def sgrad_edge(f):
    f = f.astype(np.double)
    gs = sobel(f, axis=0)
    gs2 = sobel(f, axis=1)
    return normalized(np.sqrt(gs ** 2 + gs2 ** 2))

def w_recons_CO(f, se):
    fe = morphology.erosion(f, se)
    fobr = morphology.reconstruction(fe, f, method='dilation')
    fobrc = np.max(f) - fobr
    fobrce = morphology.erosion(fobrc, se)
    fobrcbr = np.max(f) - morphology.reconstruction(fobrce, fobrc, method='dilation')
    return fobrcbr


def w_MMGR_WT(f, se_start):
    max_itr = 50
    min_impro = 0.0001

    sigma = 1.0
    g = cv2.GaussianBlur(f, (5,5), sigma)

    gg = colorspace('Lab<-RGB',g)
    a1 = sgrad_edge(normalized(gg[:,:,0]))**2
    b1 = sgrad_edge(np.abs(normalized(gg[:,:,1])))**2
    c1 = sgrad_edge(normalized(gg[:,:,2]))**2
    ngrad_f1 = np.sqrt(a1 + b1 + c1)

    f_g = np.zeros(f.shape[:2])
    diff = np.zeros(max_itr)
    for i in range(max_itr):
        se = morphology.disk(i + se_start - 1)
        gx = w_recons_CO(ngrad_f1, se)

        f_g2 = np.maximum(f_g, gx)
        f_g1 = f_g
        f_g = f_g2
        diff[i] = np.mean(np.abs(f_g1 - f_g2))
        if i > 0 and diff[i] < min_impro:
            break
    L_seg = segmentation.watershed(f_g)
    return L_seg, i+1, diff[:i+1]

def w_super_fcm(L1, data, Label_n, cluster_n, options=None):
    if options is None:
        options = [2, 50, 1e-5, 1]
    else:
        if len(options) < 4:
            options = options + [2, 50, 1e-5, 1][len(options):]
        if options[0] <= 1:
            raise ValueError('The exponent should be greater than 1!')

    expo, max_iter, min_impro, display = options

    data_n = data.shape[0]  

    U = initfcm(cluster_n, data_n)

    Num = np.ones((cluster_n, 1)) * Label_n

    Uc = []
    for i in range(max_iter):
        mf = Num * (U ** expo)  
        center = np.dot(mf, data) / np.sum(mf, axis=1)[:, np.newaxis] 

        out = np.zeros((center.shape[0], data.shape[0]))
        for k in range(center.shape[0]):
            out[k, :] = np.sqrt(np.sum((data - center[k]) ** 2, axis=1))

        dist = out + np.finfo(float).eps
        tmp = dist ** (-2 / (expo - 1))
        U = tmp / (np.sum(tmp, axis=0) + np.finfo(float).eps)

        Uc.append(U)

        if i > 0:
            if np.abs(np.max(Uc[i] - Uc[i-1])) < min_impro:
                break

    iter_n = i + 1
    center_Lab = center

    IDX2 = np.argmax(U, axis=0)

    Lr2 = np.zeros(L1.shape)
    for i in range(1, np.max(L1) + 1):
        Lr2[L1 == i] = IDX2[i-1]

    return Lr2, center_Lab, U, iter_n


def initfcm(cluster_n, data_n):
    U = np.random.rand(cluster_n, data_n)
    col_sum = np.sum(U, axis=0)
    return U / col_sum
