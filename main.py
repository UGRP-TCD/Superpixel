import time
import matplotlib.pyplot as plt
from skimage import io, morphology, color
from segmentation_utils import w_MMGR_WT, w_super_fcm
from label_image import label_image


if __name__ == "__main__":
    st = time.time()
    cluster = 2
    f_ori = io.imread('12003.jpg')

    SE = 3
    L1, _, _ = w_MMGR_WT(f_ori, SE)
    L2 = morphology.dilation(L1, morphology.square(2))
    L2 = L2.astype(int)
    if L2.ndim != 2:
        raise ValueError("L2 must be a 2D array")

    _, _, Num, centerLab = label_image(f_ori, L2)

    Lr2, center_Lab, U, iter_n = w_super_fcm(L2, centerLab, Num, cluster)

    Label = Lr2.astype(int)
    center_rgb = color.lab2rgb(center_Lab.reshape(1, cluster, 3)).reshape(cluster, 3)
    cluster_colors = center_rgb
    colored_label = cluster_colors[Label]

    et = time.time()
    print("Time: %0.1fs" %(et - st))
    plt.subplot(121)
    plt.imshow(f_ori)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(colored_label)
    plt.title('Clustered Image')
    plt.axis('off')
    plt.show()