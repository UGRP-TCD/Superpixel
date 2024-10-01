import numpy as np
import fslic
from skimage import io, color
# from skimage.segmentation import mark_boundaries
from skimage.transform import resize
import matplotlib.pyplot as plt

img = io.imread('horses.jpg')

downscale_factor = 0.3
img = resize(img, (int(img.shape[0] * downscale_factor), int(img.shape[1] * downscale_factor)),anti_aliasing=True)

img_lab = color.rgb2lab(img)
h, w, d = img_lab.shape
img_flat = img_lab.reshape(-1).tolist()

clusters = 80
compactness = 100

result = np.array(fslic.fslic(img_flat, w, h, d, clusters, compactness, 10, 1, 1))

result = result.reshape(h, w, d)

result = color.lab2rgb(result)

# boundaries = mark_boundaries(result_rgb, result[:,:,0].astype(int),color=(0, 0, 0))

plt.imshow(result)

plt.axis('off')

plt.show()

io.imsave('fslic_result.png', (result * 255).astype(np.uint8))

