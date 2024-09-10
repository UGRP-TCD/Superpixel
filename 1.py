import numpy as np
import fslic
from skimage import io, color
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


img = io.imread('horses.jpg')
img_lab = color.rgb2lab(img)


h, w, d = img_lab.shape
img_flat = img_lab.reshape(-1).tolist()

result = np.array(fslic.fslic(img_flat, w, h, d, 200, 17, 10, 1, 1))

# Reshape the result
result = result.reshape(h, w, d)

# Convert back to RGB for visualization
result_rgb = color.lab2rgb(result)


boundaries = mark_boundaries(result_rgb, result[:,:,0].astype(int))

plt.imshow(boundaries)
plt.axis('off')
plt.show()

# Save the result
io.imsave('fslic_result.png', (boundaries * 255).astype(np.uint8))