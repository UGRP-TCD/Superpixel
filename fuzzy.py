import os
import cv2
import numpy as np
from ctypes import CDLL, c_int, c_double, POINTER, py_object
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import mark_boundaries
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def show_inplace(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def cluster_superpixels(superpixels, n_clusters=10):
    # Reshape superpixels to 2D array
    pixels = superpixels.reshape((-1, 3))
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image shape
    return labels.reshape(superpixels.shape[:2])

def color_regions(clustered, original_img):
    h, w = clustered.shape
    result = np.zeros_like(original_img)
    
    for label in np.unique(clustered):
        mask = clustered == label
        color = np.mean(original_img[mask], axis=0)
        result[mask] = color
    
    return result


def r():
    return random.random()

current_dir = os.getcwd()
shared_lib_path = os.path.join(current_dir, 'shared.so')
shared_file = CDLL(shared_lib_path)

shared_file.fslic.argtypes = [py_object, c_int, c_int, c_int, c_int, c_double, c_int, c_double, c_double]
shared_file.fslic.restype = POINTER(c_double)

img_path = 'horses.jpg'

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, depth = img.shape
print(w, 'x', h)
show_inplace(img)

probability = 0.2
for y in range(h):
    for x in range(w):
        if r() <= probability:
            img[y][x][0] = int(r() * 255)
            img[y][x][1] = int(r() * 255)
            img[y][x][2] = int(r() * 255)
show_inplace(img)

img = img.astype(np.float64) / 255
lab = rgb2lab(img)

# Prepare arguments for fslic
lab_list = lab.flatten().tolist()
py_list = py_object(lab_list)
m = 200  # initial_num_clusters
compactness = 17.0
max_iterations = 10
p = 1.0  # These parameters are not in your original code, but they're in the C function
q = 1.0  # You may need to adjust these values

# Run Fuzzy SLIC
result_ptr = shared_file.fslic(py_list, w, h, depth, m, compactness, max_iterations, p, q)

# Convert result back to numpy array
superpixels = np.ctypeslib.as_array(result_ptr, shape=(h * w * depth,))
superpixels = superpixels.reshape((h, w, depth))

# Free the memory allocated in C
shared_file.free_1d_d(result_ptr)

print("Number of superpixels:", len(np.unique(superpixels.reshape((h*w, 3)), axis=0)))

# Convert result back to RGB and display result with contours
result = lab2rgb(superpixels)
unique_labels = np.unique(result.reshape((-1, 3)), axis=0, return_inverse=True)[1].reshape((h, w))
contours = mark_boundaries(result, unique_labels, color=[1,1,1])
show_inplace(contours)
cv2.imwrite('fslic.jpg', cv2.cvtColor((255 * contours).astype(np.uint8), cv2.COLOR_RGB2BGR))


# Cluster superpixels
clustered = cluster_superpixels(superpixels)

# Color regions based on clustering
colored = color_regions(clustered, img)

# Display the result
show_inplace(colored)

# Save the result
cv2.imwrite('output.jpg', cv2.cvtColor((255 * colored).astype(np.uint8), cv2.COLOR_RGB2BGR))

