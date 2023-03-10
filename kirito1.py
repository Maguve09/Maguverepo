import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the Robinson Compass Mask kernels for each direction
kernels = [
    # East
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    # Northeast
    np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
    # North
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    # Northwest
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
    # West
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    # Southwest
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
    # South
    np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    # Southeast
    np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
]

# Apply the Robinson Compass Mask to the image in each direction
results = []
for kernel in kernels:
    filtered = cv2.filter2D(img, -1, kernel)
    results.append(filtered)

# Display the results in Matplotlib subplots
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
axs = axs.flatten()

for i, result in enumerate(results):
    direction = ['East', 'Northeast', 'North', 'Northwest', 'West', 'Southwest', 'South', 'Southeast'][i]
    axs[i].imshow(result, cmap='gray')
    axs[i].set_title(direction)

plt.tight_layout()
plt.show()