import numpy as np
from PIL import Image

size = 512
half_size = int(size / 2)
img = np.zeros([size, size, 3]).astype(np.uint8)

color_1 = np.array([121,135, 119])
color_2 = np.array([162, 178, 159])

img[:, :, :] = color_1
img[:half_size, :half_size] = color_2
img[half_size:, half_size:] = color_2


gray = np.array([128, 128, 128])
white = np.array([255, 255, 255])
img[:, :] = white

# img[:, :, :] = 16
# half_thickness = 1
# img[:half_thickness] = 128
# img[:, :half_thickness] = 128
# img[-half_thickness:] = 128
# img[:, -half_thickness:] = 128
# img[:, half_size - half_thickness : half_size + half_thickness] = 128
# img[half_size - half_thickness : half_size + half_thickness, :] = 128
# Image.fromarray(img).save("checker_custom.png")
import cv2 
fname = "assets/plane/checker_custom.png"
# cv2.imwrite(fname, img)
cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# cv2.imwrite("checker_custom.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))