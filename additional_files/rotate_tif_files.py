# This script rotates a given raster (in a .tif file). 
# It saves the new raster (with some artifacts on the sides) as a new .tif file.
# Install OpenCV by: pip install opencv-python-headless
# ------------
# Sjur Barndon

import cv2

# Specify inputs:
path_input = "domain_complete_TIFF_1.tif"
path_output = "rotated_raster.tif"
rotation_angle = -3.8 # Specify the rotation angle (in degrees) found by trial and error


# Load the TIF file with float64 values
image = cv2.imread(path_input, cv2.IMREAD_UNCHANGED)

# Get the image dimensions
height, width = image.shape

# Calculate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1.0)

# Perform the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

# Save the rotated image
cv2.imwrite(path_output, rotated_image)
