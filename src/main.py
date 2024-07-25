import cv2
import math
import numpy as np

# Load an image
image = cv2.imread('test6.jpg')

# Display the image and select ROI
roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=False)

# Close the window
cv2.destroyAllWindows()

# Extract the ROI coordinates
x, y, w, h = roi

# Print the bounding box
print(f"Bounding box coordinates: x={x}, y={y}, width={w}, height={h}")

# Draw the bounding box on the image
cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

def estimate_depth(x_f, y_f, camera_matrix, image_height, image_width, camera_angle):
    # f : focal length 
    # c : image center coordinates or principle points
    
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    c_x = camera_matrix[0, 2]
    c_y = camera_matrix[1, 2]

    z_p = math.cos(math.radians(camera_angle))
    d = math.sqrt(((c_x - x_f) / (image_width / 2))**2 + ((c_y - y_f) / (image_height / 2))**2)
    theta = math.atan(d / f_x)
    z = z_p * math.cos(theta)
    return z

def camera_to_body_frame(x_c, y_c, z_c, camera_angle):
    alpha = math.radians(camera_angle)
    R = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), -math.sin(alpha)],
        [0, math.sin(alpha), math.cos(alpha)]
    ])
    camera_coords = np.array([x_c, y_c, z_c])
    body_coords = R @ camera_coords
    return body_coords

# Calculate foot coordinates
x_min, y_min, x_max, y_max = x, y, x+w, y+h

# Calculate foot coordinates
x_f = 0.5 * (x_min + x_max)
y_f = y_max

# Camera parameters (example values)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
image_height, image_width = 480, 640
camera_angle = 0  # Since the camera is straight

# Estimate depth
depth = estimate_depth(x_f, y_f, camera_matrix, image_height, image_width, camera_angle)

# Calculate 3D coordinates in camera frame
x_c = (x_f - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
y_c = (y_f - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
z_c = depth

#camera_to_body_frame function with camera_angle = 0
body_coords = camera_to_body_frame(x_c, y_c, z_c, camera_angle)

# Calculate distance in cm
distance_cm = np.linalg.norm(body_coords[:2]) * 100

# Print the object position and distance
print(f"Object position in camera frame: (x={x_c}, y={y_c})")
print(f"Object position in body frame: (x={body_coords[0]}, y={body_coords[1]}, z={body_coords[2]})")
print(f"Distance to object: {distance_cm:.2f} cm")

# Display the image with the bounding box
cv2.imshow("Image with Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

