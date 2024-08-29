import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

# Initialize SAM model
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/content/sam_vit_h_4b8939.pth"

# Define camera parameters
focal_length_mm = 5.1  
sensor_width_mm = 5.6 

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device='cuda' if torch.cuda.is_available() else 'cpu')
mask_generator = SamAutomaticMaskGenerator(sam)

# Load and prepare image
IMAGE_PATH = '/content/test5.jpeg'
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
image_height, image_width_pixels, _ = image.shape

# Convert focal length to pixels
focal_length_px = (focal_length_mm * image_width_pixels) / sensor_width_mm

# Get the dimensions of the image

# Calculate Pixels Per Millimeter (PPM)
ppm = focal_length_px / sensor_width_mm

masks = mask_generator.generate(image)

# Function to estimate distance in millimeters
def estimate_distance_mm(focal_length_mm, ppm, mask_generator, image_rgb , masks):
    image_with_distances = image_rgb.copy()  # Create a copy of the image to draw on
    # Process each mask to get the width in pixels
    for mask in masks:
        bbox = mask['bbox']  # [x_min, y_min, width, height]
        x_min, y_min, width_in_pixels, _ = bbox

        # Calculate real width in millimeters using PPM
        real_width_mm = width_in_pixels / ppm

        # Estimate distance using D' = (W x F) / P
        distance_mm = (real_width_mm * focal_length_mm) / width_in_pixels

        text = f"{distance_mm:.0f} mm"
        cv2.putText(image_with_distances, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    return image_with_distances
