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
    # Process each mask to get the width in pixels
    for mask in masks:
        bbox = mask['bbox']  # [x_min, y_min, width, height]
        width_in_pixels = bbox[2]  # Index 2 corresponds to the width

        # Calculate real width in millimeters using PPM
        real_width_mm = width_in_pixels / ppm

        # Estimate distance using D' = (W x F) / P
        distance_mm = (real_width_mm * focal_length_mm) / width_in_pixels

        print(f"Mask Width in Pixels: {width_in_pixels}")
        print(f"Real Width of the Object: {real_width_mm:.2f} millimeters")
        print(f"Estimated Distance to the Object: {distance_mm:.0f} millimeters")

    return distance_mm
