import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")  # Specify the correct path to the checkpoint
sam.to(device)

# Initialize the predictor
predictor = SamPredictor(sam)

# Load and prepare the image
image = Image.open("test.jpeg")  # Replace with your image path
image_np = np.array(image)
predictor.set_image(image_np)

# Get automatic masks by running the model without prompts
masks = predictor.get_segmentation_mask()

# Process the masks
for i, mask in enumerate(masks):
    print(f"Mask {i}:")
    print(mask.shape)

    # Visualize each mask
    plt.figure()
    plt.imshow(mask)
    plt.title(f"Segmented Object {i}")
    plt.axis("off")
    plt.show()

    # Optionally, save the mask as an image
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(f"segmented_object_{i}.png")
