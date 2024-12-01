import cv2
import numpy as np

def fade_horizontal_edges(image, fade_width=213):
    
    # Convert image to numpy array for manipulation
    img_array = add_alpha(image)
    
    # Get image dimensions
    height, width, _ = img_array.shape
    
    # Create a fade mask
    fade_mask = np.ones((height, width), dtype=float)
    
    # Left edge fade
    for x in range(373, 373 + fade_width):
        # Linear fade from 0 to 1
        fade_value = (x - 373) / fade_width
        fade_mask[:, x] *= fade_value
    
    # Right edge fade
    for x in range(1333, 1333 + fade_width):
        # Linear fade from 1 to 0
        fade_value = ((1333 + fade_width) - x - 1) / fade_width
        fade_mask[:, x] *= fade_value
    
    # Expand mask to 4 channels (RGBA)
    fade_mask = np.stack([fade_mask] * 4, axis=-1)
    
    # Apply fade mask to image
    faded_img_array = (img_array * fade_mask).astype(np.uint8)
    
    return faded_img_array

def add_alpha(image):
    """
    Convert an RGB image to RGBA by adding an alpha channel.
    
    Parameters:
    - rgb_image: OpenCV image in RGB format
    
    Returns:
    - RGBA image with full opacity
    """
    # Create a full opacity alpha channel (255 for all pixels)
    if len(image.shape) == 2:
        # For grayscale, create 3-channel image first
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create a full opacity alpha channel
    height, width = image.shape[:2]
    alpha_channel = np.ones((height, width), dtype=image.dtype) * 255
    
    # Merge the image channels with the alpha channel
    return cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], alpha_channel))