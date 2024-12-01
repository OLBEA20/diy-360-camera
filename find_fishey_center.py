import cv2
import numpy as np

def find_fisheye_center(image_path):
    """
    Find the center of a fisheye circular image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    
    Returns:
    --------
    tuple
        (x, y) coordinates of the detected circle center
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,  # Inverse ratio of the accumulator resolution
        minDist=img.shape[0]/8,  # Minimum distance between detected centers
        param1=50,  # Upper threshold for the internal Canny edge detector
        param2=30,  # Threshold for center detection 
        minRadius=int(min(img.shape)/4),  # Minimum circle radius
        maxRadius=int(min(img.shape)/2)   # Maximum circle radius
    )
    
    # If circles are detected
    if circles is not None:
        # Convert circles to integers
        circles = np.uint16(np.around(circles))
        
        # Take the first detected circle (most prominent)
        x, y, radius = circles[0][0]
        return (x, y)
    
    # Alternative method if Hough Transform fails
    def moments_method(image):
        # Compute image moments
        moments = cv2.moments(image)
        
        # Calculate center coordinates
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return None
    
    # Try moments method as a fallback
    center = moments_method(blurred)
    if center:
        return center
    
    # If both methods fail
    raise ValueError("Could not detect the center of the fisheye image")

def visualize_center(image_path, output_path=None):
    """
    Visualize the detected center of the fisheye image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    output_path : str, optional
        Path to save the output image with center marked
    
    Returns:
    --------
    tuple
        (x, y) coordinates of the detected circle center
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Find the center
    center = find_fisheye_center(image_path)
    
    # Draw a small red dot at the center
    cv2.circle(img, center, 10, (0, 0, 255), -1)
    
    # If output path is provided, save the image
    if output_path:
        cv2.imwrite(output_path, img)
    
    return center

# Example usage
if __name__ == "__main__":
    try:
        # Replace with your image path
        image_path = "data/fov0/cam0/test00600.jpeg"
        center = visualize_center(
            image_path, 
            output_path="fisheye_center_marked.jpg"
        )
        image = cv2.imread("fisheye_center_marked.jpg")
        cv2.imshow("center", image )
        cv2.waitKey()
        print(f"Detected Fisheye Center: {center}")
    except Exception as e:
        print(f"Error detecting fisheye center: {e}")

# Additional notes:
# 1. This script works best with high-contrast fisheye images
# 2. Adjust parameters like param1, param2, minRadius, maxRadius 
#    based on your specific image characteristics
# 3. The method uses two approaches:
#    a) Hough Circle Transform (primary method)
#    b) Image Moments (fallback method)