import cv2
import numpy as np
import glob

def calibrate_fisheye_camera(images_path, board_size=(9, 6), save_path='calibration_params.npz'):
    """
    Calibrate a fisheye camera using multiple chessboard images.
    
    Args:
        images_path: Path pattern to calibration images (e.g., 'calibration/*.jpg')
        board_size: Tuple of (columns, rows) of internal corners in the chessboard
        save_path: Path to save calibration parameters
    
    Returns:
        K: Camera matrix
        D: Distortion coefficients
        rms: RMS error of calibration
    """
    # Prepare object points
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(images_path)
    if not images:
        raise ValueError(f"No images found at {images_path}")
    
    successful_images = 0
    
    # Process each calibration image
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to read image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            
            # Show image with detected corners
            cv2.imshow('Chessboard Detection', cv2.resize(img, (800, 600)))
            key = cv2.waitKey(500)
            if key == 27:  # ESC key to exit
                cv2.destroyAllWindows()
                break
                
            successful_images += 1
            print(f"Successfully processed {fname}")
    
    cv2.destroyAllWindows()
    
    if successful_images < 3:
        raise ValueError("Not enough successful calibration images (minimum 3 required)")
    
    print(f"\nCalibrating camera with {successful_images} images...")
    
    # Image size
    img_shape = gray.shape[::-1]
    
    # Calibration flags
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )
    
    # Initialize calibration matrices
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    
    # Calibrate camera
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print("\nCalibration Results:")
    print("RMS error:", rms)
    print("\nCamera matrix (K):")
    print(K)
    print("\nDistortion coefficients (D):")
    print(D)
    
    # Save calibration parameters
    np.savez(save_path, K=K, D=D)
    print(f"\nCalibration parameters saved to {save_path}")
    
    return K, D, rms

def test_calibration(image_path, K, D):
    """
    Test calibration parameters on an image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Display results
    cv2.imshow('Original', cv2.resize(img, (800, 600)))
    cv2.imshow('Undistorted', cv2.resize(undistorted_img, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    try:
        # Calibrate camera
        K, D, rms = calibrate_fisheye_camera(
            images_path='calib/*.jpeg',  
            board_size=(8, 6),
            save_path='fisheye_calibration.npz'
        )
        
        # Test calibration on a sample image
        test_calibration('cam1/test00200.jpeg', K, D)
        
    except Exception as e:
        print(f"Error: {str(e)}")