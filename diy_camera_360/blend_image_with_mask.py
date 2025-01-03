import numpy as np


def blend_images_with_mask(image1, image2, weight=0.5, threshold=10):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    mask1 = np.sum(image1, axis=2) > (threshold * 3)
    mask2 = np.sum(image2, axis=2) > (threshold * 3)

    both_color_mask = mask1 & mask2
    result = np.zeros_like(image1)
    # Where both images have color, blend them
    result[both_color_mask] = image1[both_color_mask] + image2[both_color_mask]
    # Where only img1 has color, use img1
    only_img1_mask = mask1 & ~mask2
    result[only_img1_mask] = image1[only_img1_mask]
    # Where only img2 has color, use img2
    only_img2_mask = mask2 & ~mask1
    result[only_img2_mask] = image2[only_img2_mask]
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
