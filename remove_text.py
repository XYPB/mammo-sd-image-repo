import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

def remove_text_label(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8) # Convert to 8-bit if not already

    # Binarize the image using a naive non-zero thresholding
    binary_image = (image > 5).astype(np.uint8) * 255
    
    # Apply Gaussian blur to the binarized image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 2.0)
    # Binarize the blurred image again
    binary_image = (blurred_image > 0).astype(np.uint8) * 255
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output image to store the result
    output_image = image.copy()
    
    # Remove small connected components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1e4 and area < np.max(stats[:, cv2.CC_STAT_AREA]):  # Threshold for small areas, adjust as needed
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            output_image[y:y+h, x:x+w] = 0  # Set the region to black
    # if image is set to pure black, return the original image
    if np.all(output_image <= 1e-3):
        output_image = image
    if convert:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    return output_image


if __name__ == "__main__":
    # image_folder = "./csaw_orig"
    # image_folder = "./mammo-rgb"
    image_folder = "./mammo-rgb-bak"
    # image_folder = "./ours"
    # image_folder = "./ca3d_bak"
    images = glob(f"{image_folder}/*.jpg") + glob(f"{image_folder}/*.png")  # Adjust the extension if needed
    for image_path in tqdm(images):
        image = Image.open(image_path)
        processed_image = remove_text_label(image)
        processed_image_pil = Image.fromarray(processed_image)
        processed_image_pil.save(image_path)