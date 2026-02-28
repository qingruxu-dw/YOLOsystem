import cv2
import numpy as np
import os

def remove_background(input_path):
    print(f"Processing: {input_path}")
    
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return

    # Convert to RGBA (add alpha channel)
    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find white background (adjust 240 as needed)
    # Pixels > 240 are considered white background
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert mask: background becomes 0, foreground becomes 255
    mask_inv = cv2.bitwise_not(mask)

    # Use the inverted mask as the alpha channel
    # But wait, this is simple thresholding. Let's refine it.
    # Often logos have anti-aliasing.
    
    # Alternative approach: make white transparent
    # Define white range
    lower_white = np.array([230, 230, 230, 255])
    upper_white = np.array([255, 255, 255, 255])
    
    # Create mask for white pixels
    # We need to work on BGRA image directly or mask on BGR
    
    # Let's use the gray mask approach, it's robust for simple logos on white.
    # Set alpha channel: 0 for white background, 255 for foreground
    
    # Refined mask:
    # Any pixel close to white (e.g. > 240 in all channels)
    # B > 240 and G > 240 and R > 240
    
    b, g, r, a = cv2.split(img_rgba)
    
    # Create a mask where pixels are white
    white_mask = (b > 240) & (g > 240) & (r > 240)
    
    # Set alpha to 0 where mask is True
    a[white_mask] = 0
    
    # Merge back
    img_rgba = cv2.merge((b, g, r, a))
    
    # Save back to same path
    cv2.imwrite(input_path, img_rgba)
    print(f"Successfully saved transparent image to {input_path}")

if __name__ == "__main__":
    # Path provided by user
    target_path = r"c:\Users\westone\Desktop\YOLOsystem\frontend\src\assets\images\tubiao.png"
    remove_background(target_path)
