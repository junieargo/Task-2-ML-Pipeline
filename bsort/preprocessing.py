import cv2
import numpy as np
import glob
import os

def get_color_class(img_roi: np.ndarray) -> int:
    
    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average hue and saturation in the center of the ROI
    h, w, _ = hsv.shape
    center_hsv = hsv[h//4:3*h//4, w//4:3*w//4] # Crop center 50%
    avg_color = np.mean(center_hsv, axis=(0, 1))
    
    hue = avg_color[0]  # 0-179
    sat = avg_color[1]  # 0-255
    val = avg_color[2]

    # Thresholds (You must tune these using the provided images)
    # Example Logic:
    if 90 < hue < 130: # Blue range
        if val > 150: 
            return 0 # Light Blue
        else:
            return 1 # Dark Blue
    
    return 2 # Others
