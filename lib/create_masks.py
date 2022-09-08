import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_mask(image_path, save_path):
    """Creat mask for input image as same size
    """
    image = plt.imread(image_path)
    img_name = image_path.split("\\")[-1].split(".")[0]
    # if use plt to read image, didn't change COLOR BGR2RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0) # using GaussianBlur to denoise

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, -2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for counter in contours:
        area = cv2.contourArea(counter)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = counter

    mask = np.zeros((gray.shape), np.uint8)
    mask = cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    plt.imsave(save_path + str(img_name) + '_created_mask.tiff', mask, cmap='gray')

    return mask