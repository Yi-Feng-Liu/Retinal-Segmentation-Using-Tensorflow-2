from unittest import result
import cv2 
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil


def clear_old_data(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

def img_normalized(imgs, mask=None):
    img_normalized = np.empty(imgs.shape)
    img_mean = np.mean(imgs)
    img_std = np.std(imgs)
    img_normalized = (imgs - img_mean)/img_std
    
    for i in range(imgs.shape[2]):
        img_normalized[:,:,i] = (img_normalized[:,:,i] - np.min(img_normalized[:,:,i])) / (np.max(img_normalized[:,:,i])-np.min(img_normalized[:,:,i]))*255
    return img_normalized
    

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# clipLimit is mean contrast, and tileGridSize is a 3x3 block are equalize by histogram equalization in the image.
def clahe_equalized(imgs):
    # assert (imgs.shape[1]==1)  #check the channel is 1
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize = (7,7))
    img_clahe = np.empty(imgs.shape)
    
    for i in range(imgs.shape[2]):
        img_clahe[:,:,i] = clahe.apply(np.array(imgs[:,:,i], dtype = np.uint8))
    return img_clahe

# https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
def gammaCorrection(imgs, gamma = 1):
    invGamma = 1 / gamma
    table = [((i/255.0) ** invGamma)*255 for i in range(0, 256)]
    table = np.array(table, np.uint8)
    new_imgs = np.empty(imgs.shape)
    
    for i in range(imgs.shape[2]):
        new_imgs[:,:,i] =  cv2.LUT(np.array(imgs[:,:,i], dtype = np.uint8), table)
    return new_imgs

def preprocess(imgs, gamma=1.1, mask=None, use_mask=True):
    imgs = np.array(imgs)
    if use_mask==True:
        imgs[:,:,0] = imgs[:,:,0]*mask
        imgs[:,:,1] = imgs[:,:,1]*mask
        imgs[:,:,2] = imgs[:,:,2]*mask
        #-----------process-------------#
        imgs = img_normalized(imgs, mask)
        imgs = clahe_equalized(imgs)
        imgs = gammaCorrection(imgs, gamma)
    else:
        imgs = img_normalized(imgs)
        imgs = clahe_equalized(imgs)
        imgs = gammaCorrection(imgs, gamma)

    imgs = imgs/255.0 # reduce to 0-1 range
    
    return imgs


def Save_preprocess(data_name, img_list, mask_dir=None, tr_binary_savepath=None, val_binary_savepath=None, produce_validation=False):
    """
    This function will save your process file to training/validation path. 
    And the func will write the files to training/validation list, separately.
    """
    for img in img_list:
        
        aug_name = img.split("\\")[-1]
        aug_name = aug_name.split("_")[0] + "_" + aug_name.split("_")[1] + "_" + aug_name.split("_")[2]
        image = plt.imread(img)
        
        if data_name =='DRIVE':
            img_name = img.split("\\")[-1].split("_")[0] # leave the number of file name
            mask = plt.imread(mask_dir + img_name + "_training_mask.gif")
            mask = np.where(mask>0, 1, 0)
        # else:
        #     img_name = img.split("\\")[-1].split("_")[0] # leave the number of file name
        #     mask = plt.imread(mask_dir + img_name + "_created_mask.tiff")
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     mask = np.where(mask>0, 1, 0)

        # image_processed = preprocess(image, gamma=1.2, mask=mask, use_mask=True)
        binary = np.asarray(0.8*image[:,:,1] + 0.2*image[:,:,2]) # the first time process grayscale image
        
        if produce_validation == False:
            plt.imsave(tr_binary_savepath + aug_name + "_training_binary.tiff", binary, cmap='gray')

        else:
            plt.imsave(val_binary_savepath + aug_name + "_validation_binary.tiff", binary, cmap='gray') # you can change your save path



            


    


    