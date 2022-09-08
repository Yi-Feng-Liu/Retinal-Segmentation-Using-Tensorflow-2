import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pre_process import *
from patchify import patchify
from load_config import data_attributes_cfg

patch_size = data_attributes_cfg["patch_size"]

def get_data_training(training_data_name ,image_save_path_list, tr_ground_truth_dir, tr_patch_dir, patch_size, step, training:bool):
    """
    Get training / validation and ground truth patches. 

    Also, the function going to execute data augmentation and save that.
    
    """

    img_name = image_save_path_list.split("\\")[-1].split("_")[0]
    aug_name = image_save_path_list.split("\\")[-1]
    aug_name = aug_name.split("_")[0] + "_" + aug_name.split("_")[1] + "_" + aug_name.split("_")[2]
    image = plt.imread(image_save_path_list)
    
    if training_data_name == 'DRIVE':
        gt = plt.imread(tr_ground_truth_dir + aug_name + "_manual1.tiff")
    elif training_data_name == 'STARE':
        gt = plt.imread(tr_ground_truth_dir + aug_name + "_manual.tiff")
    else:
        gt = plt.imread(tr_ground_truth_dir + aug_name + "_manual.tiff")

    gt = np.asarray(np.where(gt>0, 1 ,0)).astype('float32')
    gt = cv2.cvtColor(gt, cv2.COLOR_BGRA2GRAY).astype(np.uint8)
  
    binary = np.asarray(0.9*image[:,:,1] + 0.1*image[:,:,2]) # the second time process grayscale image

    for img in range(len(image_save_path_list)): 
    
        img_patches = patchify(binary, (patch_size,patch_size), step) # the training imgs is use binary image
        gt_patches = patchify(gt, (patch_size,patch_size), step) 

        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                singe_patch_img = img_patches[i,j,:,:]
                gt_patch_img = gt_patches[i,j,:,:]

                if training==True: # if the training is "True" that will produce the training patches. Otherwise, it will produce the validation patches
                    cv2.imwrite(tr_patch_dir + aug_name + "_" + str(i) + str(j) + '_train_patch.png', singe_patch_img)
                    cv2.imwrite(tr_patch_dir + aug_name + "_" + str(i) + str(j) + '_train_groundtruth_patch.png', gt_patch_img*255)
                else:
                    cv2.imwrite(tr_patch_dir + aug_name + "_" + str(i) + str(j) + '_val_patch.png', singe_patch_img)
                    cv2.imwrite(tr_patch_dir + aug_name + "_" + str(i) + str(j) + '_val_groundtruth_patch.png', gt_patch_img*255)

def load_test_data(image):

    image = tf.image.resize(image, [patch_size,patch_size])

    return image
    

def padding_images(image, mask, stride):
    h,w = image.shape[:2]
    new_h,new_w=h,w
    while (new_h-patch_size)%stride!=0:
        new_h+=1
    while (new_w-patch_size)%stride!=0:
        new_w+=1
    pad_image=np.zeros((new_h,new_w,3))
    pad_image[:h,:w,:]=image
    
    pad_mask=np.zeros((new_h,new_w))
    pad_mask[:h,:w]=mask
    
    return pad_image,pad_mask

