
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pre_process import preprocess
import random
from load_config import train_cfg
random.seed(train_cfg["SEED"])
i = random.randint(1, 15)
j = random.randint(1, 25)

def Generate_augment_image(Generate_dataset_name, files, save_path, mask_dir=None, image_name=None, gen_groundtruth=True):
    """
    This function is used to generate training image.

    The augment method including rotation, flip and mirror.
    """
    
    for angle in range(-45, 65, 15):
        for file_path in files:

            if Generate_dataset_name == 'DRIVE':
                img_name = file_path.split("\\")[-1].split("_")[0]
                image = plt.imread(file_path)
                mask = plt.imread(mask_dir + img_name + "_training_mask.gif")
                mask = np.where(mask>0, 1, 0).astype('float32')
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                aug_mask = rotate_image(mask, angle)
                aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
            else:
                img_name = file_path.split("\\")[-1].split(".")[0]
                image = plt.imread(file_path)                
                mask = plt.imread(mask_dir + img_name + "_created_mask.tiff")
                mask = np.where(mask>0, 1, 0).astype('float32')
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                aug_mask = rotate_image(mask, angle)
                aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
                   

            if gen_groundtruth == False:
                rot = rotate_image(image, angle)
                rot = preprocess(rot, gamma=1.2, mask=aug_mask, use_mask=True)
                plt.imsave(save_path + str(img_name) + "_" + "0" + "_" + str(abs(angle+i)) + "_" + image_name , rot)
            else:
                rot2 = rotate_image(image, angle)
                plt.imsave(save_path + str(img_name) + "_" + "0" + "_" + str(abs(angle+i)) + "_" + image_name , rot2, cmap='gray')

    for parameter in range(-1, 2, 1):
        for file_path in files:
            if Generate_dataset_name == 'DRIVE':
                img_name = file_path.split("\\")[-1].split("_")[0]
                image1 = plt.imread(file_path)                
                mask = plt.imread(mask_dir + img_name + "_training_mask.gif")
                mask = np.where(mask>0, 1, 0).astype('float32')
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                aug_mask = flip_image(mask, parameter)
                aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
            else:
                img_name = file_path.split("\\")[-1].split(".")[0]
                image1 = plt.imread(file_path)                  
                mask = plt.imread(mask_dir + img_name + "_created_mask.tiff")
                mask = np.where(mask>0, 1, 0).astype('float32')
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                aug_mask = flip_image(mask, parameter)
                aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
                   

            if gen_groundtruth == False:
                flip = flip_image(image1, parameter)
                flip = preprocess(flip, gamma=1.2, mask=aug_mask, use_mask=True)
                plt.imsave(save_path + str(img_name) + "_" + "1" + "_" + str(parameter+1) + "_" + image_name, flip)
            else:
                flip2 = flip_image(image1, parameter)
                plt.imsave(save_path + str(img_name) + "_" + "1" + "_" + str(parameter+1) + "_" + image_name , flip2, cmap='gray')


    for angle in range(-45, 75, 10):
        for parameter in range(-1, 2, 1):
            for file_path in files:   
                if Generate_dataset_name == 'DRIVE':
                    img_name = file_path.split("\\")[-1].split("_")[0]
                    image2 = plt.imread(file_path)                      
                    mask = plt.imread(mask_dir + img_name + "_training_mask.gif")
                    mask = np.where(mask>0, 1, 0).astype('float32')
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                    aug_mask = rotate_image(mask, angle)
                    aug_mask = flip_image(aug_mask, parameter)
                    aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)
                else:
                    img_name = file_path.split("\\")[-1].split(".")[0]
                    image2 = plt.imread(file_path)                    
                    mask = plt.imread(mask_dir + img_name + "_created_mask.tiff")
                    mask = np.where(mask>0, 1, 0).astype('float32')
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                    aug_mask = rotate_image(mask, angle)
                    aug_mask = flip_image(aug_mask, parameter)
                    aug_mask = cv2.cvtColor(aug_mask, cv2.COLOR_BGR2GRAY)   
                         
            
                if gen_groundtruth==False:
                    conb = rotate_image(image2, angle)
                    conb = flip_image(conb, parameter)
                    conb = preprocess(conb, gamma=1.2, mask=aug_mask, use_mask=True) 
                    plt.imsave(save_path + str(img_name) + "_" + str(abs(angle+55)) + "_" + str(parameter+j) + "_" + image_name, conb)
                else:
                    conb2 = rotate_image(image2, angle)
                    conb2 = flip_image(conb2, parameter)
                    plt.imsave(save_path + str(img_name) + "_" +  str(abs(angle+55)) + "_" + str(parameter+j) + "_" + image_name, conb2, cmap='gray')
                

# Augmentation function
# Roatation angle keep in -45 < angle < 45, cuz after rotation, we going to translation the images.
def rotate_image(image, angle):   
    image_center = tuple(np.array(image.shape[1::-1])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_AREA, borderValue=(0,0,0))
    return result

def flip_image(image, parameter):        
    flip = cv2.flip(image, parameter)
    return flip

            

