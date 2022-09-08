import os
import tensorflow as tf
from tqdm import tqdm
import time
from lib.load_config import dataset_cfg, data_attributes_cfg, train_cfg, testing_cfg
from model.models import Res_Triangular_wave_Net, triangular_wave_CNN
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from lib.extract_patch import load_test_data, padding_images
from patch_reconstructor.recon_from_patches import recon_im, get_patches
from lib.pre_process import preprocess
from lib.create_masks import create_mask
import cv2
import random
random.seed(train_cfg["SEED"])

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # use GPU-0
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

patch_size = data_attributes_cfg["patch_size"]
checkpoint_dir = train_cfg["checkpoint_dir"]
save_result_path = train_cfg["save_result"]
file_name = train_cfg["file_name"]

testmodel = Res_Triangular_wave_Net()
# testmodel2 = triangular_wave_CNN()

ckpt = tf.train.Checkpoint(model=testmodel)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()


def test_DRIVE(image_path, test_mask_dir, DRIVE_test_pred_savepath):

    if not os.path.exists(DRIVE_test_pred_savepath):
        os.mkdir(DRIVE_test_pred_savepath)

    img_name = image_path.split("\\")[-1].split("_")[0]

    original_img = plt.imread(image_path)

    mask = plt.imread(test_mask_dir + img_name + "_test_mask.gif")
    mask = np.where(mask>0.5, 1 ,0)

    image, pad_mask = padding_images(original_img, mask, 10)
    image = preprocess(image, mask=pad_mask, use_mask=True) # was normalized in preprocess
    image = np.asarray(0.8*image[:,:,1] + 0.2*image[:,:,2]).astype('float32') # 轉灰階，把通道拿掉後
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # 再把 channel 補回去，但此時的圖像不會因為轉去 RGB 而有所改變.

    test_patch_list , im_h, im_w, channels = get_patches(image, 15, 224)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(10)

    pred_patches = []
    
    print(f"\nTest image: {int(img_name)}")
    for _, patch in enumerate(test_dataset):
        pred = testmodel(patch, training=False)
        pred = tf.nn.sigmoid(pred)
        pred = pred.numpy()
        pred_patches.append(pred)
    
    pred_patches = np.concatenate(pred_patches, axis=0)
    reconstrcut_image = recon_im(pred_patches, im_h, im_w, channels, 15)

    reconstrcut_image = reconstrcut_image[:original_img.shape[0], :original_img.shape[1]] 

    plt.imsave(DRIVE_test_pred_savepath + str(img_name) + ".png", reconstrcut_image, cmap='gray')   


def test_STARE(specify_test_img, STARE_test_pred_savepath):
    if not os.path.exists(STARE_test_pred_savepath):
        os.mkdir(STARE_test_pred_savepath)

    stare_path = testing_cfg["STARE_path"]
    test_img = stare_path + testing_cfg["STARE_IMG"]

    specify_test_img_name = specify_test_img.split("\\")[-1].split(".")[0]

    original_img = plt.imread(test_img + specify_test_img_name + '.ppm')
    specify_test_img = cv2.copyMakeBorder(original_img, 5, 5, 10, 10, cv2.BORDER_CONSTANT) # padding image

    specify_test_mask = create_mask(specify_test_img) # the mask was padded at previous step
    
    specify_test_img = preprocess(specify_test_img, mask=specify_test_mask, use_mask=True)
    image = np.asarray(0.9*specify_test_img[:,:,1] + 0.1*specify_test_img[:,:,2]).astype('float32') # 轉灰階，把通道拿掉後
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # 再把 channel 補回去，但此時的圖像不會因為轉去 RGB 而有所改變.


    test_patch_list , im_h, im_w, channels = get_patches(image, 15, 224)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(10)

    pred_patches = []
    print(f"\nTest image: {str(specify_test_img_name)}")
    for _, patch in enumerate(test_dataset):
        pred = testmodel(patch, training=False)
        pred = tf.nn.sigmoid(pred)
        pred = pred.numpy()
        pred_patches.append(pred)
    
    pred_patches = np.concatenate(pred_patches, axis=0)
    reconstrcut_image = recon_im(pred_patches, im_h, im_w, channels, 15)

    reconstrcut_image = reconstrcut_image[:original_img.shape[0], :original_img.shape[1]] 

    plt.imsave(STARE_test_pred_savepath + str(specify_test_img_name) + ".png", reconstrcut_image, cmap='gray')       


def test_HRF(specify_test_img, test_name, HRF_test_pred_savepath):
    if not os.path.exists(HRF_test_pred_savepath):
        os.mkdir(HRF_test_pred_savepath)

    HRF_path = testing_cfg["HRF_path"]

    NAME_LIST = ["diabetic", "glaucoma", "healthy"]
    
    if test_name == NAME_LIST[0]:
        test_img = HRF_path + testing_cfg["HRF_diabetic"] # ground truth file is tif, testting file is jpg
    elif test_name == NAME_LIST[1]:
        test_img = HRF_path + testing_cfg["HRF_glaucoma"] # ground truth file is tif, testting file is jpg
    elif test_name == NAME_LIST[2]:
        test_img = HRF_path + testing_cfg["HRF_healthy"] # ground truth file is tif, testting file is jpg.
    else:
       raise NameError("The test image dir does not exist")

    specify_test_img_name = specify_test_img.split("\\")[-1].split(".")[0]
    # specify_test_img = sorted(glob(test_img + specify_test_img_name + '.jpg'))

    original_test_img = plt.imread(specify_test_img)
    test_img = cv2.copyMakeBorder(original_test_img, 25, 25, 10, 10, cv2.BORDER_CONSTANT) # padding image

    test_img_mask = create_mask(test_img)

    test_img = preprocess(test_img, mask=test_img_mask, use_mask=True)

    image = np.asarray(0.9*test_img[:,:,1] + 0.1*test_img[:,:,2]).astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

    test_patch_list , im_h, im_w, channels = get_patches(image, 80, 224)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(10)

    pred_patches = []
    print(f"\nTest image: {str(specify_test_img_name)}")
    for _, patch in enumerate(test_dataset):
        pred = testmodel(patch, training=False)
        pred = tf.nn.sigmoid(pred)
        pred = pred.numpy()
        pred_patches.append(pred)
    
    pred_patches = np.concatenate(pred_patches, axis=0)
    reconstrcut_image = recon_im(pred_patches, im_h, im_w, channels, 80)

    reconstrcut_image = reconstrcut_image[:original_test_img.shape[0], :original_test_img.shape[1]] 

    plt.imsave(HRF_test_pred_savepath + str(specify_test_img_name) + ".png", reconstrcut_image, cmap='gray')  


def choose_test_data(name_of_dataset:str):
    all_pred_time = []
    # test dataset DRIVE
    if name_of_dataset == testing_cfg["dataset_DRIVE"]:
        assert name_of_dataset == testing_cfg["dataset_DRIVE"]

        test_dir = dataset_cfg["dataset_path"] + dataset_cfg["test_dir"]
        test_img = test_dir + dataset_cfg["test_image"]
        test_mask = test_dir + dataset_cfg["test_mask"]
        test_pred_savepath = test_dir + dataset_cfg["test_pred_savepath"]
        test_image_path_list = sorted(glob(test_img + "*tif"))

        for i in tqdm(range(len(test_image_path_list)), desc = "Predicting..."):
            start_time = time.time()
            image_path = test_image_path_list[i]
            test_DRIVE(image_path, test_mask, test_pred_savepath)
            end_time = time.time() - start_time
            all_pred_time.append(end_time)

    # test dataset STARE
    if name_of_dataset == testing_cfg["dataset_STARE"]:
        assert name_of_dataset == testing_cfg["dataset_STARE"]
        stare_path = testing_cfg["STARE_path"]
        test_img = stare_path + testing_cfg["STARE_IMG"]
        test_gt = stare_path + testing_cfg["STARE_VKGT"]
        STARE_test_pred_savepath = stare_path + testing_cfg["STARE_test_pred_savepath"]
        test_gtList = sorted(glob(test_gt + '*ppm'))

        test_gt_list = random.sample(test_gtList, int(len(test_gtList)*0.3))
        
        for i in range(len(test_gt_list)):
            start_time = time.time()
            specify_test_img = test_gtList[i]
            test_STARE(specify_test_img, STARE_test_pred_savepath)
            end_time = time.time() - start_time
            all_pred_time.append(end_time)

    # test dataset HRF       
    if name_of_dataset == testing_cfg["dataset_HRF"]:
        assert name_of_dataset == testing_cfg["dataset_HRF"]
        HRF_path = testing_cfg["HRF_path"]
        HRF_dr_test_pred_savepath = HRF_path + testing_cfg["HRF_dr_test_pred_savepath"]
        HRF_g_test_pred_savepath = HRF_path + testing_cfg["HRF_g_test_pred_savepath"]
        HRF_h_test_pred_savepath = HRF_path + testing_cfg["HRF_h_test_pred_savepath"]

        NAME_LIST = ["diabetic", "glaucoma", "healthy"]
            
        inputs = input("Choose the HRF data wanna test : ")
        inputs = inputs.lower()
        print("Thanks for your confirmation !")
        if inputs == NAME_LIST[0].lower():
            test_img_dr = HRF_path + testing_cfg["HRF_diabetic"] # ground truth file is tif, testting file is jpg
            test_img_dr_list = sorted(glob(test_img_dr + "*jpg"))
            for i in range(len(test_img_dr_list)):
                start_time = time.time()
                image = test_img_dr_list[i]
                test_HRF(image, inputs, HRF_dr_test_pred_savepath)
                end_time = time.time() - start_time
                all_pred_time.append(end_time)
        elif inputs == NAME_LIST[1].lower():
            test_img_g = HRF_path + testing_cfg["HRF_glaucoma"] # ground truth file is tif, testting file is jpg
            test_img_g_list = sorted(glob(test_img_g + "*jpg"))
            for i in range(len(test_img_g_list)):
                start_time = time.time()
                image = test_img_g_list[i]
                test_HRF(image, inputs, HRF_g_test_pred_savepath)
                end_time = time.time() - start_time
                all_pred_time.append(end_time)
        elif inputs == NAME_LIST[2].lower(): 
            test_img_h = HRF_path + testing_cfg["HRF_healthy"] # ground truth file is tif, testting file is jpg.
            test_img_h_list = sorted(glob(test_img_h + "*jpg"))
            for i in range(len(test_img_h_list)):
                start_time = time.time()
                image = test_img_h_list[i]
                test_HRF(image, inputs, HRF_h_test_pred_savepath)
                end_time = time.time() - start_time
                all_pred_time.append(end_time)
        else:
            raise NameError("Data name doesn't exist")

    average_predict_time =  np.round(np.mean(all_pred_time), decimals=4)
    with open(save_result_path + name_of_dataset + "_" + file_name + '.txt', 'w') as f:
        f.write(
           str(testmodel) + '_result : ' + 
           '\nAverage predict time: ' + str(average_predict_time)
        )
    
    return average_predict_time


if __name__ == '__main__':


    test_dataset = 'DRIVE' # choose which datasets do you want to predict
    
    average_predict_time = choose_test_data(test_dataset)
   
    print(f"\n[INFO]Average predict time: {average_predict_time}")
    print("[INFO] All prediction image was saved !")