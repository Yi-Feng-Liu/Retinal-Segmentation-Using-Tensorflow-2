
from os import truncate
import tensorflow as tf
import random
from glob import glob
from pre_process import *
from tqdm import tqdm
from sklearn.utils import shuffle
from extract_patch import *
from load_config import data_attributes_cfg, train_cfg, testing_cfg
from gen_data import Generate_augment_image
import numpy as np
import warnings
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
import random 
random.seed(train_cfg["SEED"])

# kf = KFold(5, shuffle=True, random_state=train_cfg["SEED"])
resize = data_attributes_cfg["resize"]
patch_size = data_attributes_cfg["patch_size"]
steps = data_attributes_cfg["overlap"]
training_ratio = train_cfg["training_ratio"]
epochs = train_cfg["Epoch"]
BATCH_SIZE = train_cfg["batch_size"]
AUTOTONE = tf.data.experimental.AUTOTUNE
data_name = testing_cfg['dataset_DRIVE']

def load_img_and_groundTruth(img_path, gt_path , seed):
# def load_img_and_groundTruth(img_path, gt_path):
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    
    groundTruth = tf.io.read_file(gt_path)
    groundTruth = tf.image.decode_png(groundTruth, channels=1)

    # Data augmentation
    unit_seed = random.uniform(0,1)
    if unit_seed > 0.5:
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :] # Make a new seed

        for i in range(3):
            img = tf.image.rot90(img, i)
            groundTruth = tf.image.rot90(groundTruth, i)
        
        img = tf.image.stateless_random_contrast(img, 0.2, 0.8, new_seed)

        img = tf.image.central_crop(img, 0.5)
        groundTruth = tf.image.central_crop(groundTruth, 0.5)
        
        img = tf.image.stateless_random_flip_left_right (img, new_seed)
        groundTruth = tf.image.stateless_random_flip_left_right(groundTruth, new_seed)
        
        img = tf.image.stateless_random_flip_up_down(img, new_seed)
        groundTruth = tf.image.stateless_random_flip_up_down(groundTruth, new_seed)
     
    img = tf.image.resize(img, [resize, resize])
    groundTruth = tf.image.resize(groundTruth, [resize, resize])  
                
    img /= 255
    groundTruth /= 255

    return img, groundTruth

def get_dataset(dataset_cfg, re_generate=True):
    # load configs
    
    dataset_path = dataset_cfg["dataset_path"]
    train_dir = dataset_path + dataset_cfg["train_dir"]
    train_image = train_dir + dataset_cfg["train_image"]
    tr_mask_dir = train_dir + dataset_cfg["train_mask"] 
    train_patches = train_dir + dataset_cfg["train_patch"]
    train_gt = train_dir + dataset_cfg["train_groundtruth"]

    aug_tr_image = dataset_path + dataset_cfg["train_augment_path"]
    aug_tr_gt_image =dataset_path + dataset_cfg["train_gt_augment_path"]

    # It's processed img and that has been splilt
    tr_binary_savepath = dataset_path + dataset_cfg["tr_binary_savepath"]
    val_binary_savepath = dataset_path + dataset_cfg["val_binary_savepath"]

    train_image_list = glob(train_image + "*.tif")
    train_gt_list = glob(train_gt + '*.gif')

    aug_tr_image_list = glob(aug_tr_image + '*.tiff')

    if re_generate:
        clear_old_data(aug_tr_image)
        clear_old_data(aug_tr_gt_image)
     
        Generate_augment_image(data_name, train_image_list, save_path=aug_tr_image, mask_dir=tr_mask_dir, image_name="training.tiff", gen_groundtruth=False)
        Generate_augment_image(data_name, train_gt_list, save_path=aug_tr_gt_image, mask_dir=tr_mask_dir, image_name="manual1.tiff", gen_groundtruth=True)
        
    
    aug_tr_image_list = glob(aug_tr_image + '*.tiff')
    val_img_list = random.sample(aug_tr_image_list, int(len(aug_tr_image_list)*((1 - training_ratio))))
    aug_tr_image_list = [i for i in aug_tr_image_list if i not in val_img_list]

    print(f"[INFO] Number of training imgs : {len(aug_tr_image_list)}")
    print(f"[INFO] Number of validate imgs : {len(val_img_list)}")

   # generate the patches
    if re_generate:
        clear_old_data(train_patches)
        clear_old_data(tr_binary_savepath)
        clear_old_data(val_binary_savepath)

        # first time to pre process img to gray and SAVE them it separately. If show is true, you can watch the images are changing.
        Save_preprocess(data_name, aug_tr_image_list, tr_mask_dir, tr_binary_savepath, val_binary_savepath, produce_validation=False)
        Save_preprocess(data_name, val_img_list, tr_mask_dir, tr_binary_savepath, val_binary_savepath, produce_validation=True)
        
        train_binary_list = glob(tr_binary_savepath + "*.tiff")
        print(f"The training image including : {train_binary_list[:5]}\n")

        val_binary_list = glob(val_binary_savepath + "*.tiff")
        print(f"The validation image including : {val_binary_list[:5]}\n")
    
        # It's second time to process and generate the training data 
        for i in tqdm(range(len(train_binary_list)), desc = "Generate the training patches: "):
            get_data_training(data_name, train_binary_list[i], aug_tr_gt_image, train_patches, patch_size, step=steps, training=True)

        for i in tqdm(range(len(val_binary_list)), desc = "Generate the validation patches: "):
            get_data_training(data_name, val_binary_list[i], aug_tr_gt_image, train_patches, patch_size, step=steps, training=False)
    else:
        print("[INFO] Use generated patches...")

    # Load the training and validation patches
    tr_patches_list = sorted(glob(train_patches + "*_*_*_*_train_patch.png"))
    tr_gt_patch_list = sorted(glob(train_patches + "*_*_*_*_train_groundtruth_patch.png"))

    tr_patches_list, tr_gt_patch_list = shuffle(tr_patches_list, tr_gt_patch_list)

    val_patch_list = sorted(glob(train_patches + "*_*_*_*_val_patch.png"))
    val_gt_patch_list = sorted(glob(train_patches + "*_*_*_*_val_groundtruth_patch.png"))

    val_patch_list, val_gt_patch_list = shuffle(val_patch_list, val_gt_patch_list)

    print(f"[INFO] Training patches & groundtruth have ({len(tr_patches_list)}, {len(tr_gt_patch_list)}) images in total !")
    print(f"[INFO] Validation patches & groundtruth have ({len(val_patch_list)}, {len(val_gt_patch_list)}) images in total !")

    # Make sure that image and groundtruth are still corresponding 
    print(tr_patches_list[:2])
    print(tr_gt_patch_list[:2])

    print(val_patch_list[:2])
    print(val_gt_patch_list[:2])

    # Training Dataloader
    #-----------------------------------------------------------------------#
    train_img_dataset = tf.data.Dataset.from_tensor_slices((tr_patches_list))
    train_lb_dataset = tf.data.Dataset.from_tensor_slices((tr_gt_patch_list))
    counter = tf.data.experimental.Counter()
    train_dataset = tf.data.Dataset.zip((train_img_dataset, train_lb_dataset,(counter, counter)))
    #-----------------------------------------------------------------------#
    train_dataset = train_dataset.shuffle(buffer_size = 25000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(load_img_and_groundTruth, num_parallel_calls=AUTOTONE)
    train_dataset = train_dataset.prefetch(AUTOTONE).batch(BATCH_SIZE)
    print(train_dataset)
    
    
    # Validation Dataloader
    val_img_dataset = tf.data.Dataset.from_tensor_slices((val_patch_list))
    val_lb_dataset = tf.data.Dataset.from_tensor_slices((val_gt_patch_list))
    counter = tf.data.experimental.Counter()
    val_dataset = tf.data.Dataset.zip((val_img_dataset, val_lb_dataset,(counter, counter)))
    #-----------------------------------------------------------------------#
    val_dataset = val_dataset.shuffle(buffer_size = 25000, reshuffle_each_iteration=True)
    val_dataset = val_dataset.map(load_img_and_groundTruth, num_parallel_calls=AUTOTONE)
    val_dataset = val_dataset.prefetch(AUTOTONE).batch(BATCH_SIZE) 
    print(val_dataset)
    
    return train_dataset, val_dataset

