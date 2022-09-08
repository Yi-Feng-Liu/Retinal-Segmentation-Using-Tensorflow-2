import tensorflow as tf
import random
from glob import glob
from lib.pre_process import *
from tqdm import tqdm
from sklearn.utils import shuffle
from lib.extract_patch import *
from lib.load_config import data_attributes_cfg, train_cfg, testing_cfg
from lib.gen_data import Generate_augment_image
import warnings
warnings.filterwarnings('ignore')
from lib.create_masks import create_mask
from lib.DRIVE_dataloader import load_img_and_groundTruth


resize = data_attributes_cfg["resize"]
patch_size = data_attributes_cfg["patch_size"]
step = data_attributes_cfg["overlap"]
training_ratio = train_cfg["training_ratio"]
epochs = train_cfg["Epoch"]
BATCH_SIZE = train_cfg["batch_size"]
AUTOTONE = tf.data.experimental.AUTOTUNE
data_name = testing_cfg["dataset_STARE"]

def get_stare_dataset(test_cfg, re_generate=True):
    
    STARE_path = test_cfg["STARE_path"]
    train_img = STARE_path + test_cfg["STARE_IMG"]
    train_gt = STARE_path + test_cfg["STARE_VKGT"]
    train_patches = STARE_path + test_cfg["STARE_train_patches"]

    stare_traing_aug_path = STARE_path + test_cfg["STARE_train_augment"]
    stare_gt_aug_path = STARE_path + test_cfg["STARE_gt_augment"]
    stare_mask = STARE_path + test_cfg["STARE_mask"]

    tr_binary_savepath = STARE_path + test_cfg["STARE_tr_binary_savepath"]
    val_binary_savepath = STARE_path + test_cfg["STARE_val_binary_savepath"]
    train_gt_items = glob(train_gt + '*.ppm')

    train_image = []

    for i in range(len(train_gt_items)):
        img = train_gt_items[i]
        specify_test_img = img.split("\\")[-1].split(".")[0]
        train_image_list = str(train_img + specify_test_img + '.ppm')
        train_image.append(train_image_list)

    test_image_list = random.sample(train_image, int(len(train_image)*0.3))
    train_image_list = [i for i in train_image if i not in test_image_list]

    train_gt_list = []
    for i in range(len(train_image_list)):
        img = train_image_list[i]
        create_mask(img, stare_mask) # creat corresponding mask for training image
        
        specify_test_img = img.split("\\")[-1].split(".")[0]
        train_gt_path = str(train_gt + specify_test_img + '.vk.ppm')
        train_gt_list.append(train_gt_path)


    if re_generate:
        clear_old_data(stare_traing_aug_path)
        clear_old_data(stare_gt_aug_path)

        Generate_augment_image(
            data_name, train_image_list, 
            save_path=stare_traing_aug_path, 
            mask_dir=stare_mask, 
            image_name="training.tiff", 
            gen_groundtruth=False
            )
        Generate_augment_image(
            data_name, train_gt_list, 
            save_path=stare_gt_aug_path, 
            mask_dir=stare_mask,
            image_name="manual.tiff", 
            gen_groundtruth=True
            )        

    aug_tr_image_list = glob(stare_traing_aug_path + '*.tiff')
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
        Save_preprocess(data_name, aug_tr_image_list, mask_dir=None, tr_binary_savepath=tr_binary_savepath, produce_validation=False)
        Save_preprocess(data_name, val_img_list,  mask_dir=None, val_binary_savepath=val_binary_savepath, produce_validation=True)
        
        train_binary_list = glob(tr_binary_savepath + "*.tiff")
        print(f"The training image including : {train_binary_list[:5]}\n")

        val_binary_list = glob(val_binary_savepath + "*.tiff")
        print(f"The validation image including : {val_binary_list[:5]}\n")
    
        # It's second time to process and generate the training data 
        for i in tqdm(range(len(train_binary_list)), desc = "Generate the training patches: "):
            get_data_training(data_name, train_binary_list[i], stare_gt_aug_path, train_patches, patch_size, step=step, training=True)

        for i in tqdm(range(len(val_binary_list)), desc = "Generate the validation patches: "):
            get_data_training(data_name, val_binary_list[i], stare_gt_aug_path, train_patches, patch_size, step=step, training=False)
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
