
import os
from xml.dom import NotFoundErr
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras import metrics
import seaborn as sns
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from lib.load_config import dataset_cfg, train_cfg, test_cfg
from sklearn.metrics import roc_curve, cohen_kappa_score
from lib.getDataFromTensorboard import tabulate_events, plot_loss_acc


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



save_result_path = train_cfg["save_result"]
file_name = train_cfg["file_name"]

if not os.path.exists(save_result_path):
    os.mkdir(save_result_path) 

def chohen_kappa(tn, fp, fn, tp):

    p_0 = (tn+tp)/(tn+fp+fn+tp)

    P_a = ((tn+fp)/(tn+fp+fn+tp))*((tn+fn)/(tn+fp+fn+tp))

    P_b = ((fn+tp)/(tn+fp+fn+tp))*((fp+tp)/(tn+fp+fn+tp))

    pe = P_a + P_b

    kappa = (p_0-pe)/(1-pe)

    return kappa


def performance_evaluation(test_groundtruth_path_list:list, pred_save_path, name_of_dataset:str):
    roc_list = []
    acc_list = []

    sensitivity_list = [] # TP/(TP+FN)=> true positive rate
    specificity_lsit = [] # TN/(TN+FP)=> true negative rate
    cohen_score_list = []

    roc = metrics.AUC(num_thresholds=200, curve='ROC')
    tp = metrics.TruePositives(name='TP')
    fn = metrics.FalseNegatives(name='FN')
    fp = metrics.FalsePositives(name='FP')
    tn = metrics.TrueNegatives(name='TN')
    acc = metrics.BinaryAccuracy()

    # for comfusion matrix
    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0

    # read test file
    for idx in range(len(test_groundtruth_path_list)):

        if name_of_dataset == 'DRIVE':
            name = test_groundtruth_path_list[idx].split("\\")[-1].split(".")[0].split('_')[0]
            pred_images = plt.imread(pred_save_path + str(int(name)) + ".png")
        elif name_of_dataset == 'STARE':
            name = test_groundtruth_path_list[idx].split("\\")[-1].split(".")[0]
            pred_images = plt.imread(pred_save_path + str(name) + ".png")
        else:
            name = test_groundtruth_path_list[idx].split("\\")[-1].split(".")[0]
            pred_images = plt.imread(pred_save_path + str(name) + ".png")

        roc.reset_states()
        acc.reset_states()
        tn.reset_states()
        fn.reset_states()
        tp.reset_states()
        fp.reset_states()

        groundtruth = plt.imread(test_groundtruth_path_list[idx])
        pred_images = plt.imread(pred_save_path + str(int(name)) + ".png")

        groundtruth = np.array(groundtruth, dtype=np.float32)
        groundtruth /= 255.0
        groundtruth = np.array(groundtruth, dtype=np.uint8)

        acc.update_state(groundtruth, pred_images[:,:,0])
        roc.update_state(groundtruth, pred_images[:,:,0])
        tp.update_state(groundtruth, pred_images[:,:,0])
        tn.update_state(groundtruth, pred_images[:,:,0])
        fp.update_state(groundtruth, pred_images[:,:,0])
        fn.update_state(groundtruth, pred_images[:,:,0])

        # fprr and tprr, there calculation is difference between keras and tensorflow
        fprr, tprr, _ = roc_curve(groundtruth.ravel(), pred_images[:,:,0].ravel()) 
        # cohen_score = cohen_kappa_score(groundtruth.ravel(), np.round(pred_images[:,:,0]).ravel())
        

        acc_list.append(acc.result())
        roc_list.append(roc.result())
        
        current_tp = tp.result().numpy()
        current_fn = fn.result().numpy()
        current_fp = fp.result().numpy()
        current_tn = tn.result().numpy()

        cohen_score = chohen_kappa(current_tn, current_fp, current_fn ,current_tp)
        cohen_score_list.append(cohen_score)
        
        sum_tp += current_tp
        sum_fn += current_fn
        sum_tn += current_tn
        sum_fp += current_fp

        tpr = current_tp / (current_tp + current_fn)
        tnr = current_tn / (current_tn + current_fp)

        sensitivity_list.append(tpr)
        specificity_lsit.append(tnr)
 

    with open(save_result_path + name_of_dataset + "_" + file_name + '.txt', 'a') as f:
        f.write(
            '\nAverage Accuracy for all prediction :' + str(np.round(tf.reduce_mean(acc_list).numpy(), 4))+
            '\nAverage AUC_ROC for all prediction :' + str(round(tf.reduce_mean(roc_list).numpy(), 4))+
            '\nAverage Sensitivity for all prediction :' + str(np.round(np.mean(sensitivity_list), 4))+
            '\nAverage Specificity for all prediction :' + str(np.round(np.mean(specificity_lsit), 4))+ 
            '\nAverage Cohen kappa for all prediction :' + str(np.round(np.mean(cohen_score_list), 4)) 
        )


    print(f'[INFO] Average Accuracy for all prediction : {tf.reduce_mean(acc_list).numpy():.4f}', )
    print(f'[INFO] Average AUROC for all prediction : {tf.reduce_mean(roc_list).numpy():.4f}', )
    print(f'[INFO] Average Sensitivity for all prediction : {np.mean(sensitivity_list):.4f}')
    print(f'[INFO] Average Specificity for all prediction : {np.mean(specificity_lsit):.4f}')
    print(f'[INFO] Average Cohen kappa for all prediction : {np.mean(cohen_score_list):.4f}')

    return fprr, tprr, roc_list, sum_tp, sum_tn, sum_fp, sum_fn

def save_confusion_matrix(TP, TN, FP, FN): 

    cm = [[TP, FN], [FP, TN]]

    clsses_name = ['Negative', 'Positive']
    # Normalize
    cm = np.asarray(cm, dtype=np.float32)
    print(cm)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.figure(figsize=(8,8))
    
    font_size = 14
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Confusion Matrix", fontdict={'size' : font_size})
    
    sns.heatmap(cm, cmap="Reds", annot=True, square=1, linewidth=2., xticklabels=clsses_name, yticklabels=clsses_name)
    plt.xlabel("Predicted label", fontsize=font_size)
    plt.ylabel("True label", fontsize=font_size)

    plt.tight_layout()
    plt.savefig(save_result_path + "Comfusion Matrix_" + file_name)
    plt.show()

def plot_roc_curve(fpr, tpr, auc_result):

    font_size = 18

    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label=f'AUROC = {np.mean(auc_result):.4f}' )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize = font_size)
    plt.ylabel('True Positive Rate', fontsize = font_size)
    plt.title('Receiver Operating Characteristic', fontsize = font_size)
    plt.legend(loc="lower right", prop = {'size': font_size})
    plt.savefig(save_result_path + 'ROC Curve_' + file_name)

def choose_dataset_for_test_performence(name_of_dataset, HRF_data_name=None):

    if name_of_dataset == 'DRIVE':
        test_dir = dataset_cfg["dataset_path"] + dataset_cfg["test_dir"]
        test_groundtruth = test_dir + dataset_cfg["test_groundtruth"]
        pred_save_path = test_dir + dataset_cfg["test_pred_savepath"]

        test_groundtruth_path_list = sorted(glob(test_groundtruth + '*tif'))
        pred_save_path_list = sorted(glob(pred_save_path + '*.png'))

    elif name_of_dataset == 'STARE':
        test_groundtruth = test_cfg["STARE_path"] + test_cfg["STARE_VKGT"]
        pred_save_path = test_cfg["STARE_path"] + test_cfg["STARE_test_pred_savepath"]

        test_groundtruth_path_list = sorted(glob(test_groundtruth + '*ppm'))
        pred_save_path_list = sorted(glob(pred_save_path + '*.png'))

    elif name_of_dataset == 'HRF':
        if HRF_data_name == 'diabetic':
            test_groundtruth = test_cfg["HRF_path"] + test_cfg["HRF_diabetic"]
            pred_save_path = test_cfg["HRF_path"] + test_cfg["HRF_dr_test_pred_savepath"]
            test_groundtruth_path_list = sorted(glob(test_groundtruth + '*tif'))
            pred_save_path_list = sorted(glob(pred_save_path + '*.png'))

        elif HRF_data_name == 'glaucoma':
            test_groundtruth = test_cfg["HRF_path"] + test_cfg["HRF_glaucoma"]
            pred_save_path = test_cfg["HRF_path"] + test_cfg["HRF_g_test_pred_savepath"]
            test_groundtruth_path_list = sorted(glob(test_groundtruth + '*tif'))
            pred_save_path_list = sorted(glob(pred_save_path + '*.png'))
        else:
            test_groundtruth = test_cfg["HRF_path"] + test_cfg["HRF_healthy"]
            pred_save_path = test_cfg["HRF_path"] + test_cfg["HRF_h_test_pred_savepath"]
            test_groundtruth_path_list = sorted(glob(test_groundtruth + '*tif'))
            pred_save_path_list = sorted(glob(pred_save_path + '*.png'))

    else:
        raise NameError("The data name doesn't exist")

    return test_groundtruth_path_list, pred_save_path_list, pred_save_path

def load_and_save_events(log_file_path):

    df, epoch = tabulate_events(log_file_path)
    plot_loss_acc(df["train_loss"], df["val_loss"], df["train_acc"], df["val_acc"], epoch)


if __name__== '__main__':

    name_of_dataset = 'DRIVE'

    HRF_data_name = None

    log_file_path = 'logs/ResTWNet-25-T0.9-20L-2022-07-24-000610'

    test_GT_list, pred_list, pred_save_path = choose_dataset_for_test_performence(name_of_dataset=name_of_dataset, 
                                                                                     HRF_data_name=HRF_data_name)
    
    assert len(test_GT_list) == len(pred_list)

    FPR, TPR, roc, sum_tp, sum_tn, sum_fp, sum_fn = performance_evaluation(test_GT_list, pred_save_path, name_of_dataset)
    plot_roc_curve(FPR, TPR, roc)
    
    save_confusion_matrix(sum_tp, sum_tn, sum_fp, sum_fn)

    load_and_save_events(log_file_path)




