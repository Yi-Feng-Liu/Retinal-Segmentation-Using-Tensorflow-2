
import os
from tkinter import font
from unicodedata import name
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from load_config import train_cfg


def tabulate_events(dpath):
    for dname in os.listdir(dpath):

        print(f"Converting run {dname}")

        event_accumulator = EventAccumulator(os.path.join(dpath, dname), size_guidance={'tensors':0}).Reload()
        tags = event_accumulator.Tags()["tensors"]
        
        metrics = ["train_loss", "train_acc", "val_loss", "val_acc"]
        final_value=[]
        for tag in metrics:
         
            tag_value = [(tf.make_ndarray(t)) for _, _, t in event_accumulator.Tensors(tag)]
            final_value.append(tag_value)
            train_step = [s for _, s, _ in event_accumulator.Tensors("train_loss")]
            # val_step = [s for _, s, _ in event_accumulator.Tensors("val_loss")]
            

        df1 = pd.DataFrame(
                          {metrics[0]:final_value[0], 
                           metrics[1]:final_value[1]}
                           )
        df2 = pd.DataFrame(
                          {metrics[2]:final_value[2],
                           metrics[3]:final_value[3]})

        final_df = pd.concat([df1, df2], axis=1, join='outer')
        # final_df.to_csv(f'{dname}.csv')

    return final_df, train_step



def plot_loss_acc(TR_loss_value, VAL_loss_value, TR_acc_value, VAL_acc_value, train_step):

    file_name = train_cfg["file_name"]
    font_size = 12
    fig = plt.figure(figsize=(8,8))
    # loss_label = ['Training loss of ReSinet', 'Valdiation loss of ReSinet']
    # acc_label = ['Training accuracy of ReSinet', 'Validation accuracy of ReSinet']
    # plt.subplot(111)
    plt.plot(train_step, TR_loss_value, label='Training loss of Res Triangular Wave Net')
    plt.plot(train_step, VAL_loss_value, label='Valdiation loss of Res Triangular Wave Net')
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("loss value", fontsize=font_size)
    plt.legend(loc="upper right", prop={"size":font_size})

    # plt.subplot(122)
    # plt.plot(train_step, VAL_loss_value, label='Validation loss of ReSinet')
    # plt.xlabel("Epoch", fontsize=font_size)
    # plt.ylabel("loss value", fontsize=font_size)
    # plt.legend(loc="upper right", prop={"size":font_size})
    plt.tight_layout()
    plt.savefig(train_cfg["save_result"] + 'Loss Comparison result_' + file_name, dpi=300)

    fig2 = plt.figure(figsize=(8,8))
    # plt.subplot(111)
    plt.plot(train_step, TR_acc_value, label='Training accuracy of Res Triangular Wave Net')
    plt.plot(train_step, VAL_acc_value, label='Validation accuracy of Res Triangular Wave Net')
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Accuracy", fontsize=font_size)
    plt.legend(loc="lower right", prop={"size":font_size})

    # plt.subplot(122)
    # plt.plot(train_step, VAL_acc_value, label='Validation accuracy of ReSinet')
    # plt.xlabel("Epoch", fontsize=font_size)
    # plt.ylabel("Accuracy", fontsize=font_size)
    # plt.legend(loc="lower right", prop={"size":font_size})
    plt.tight_layout()

    plt.savefig(train_cfg["save_result"] + 'Accuracy Comparison result_' + file_name, dpi=300)
    plt.show()

# if __name__ == '__main__':
#     df, epoch= tabulate_events("logs/BlockSinet-19-T0.8-112-128-2022-07-04-210801")
#     plot_loss_acc(df["train_loss"], df["val_loss"], df["train_acc "], df["val_acc"], epoch)
