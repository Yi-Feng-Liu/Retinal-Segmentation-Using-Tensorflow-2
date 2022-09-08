import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import tensorflow as tf
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.utils import plot_model as plot
from lib.load_config import train_cfg, dataset_cfg
from lib.BCE import binary_cross_entropy
from model.models import Res_Triangular_wave_Net, triangular_wave_CNN
from lib.DRIVE_dataloader import *
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from lib.same_seed import setup_same_seed

setup_same_seed(train_cfg["SEED"])# Make sure model has reproducibility

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

checkpoint_dir = train_cfg["checkpoint_dir"]
log_dir = train_cfg["logs"]
SGD_lr = train_cfg["SGD_learning_rate"]
Adam_lr = train_cfg["Adam_learning_rate"]
epochs = train_cfg["Epoch"]
save_result_path = train_cfg["save_result"]
file_name = train_cfg["file_name"]


clear_old_data(checkpoint_dir) # clear original 

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)  

model1 = Res_Triangular_wave_Net()
# model2 = triangular_wave_CNN()

decay_rate = Adam_lr/epochs

# Set optimizers
optimizer1 = optimizers.Adam() 
optimizer2 = optimizers.SGD(learning_rate=SGD_lr, momentum=0.9, decay=decay_rate, nesterov=False)

# metric record
train_loss = metrics.Mean(name='train_loss')
train_acc = metrics.BinaryAccuracy(name='train_acc')
 
val_loss = metrics.Mean(name='val_loss')
val_acc = metrics.BinaryAccuracy(name='val_acc')

# Create a Checkpoint that will manage two objects with trackable state
ckpt = tf.train.Checkpoint(model=model1) 

# tensorboard (for visualize)
log_dir = log_dir + "ResTWNet-25-T0.9-20L-" + datetime.now().strftime('%Y-%m-%d-%H%M%S')
log_writer = tf.summary.create_file_writer(log_dir)
early_stop_count = 0


# Run a training loop   
def train_step(step:int, patch, groundtruth):
    with tf.GradientTape() as tape:
        pred_seg = model1(patch, training=True)
        loss_value = binary_cross_entropy(groundtruth, pred_seg)
        
    grads = tape.gradient(loss_value, model1.trainable_weights)

    if step <= epochs-50:
        optimizer1.apply_gradients(zip(grads, model1.trainable_weights))
    if step > epochs-50:
        optimizer2.apply_gradients(zip(grads, model1.trainable_weights))

    # record training loss and accuracy
    train_loss.update_state(loss_value)
    train_acc.update_state(groundtruth, pred_seg)

    return loss_value

# Run a validation loop  
def val_step(step, patch, groundtruth):

    pred_seg = model1(patch, training=False) 
    loss_value = binary_cross_entropy(groundtruth, pred_seg)
 
    val_loss.update_state(loss_value)
    val_acc.update_state(groundtruth, pred_seg)
    tf.summary.image("image", patch, step=step)
    tf.summary.image("groundtruth", groundtruth*255, step=step)
    tf.summary.image("prediction", pred_seg, step=step)
    log_writer.flush()


def train_func(train_data, val_data):

    training_step = 0
    # val_time = train_cfg["validation_time"]
    min_loss = np.Inf

    start_time = datetime.now()
    with log_writer.as_default():
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch+1}/{epochs}")
            epoch_strat_time = time.time()
            
            for tstep, (patch, groundtruth) in enumerate(train_data):
                loss_value = train_step(training_step, patch, groundtruth)
                # Log every 400 batches
                if tstep % 400 == 0:
                    print(
                        f"The training loss (for one batch) at step {tstep} : {float(loss_value):.4f}"
                    )
                    print(f"Seen so far : {(tstep + 1)*BATCH_SIZE} sample")

                if epoch <= (epochs-140): # Adam
                    tf.summary.scalar("Adam learning rate", optimizer1._decayed_lr(tf.float32), step=training_step)
                if epoch > (epochs-140): # SGD with Momentum
                    tf.summary.scalar("SGD learning rate", optimizer2._decayed_lr(tf.float32), step=training_step)

                training_step += 1

            # if (epoch+1) % val_time == 0:
            for _, (patch, groundtruth) in enumerate(val_data):
                val_step(training_step, patch, groundtruth)
               

            print(f"\rTraining & Validation over epoch-----> loss={train_loss.result():.4f}, acc={train_acc.result():.4f}, val loss={val_loss.result():.4f}, val acc={val_acc.result():.4f}", end="")
            tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
            tf.summary.scalar("val_acc", val_acc.result(), step=epoch)

            if val_loss.result() < min_loss:
                min_loss = val_loss.result() 
                ckpt.save(checkpoint_dir)
                print(f'\nSaving model with mean loss {min_loss:.4f}....')
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(f"\nPatience of early stopping: {early_stop_count}")
                if early_stop_count > train_cfg["early_stop"]:
                    print("\nModel is not improving, we going to stop the training session.")
                    return
                           
            print("\nTime taken: %.2fs" % (time.time() - epoch_strat_time))
            # Display metrics at the end of each epoch
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch) 
            tf.summary.scalar("train_acc", train_acc.result(), step=epoch)
            log_writer.flush()

            # Reset metrics at the end of each epoch 
            train_loss.reset_states()
            train_acc.reset_states()
            val_loss.reset_states()
            val_acc.reset_states()

    end_time = datetime.now()
    log_writer.close()
    print(f"\nThe training time is total cost : {end_time-start_time}")
    

if __name__== "__main__":

    model1.model(data_attributes_cfg["resize"], data_attributes_cfg["resize"], 3).summary()

    dot_img_file = '/tf/resnet/model_2.png'

    plot(model1.model(data_attributes_cfg["resize"], data_attributes_cfg["resize"], 3), to_file=dot_img_file, show_shapes=True)

    train_dataset, val_dataset = get_dataset(dataset_cfg, re_generate=False)

    train_func(train_dataset, val_dataset)




