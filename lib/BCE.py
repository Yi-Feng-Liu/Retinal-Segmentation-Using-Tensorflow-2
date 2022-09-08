import tensorflow as tf
from tensorflow.keras import backend as K
import numpy

# define the loss function
def binary_cross_entropy(y_true, y_pred):
    """
    NOTICE:
    The BCE is sigmoid function + cross entropy.
    These functions is that they have a sigmoid built in, 
    so your model shouldn't have any activation function on the last layer.
    """
    x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = y_pred))
    return x
