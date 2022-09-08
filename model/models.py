import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, BatchNormalization, Input, ReLU, Activation, LeakyReLU, Dropout

class Resblock(tf.keras.Model):
    """
    Resblock is based on residual learning. In our architecture, three conv layer is seted for each block.
    """
    def __init__(self, filter1:int, filter2:int, filter3:int, strides=2, dif_dim=False):
        super(Resblock, self).__init__(self)
        # If you didn't separate the every single layer that will difficult to observe Model architecture
        self.dif_dim = dif_dim
        self.conv1 = Conv2D(filter1, kernel_size=3, strides=1, padding='same', name='res_conv1', kernel_initializer='HeUniform')
        self.conv2 = Conv2D(filter2, kernel_size=3, strides=1, padding='same', name='res_conv2', kernel_initializer='HeUniform')
        self.conv3 = Conv2D(filter3, kernel_size=3, strides=1, padding='same', name='res_conv3', kernel_initializer='HeUniform')

        self.bn = BatchNormalization(name='bn')
        self.bn1 = BatchNormalization(name='bn1')
        self.bn2 = BatchNormalization(name='bn2')
        self.relu = LeakyReLU()
        self.relu1 = LeakyReLU()
        self.relu2 = LeakyReLU()
        self.add = Add()
        
        
        if dif_dim==True:
            self.shortcut_conv1 = Conv2D(filter1, kernel_size=1, strides=strides, padding='same', name='res_conv1', kernel_initializer='HeUniform')
            self.shortcut_conv2 = Conv2D(filter2, kernel_size=3, strides=1, padding='same', name='res_conv2', kernel_initializer='HeUniform')
            self.shortcut_conv3 = Conv2D(filter3, kernel_size=1, strides=1, padding='same', name='res_conv3', kernel_initializer='HeUniform')
            self.shortcut_conv4 = Conv2D(filter3, kernel_size=1, strides=strides, padding='same', name='shortcut_conv', kernel_initializer='HeUniform')
            self.shortcut_bn = BatchNormalization(name='bn3')
            self.shortcut_bn1 = BatchNormalization(name='bn4')
            self.shortcut_bn2 = BatchNormalization(name='bn5')
            self.shortcut_bn3 = BatchNormalization(name='bn6')
            self.shortcut_relu = LeakyReLU()
            self.shortcut_relu1 = LeakyReLU()
            self.shortcut_relu2 = LeakyReLU()
            

    def call(self, pre_input):

        if self.dif_dim == False:
            xs = self.relu(self.bn(self.conv1(pre_input))) # identity x : there dimension are the same
            xs = self.relu1(self.bn1(self.conv2(xs)))
            xs = self.bn2(self.conv3(xs))
            xs = self.add([xs, pre_input])
            xs = self.relu2(xs) 
        
        # short cut -> In order let pre input have same dimension to add xs.
        if self.dif_dim == True:
            xs = self.shortcut_relu(self.shortcut_bn(self.shortcut_conv1(pre_input)))
            xs = self.shortcut_relu1(self.shortcut_bn1(self.shortcut_conv2(xs)))
            xs = self.shortcut_bn2(self.shortcut_conv3(xs))
            short_cut = self.shortcut_bn3(self.shortcut_conv4(pre_input)) # this conv is 1*1, so strides is 2 for the pre input. 
            xs = self.add([xs, short_cut]) # add the third layer output xs with pre_input(shortcut), that means the original information was added 
            xs = self.shortcut_relu2(xs)
        return xs

class Res_Triangular_wave_Net(tf.keras.Model):
    """Res Triangular wave_Net is modifed from Sine Net, when layer into resblock that will do conv three times for previous layer.

    It differ from Sine Net, becouse they didn't use conv block in Sine Net. 
    In our architecture, every resblock will operate after Up sample and downsample. 

    On last three layers, we did the some change, that filters is decrese from 64 to 32 and last conv layer of filter is 1.

    Also, this architecture did not have activation function on the last layer.
    
    """    
    def __init__(self):
        super(Res_Triangular_wave_Net, self).__init__(self)
        
        self.conv1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')

        self.conv_32up = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_32 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.res_block1 = Resblock(32, 64, 128, dif_dim=True)
        self.res_block2 = Resblock(128, 128, 128, dif_dim=True)
        self.conv_128up = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_64= Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv4 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')
  
        self.conv_last= Conv2D(1 , kernel_size=1, strides=1, padding='same', activation='relu', name='output', kernel_initializer='HeUniform')

        self.bn1 = BatchNormalization(name='bn1')
        self.bn2 = BatchNormalization(name='bn2')
        self.bn3 = BatchNormalization(name='bn3')
        self.bn4 = BatchNormalization(name='bn4')
        self.bn5 = BatchNormalization(name='bn5')
        self.bn6 = BatchNormalization(name='bn6')
        self.bn7 = BatchNormalization(name='bn7')
        self.bn8 = BatchNormalization(name='bn8')
        self.bn9 = BatchNormalization(name='bn9')
        self.bn10 = BatchNormalization(name='bn10')

    def call(self, x, training=True):
#  #-----------------------------------------------------------------------------------#
        stage1 = self.bn1(self.conv1(x, training=training)) #32
        #--------------------------For DRIVE-------------------------------------------#
        stage2 = self.bn2(self.conv_32up(stage1, training=training)) # 32
        stage3 = self.bn3(self.res_block1(stage2, training=training)) # 32 64 128
        stage4 = self.bn4(self.res_block2(stage3, training=training)) # 64 128 256
        stage5 = self.bn5(self.conv_128up(stage4, training=training)) # 128
        stage6 = self.bn6(self.conv_64(stage5, training=training)) # 64
        stage7 = self.bn7(self.conv4(stage6, training=training)) # 32
        #--------------------------For STARE-------------------------------------------#
        # stage2 = self.bn2(self.conv_32up(stage1, training=training)) # 32
        # stage3 = self.bn4(self.res_block1(stage2, training=training)) # 32 64 128
        # stage4 = self.bn5(self.res_block2(stage3, training=training)) # 128 128 128
        # stage5 = self.bn6(self.conv_128up(stage4, training=training)) # 128
        # stage6 = self.bn7(self.conv_64(stage5, training=training)) # 64
        # stage7 = self.bn8(self.conv4(stage6, training=training)) # 32
        #--------------------------For HRF-------------------------------------------#
        output = self.bn9(self.conv_last(stage7, training=training)) # 1
#  #-----------------------------------------------------------------------------------#
        return output

    def model(self, patch_w, patch_h, channel):
        x = Input(shape=(patch_w, patch_h, channel))
        return Model(inputs=[x], outputs = self.call(x, training=False), name = 'Res Triangular wave Net')



class triangular_wave_CNN(tf.keras.Model):

    def __init__(self):
        super(triangular_wave_CNN, self).__init__(self)
        
        self.conv1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_32up = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_64down = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_128down = Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_64up = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv2 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='HeUniform')
        self.conv_last= Conv2D(1 , kernel_size=1, strides=1, padding='same', activation='relu', name='output', kernel_initializer='HeUniform')

        self.bn1 = BatchNormalization(name='bn1')
        self.bn2 = BatchNormalization(name='bn2')
        self.bn3 = BatchNormalization(name='bn3')
        self.bn4 = BatchNormalization(name='bn4')
        self.bn5 = BatchNormalization(name='bn5')
        self.bn6 = BatchNormalization(name='bn6')
        self.bn7 = BatchNormalization(name='bn7')


    def call(self, x, training=True):
#  #-----------------------------------------------------------------------------------#

        stage1 = self.bn1(self.conv1(x, training=training)) #32
        stage2 = self.bn2(self.conv_32up(stage1, training=training)) #32
        stage3 = self.bn3(self.conv_64down(stage2, training=training)) #64
        stage4 = self.bn4(self.conv_128down(stage3, training=training)) # 128
        stage5 = self.bn5(self.conv_64up(stage4, training=training)) # 64
        stage6 = self.bn6(self.conv2(stage5, training=training)) # 32
        output = self.bn7(self.conv_last(stage6, training=training)) # 1

#  #-----------------------------------------------------------------------------------#
        return output

    def model(self, patch_w, patch_h, channel):
        x = Input(shape=(patch_w, patch_h, channel))
        return Model(inputs=[x], outputs = self.call(x, training=False), name = 'Triangular_wave_CNN')

if __name__== "__main__":

    model = Res_Triangular_wave_Net() 
    model.model(224, 224, 3).summary()
    
    # dot_img_file = '/tf/resnet/Res_Triangular_wave_Net.png'

    # plot(model.model(224, 224, 1), to_file=dot_img_file, show_shapes=True)
    
    
    
    



