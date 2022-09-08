# Retinal-Segmentation-Using-Tensorflow-2
This repository contains the implementation of a CNN used to segment retinal vessels in retina fundus images. This is a binary classification task: the neural network predicts if each pixel in the fundus image is either a vessel or not. The performance of this network is tested on the DRIVE database, and it achieves the best score in terms of area under the ROC curve in comparison to the other methods published so far.

#### Remenber to modify the dataset path according to your setting.

# Dataset
The dataset can be found [HERE][]
 
 [HERE]: https://drive.grand-challenge.org/ "HERE"

# Method

Before training, the 20 images of the DRIVE training datasets are pre-processed with the following transformations:

> * Convert to Gray scale

> * Standardization

> * Contrast-limited adaptive histogram equalization (CLAHE)

> * Gamma adjustment

> * Data augmentation (off-line)

> * Create training patches

The training of the neural network is performed on sub-images (patches) of the pre-processed full images. Each patch, of dimension 48x48, is obtained by randomly selecting its center inside the full image. 

Training is performed for 200 epochs, with a mini-batch size of 10 patches. Using a GeForce RTX 3090 GPU the training lasts for about 6.5 hours.


# Result on DRIVE database
Testing is performed with the 20 images of the DRIVE testing dataset, using the gold standard as ground truth. 

# Reference
[1]	J. Ba and R. Caruana, "Do deep nets really need to be deep?," Advances in neural information processing systems, vol. 27, 2014.

[2] J. Chen, D. Zhou, Y. Tang, Z. Yang, Y. Cao, and Q. Gu, "Closing the generalization gap of adaptive gradient methods in training deep neural networks," arXiv    preprint arXiv:1806.06763, 2018.

[3] L. Luo, Y. Xiong, Y. Liu, and X. Sun, "Adaptive gradient methods with dynamic bound of learning rate," arXiv preprint arXiv:1902.09843, 2019.

[4] I. Atli and O. S. Gedik, "Sine-Net: A fully convolutional deep learning architecture for retinal blood vessel segmentation," Engineering Science and Technology, an International Journal, vol. 24, no. 2, pp. 271-283, 2021.
