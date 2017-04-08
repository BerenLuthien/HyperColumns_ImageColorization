# HyperColumns of CNN and Image Colorization

## 1 Introduction
HyperColumnsImageColorization is the project that I've created during my time as an Artificial Intelligence Fellow with Insight Data Science in early 2017. In this project, from the pre-trained VGG model "HyperColumns" is harvested and is used to colorize gray images.
The major target of this project is to explore HyperColumns and how it can be used in such computer vision tasks as image auto-colorizations. The training data is flower data set which is separated into train, validation and test sets. The trained model is also tested on images that are not from the flower data set. The project is done using Tensorflow 1.0.1.
![](pics/head.jpg)

## 2 Task description
A colorful image can be decomposed into three channels, such as RGB, LAB, HSL and HSV.  LAB is used in this project (https://en.wikipedia.org/wiki/Lab_color_space) where L means "lightness". L-channel representing a gray color image is the the input of my model,  and the output is the predicted colorful image.
![](pics/2.jpg)


### 2.1 Does one channel contains all information of the other two channels ?
This is the first question many people would ask themselves at the very beginning. Me too. More specifically, does L channel contains all information of the A & B channels ? No. Then how can we recover A & B channels from L channel ?

The answer to this question leads to the usage of HyperColumns and a pre-trained convolutional neural network (CNN). In this project, pre-trained VGG is adopted and tweaked. VGG was trained on huge amount of images and it contains a lot of information regarding quite many of (if not all) objects in the world. Taking advantage of VGG, we should be able to colorize the gray images. VGG as "external information" is the essential reason why this colorization task can be done.
![](pics/1.jpg)

Making an analogy. Given three data points, we need to output a curve to fit them. There are tons of various curves that can fit these three data points. However, if somebody tells us (external information !) that the curve is most probably a quardratic curve, then we probably will produce the blue color curve.
![](pics/4.jpg)
In order to harvest this external information that VGG has to provide, we need HyperColumns.


## 3. HyperColumns and VGG

Regarding CNN and VGG, refer to http://cs231n.github.io/convolutional-networks/

### 3.1 Load a pre-trained VGG

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

        def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )


### 3.2 HyperColumns from VGG

The layers of a convolutional network is like as a non-linear counterpart of the image pyramids. The feature maps have different sizes. The topper they are on the VGG model, the smaller their sizes are. However, we need them to be of the same size, e.g., the size of the input grey image. Thus, the feature maps are upscaled by bilinear interpolation and are contatenated together to give us a "HyperColumn".  (It looks like HyperMaps rather than HyperColumns ?!)
![](pics/HyperColumns.jpg)


        relu1_2  = image_net["relu1_2"]
        layer_relu1_2 = tf.image.resize_bilinear(relu1_2, (IMAGE_SIZE, IMAGE_SIZE)) 

        relu2_1  = image_net["relu2_1"]
        layer_relu2_1 = tf.image.resize_bilinear(relu2_1, (IMAGE_SIZE, IMAGE_SIZE)) 
        
        relu2_2  = image_net["relu2_2"]
        layer_relu2_2 = tf.image.resize_bilinear(relu2_2, (IMAGE_SIZE, IMAGE_SIZE)) 

        relu3_1  = image_net["relu3_1"]
        layer_relu3_1 = tf.image.resize_bilinear(relu3_1, (IMAGE_SIZE, IMAGE_SIZE))         
        relu3_2  = image_net["relu3_2"]
        layer_relu3_2 = tf.image.resize_bilinear(relu3_2, (IMAGE_SIZE, IMAGE_SIZE)) 
        relu3_3  = image_net["relu3_3"]
        layer_relu3_3 = tf.image.resize_bilinear(relu3_3, (IMAGE_SIZE, IMAGE_SIZE)) 
        
        relu3_4  = image_net["relu3_4"]
        layer_relu3_4 = tf.image.resize_bilinear(relu3_4, (IMAGE_SIZE, IMAGE_SIZE))         
        relu4_1  = image_net["relu4_1"]
        layer_relu4_1 = tf.image.resize_bilinear(relu4_1, (IMAGE_SIZE, IMAGE_SIZE)) 
        relu4_2  = image_net["relu4_2"]
        layer_relu4_2 = tf.image.resize_bilinear(relu4_2, (IMAGE_SIZE, IMAGE_SIZE)) 
        relu4_3  = image_net["relu4_3"]
        layer_relu4_3 = tf.image.resize_bilinear(relu4_3, (IMAGE_SIZE, IMAGE_SIZE)) 
        relu4_4  = image_net["relu4_4"]
        layer_relu4_4 = tf.image.resize_bilinear(relu4_4, (IMAGE_SIZE, IMAGE_SIZE)) 
        
        relu5_1  = image_net["relu5_1"]
        layer_relu5_1 = tf.image.resize_bilinear(relu5_1, (IMAGE_SIZE, IMAGE_SIZE))         
        relu5_2  = image_net["relu5_2"]
        layer_relu5_2 = tf.image.resize_bilinear(relu5_2, (IMAGE_SIZE, IMAGE_SIZE))         
        relu5_3  = image_net["relu5_3"]
        layer_relu5_3 = tf.image.resize_bilinear(relu5_3, (IMAGE_SIZE, IMAGE_SIZE))         
        relu5_4  = image_net["relu5_4"]
        layer_relu5_4 = tf.image.resize_bilinear(relu5_4, (IMAGE_SIZE, IMAGE_SIZE))        
        
        HyperColumns = tf.concat([layer_relu1_2, \
                                     layer_relu2_1, layer_relu2_2, \
                                     layer_relu3_1, layer_relu3_2, layer_relu3_3, layer_relu3_4, \
                                     layer_relu4_1, layer_relu4_2, layer_relu4_3, layer_relu4_4, \
                                     layer_relu5_1, layer_relu5_2, layer_relu5_3, layer_relu5_4  \
                                    ] ,3)

### 3.3 Why call it HyperColumns ?
This terminology comes from neuroscience, where some part of our brain is called hypercolumn (refer to the paper link at the bottom). 

Here is a quick illustration from wiki: "A cortical column, also called hypercolumn, macrocolumn, functional column or sometimes cortical module, is a group of neurons in the cortex of the brain that can be successively penetrated by a probe inserted perpendicularly to the cortical surface, and which have nearly identical receptive fields."
![](pics/cortical.jpg)

This analogy is quite similar to the feature maps of CNN model.

## 4. Model
Since gray color image contains only one channel, in order for VGG to be able to process it, the first convolutional filter of VGG is replaced with a new filter. This new filter takes into one channel tensor and then output 64-channels tensor which is fed into the rest part of VGG.  

        W0 = utils.weight_variable([3, 3, 1, 64], name="W0")
        b0 = utils.bias_variable([64], name="b0")
        conv0 = utils.conv2d_basic(images, W0, b0)
        hrelu0 = tf.nn.relu(conv0, name="relu")
        image_net = vgg_net(weights, hrelu0)


Further and most importantly, the HyperColumns layer is added upon it, and then the HyperColumns are "squeezed" into a two-channels tensor which correspond to the prediction of the A & B channels. 
Imagine a chopstick is pushed through many slices of bread and get many holes, then all the holes comprise one hypercolumn, and it corresponds to one pixel of the original image.
This process was done by 1-by-1 convolution that "stiches" the feature maps in the HyperColumns together.


        HyperColumns = tf.concat([layer_relu1_2, \
                                     layer_relu2_1, layer_relu2_2, \
                                     layer_relu3_1, layer_relu3_2, layer_relu3_3, layer_relu3_4, \
                                     layer_relu4_1, layer_relu4_2, layer_relu4_3, layer_relu4_4, \
                                     layer_relu5_1, layer_relu5_2, layer_relu5_3, layer_relu5_4  \
                                    ] ,3)
        # wc1: Weights Combiner, 1-by-1 conv
        wc1 = utils.weight_variable([1, 1, 5440, 2], name="wc1")
        wc1_biase = utils.bias_variable([2], name="wc1_biase")
        pred_AB_conv = tf.nn.conv2d(HyperColumns, wc1, [1, 1, 1, 1], padding='SAME')
        pred_AB = tf.nn.bias_add(pred_AB_conv, wc1_biase)        
    return tf.concat(values=[images, pred_AB], axis=3,  name="pred_image")



![](pics/6.jpg)

### 4.1 Sampled results
Here are some sampled results. Some predicted image (top right) looks even better than the orginal one. There are also sort of failures, such as the bottom left, where the statue was given some pink color.

![](pics/9.jpg)
Applied the model on illustration pictures of elder books:

![](pics/illustration.png)

Further, we can see the progress that the model made during training: 

![](pics/train.png)


### 4.2 Tensorboard: how/which weights were updated during training 
Tensorboard allows us to peek into how the network weights (conv filters) change during training. Here shows some of the filters and biases:

![](pics/b.jpg)

Actually all layers of filters have been updated to a considerable extent during the training. This indicates that all of the feature maps are useful and they probably contains different information. We should better incorporate all of the feature maps into our HyperColumns to avoid information loss.
That said, what if we only sample a portion of the feature maps ?

## 5. Simplified model
### 5.1 Simplified model, without top layer
        wc1 = utils.weight_variable([1, 1, 960, 2], name="wc1")
        vgg_pool_1 = image_net["pool1"]
        vgg_pool_2 = image_net["pool2"]
        vgg_pool_3 = image_net["pool3"]
        vgg_pool_4 = image_net["pool4"]
         
        layer1 = tf.image.resize_bilinear(vgg_pool_1, (IMAGE_SIZE, IMAGE_SIZE))
        layer2 = tf.image.resize_bilinear(vgg_pool_2, (IMAGE_SIZE, IMAGE_SIZE))
        layer3 = tf.image.resize_bilinear(vgg_pool_3, (IMAGE_SIZE, IMAGE_SIZE))
        layer4 = tf.image.resize_bilinear(vgg_pool_4, (IMAGE_SIZE, IMAGE_SIZE))

        HyperColumns = tf.concat([layer1, layer2,layer3, layer4] ,3)        
        pred_AB_conv = tf.nn.conv2d(HyperColumns, wc1, [1, 1, 1, 1], padding='SAME')
        wc1_biase = utils.bias_variable([2], name="wc1_biase")
        pred_AB = tf.nn.bias_add(pred_AB_conv, wc1_biase)        
    return tf.concat(values=[images, pred_AB], axis=3,  name="pred_image")

This simplified model picks up the output of the pooling of the first four layers of VGG, upscale them, and then concatenated them into a thinner HyperColumn. 


![](pics/SimpleModel_FourLayers.png)

Apparently some information that VGG has to provied is lost, but this thinner model requests less computation and less memory. It is not powerful enough. Here are some samples of its predictions :

![](pics/partial_model_results.png)

The predictions are not as good as the full model above, but still fine. Its training loss is larger than the full model.

![](pics/FOURlayers_loss.png)

### 5.2 Simplified model, with top layer, before pooling
This simplified model picks up the output of ReLu (which means before pooling) of the first five layers (which means the top conv layer is included) of VGG, upscale them, and then concatenated them into a thinner HyperColumn. Surprisingly, its performance is almost as good as the above full model.

        wc1 = utils.weight_variable([1, 1, 1472, 2], name="wc1")
        vgg_1 = image_net["relu1_2"]
        vgg_2 = image_net["relu2_2"]
        vgg_3 = image_net["relu3_4"]
        vgg_4 = image_net["relu4_4"]
        vgg_5 = image_net["relu5_4"]
         
        layer1 = tf.image.resize_bilinear(vgg_1, (IMAGE_SIZE, IMAGE_SIZE))
        layer2 = tf.image.resize_bilinear(vgg_2, (IMAGE_SIZE, IMAGE_SIZE))
        layer3 = tf.image.resize_bilinear(vgg_3, (IMAGE_SIZE, IMAGE_SIZE))
        layer4 = tf.image.resize_bilinear(vgg_4, (IMAGE_SIZE, IMAGE_SIZE))
        layer5 = tf.image.resize_bilinear(vgg_5, (IMAGE_SIZE, IMAGE_SIZE))

        HyperColumns = tf.concat([layer1, layer2,layer3, layer4, layer5] ,3)        
        pred_AB_conv = tf.nn.conv2d(HyperColumns, wc1, [1, 1, 1, 1], padding='SAME')
        wc1_biase = utils.bias_variable([2], name="wc1_biase")
        pred_AB = tf.nn.bias_add(pred_AB_conv, wc1_biase)        
    return tf.concat(values=[images, pred_AB], axis=3,  name="pred_image")



![](pics/FIVElayers_simplifiedModel.png)

Some predictions are compared against the full model:

![](pics/Simplified_with_top_layer_pred.png)

This result implies that layer-5 contains important information regarding color. Further, we should probably pick up feature map before pooling rather than after pooling. Pooling operation loses some information after all.

It may be interesting to try different combinations of layers, such as only output of layers 1,2,3, or only output of layeys 3,4,5, and so on.

## 6. Other models I tried
I come up with two other models based on the concept of HyperColumns. The two models try to introduce more capacity, but they do not give better performance.  Thus, it looks that the original HyperColumns already have enough capacity.
Anyway, here are what I've tried:
### 6.1 other-model-1
The idea is to introduce more non-linearity into the network. Basically the feature maps from the same layer of VGG are concatenated together to give a layer-based-HyperColumns. Each layer-based-HyperColumns is "squeezed" by 1by1 conv into a single feature map. At last, these five feature maps go through non linear function ReLu respectively, are concatenated into one HyperColumns, and at last is used to make prediction.
Without ReLu functions, this model will be equivalent to the full HyperColumns model illustrated earlier.

![](pics/layered_hypercolumns.png)

### 6.2 other-model-2
The idea is to introduce more conv filters into the network. Based on other-model-1, before the final 1-by-1 conv, two new conv filters are inserted to give the model more capacity.

![](pics/model_2.png)

## 7. More... Cartoon ?
Apply the model on cartoon, and it did not give a satisfying result. This is what I had expected because the model was never trained on cartoons. What if I have enough training data set of cartoons ? It will be interesting to see if it can colorize the cartoons from the same author. After all, an author usually presents a consistent style of art of herself/himself.


# useful linkes
## github or posts
* [Colorful Image Colorization](http://richzhang.github.io/colorization/) 
* [Automatic Colorization](http://tinyclouds.org/colorize/) 
* [Image colorization](https://github.com/shekkizh) 


## papers
* [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752) - Bharath Hariharan, et. al.
* [Learning Representations for Automatic Colorization](https://arxiv.org/pdf/1603.06668.pdf) - Gustav Larsson, et. al.
* [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf) - Richard Zang et. al.
