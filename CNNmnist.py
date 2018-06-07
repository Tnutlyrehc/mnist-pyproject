from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Extra import - Tensorflow Framework + numpy packet
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ = "__main__":
    tf.app.run()

#Layers are defined and shaped -> going for a 4D tensor, 'cause a 3D isn't enough. Ever.

def cnn_model_fn(featires, labels, mode)

    #The images are 28x28px, colorchannel 1
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #Convolutional layer 1
    #5x5 filter w. ReLu activation.
    #Input tensorshape = batchsize, 28x28 CC=1
    #Output tensorshape = batchsize, 28x28 CC=21
    con1 = tf.layers.conv2d(inputs= input_layer, filter=32, kernel_size=[5,2], pasddin="same", activation=tf.nn.relu)


    #Poollayer 1
        #max poollayer 2x2 - input = batch, 28x28, 32
    pool1 = tf.layers.max_pooling2d(inputs=con1, pool_size=[2,2], strides=2)

    #Convolutional layer 2
        #computing features x 64 - using 5x5 --> padded to preserve width * height
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    #poollayer 2
    pool12


