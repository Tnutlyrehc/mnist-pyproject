from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Extra import - Tensorflow Framework + numpy packet
import numpy as np
import tensorflow as tf
from tensorflow.contrib.gan.python import features

tf.logging.set_verbosity(tf.logging.INFO)

NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE



#if __name__ == "__main__":
#    tf.app.run()

#Layers are defined and shaped -> going for a 4D tensor, 'cause a 3D isn't enough. Ever.

def cnn_model_fn(featires, labels, mode):

    #The images are 28x28px, colorchannel 1
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    #input_layer = tf.reshape(features["X"], [-1, 18, 18, 1]) <- For my lazy computer

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
    pool12 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    #Gotta make those layers dense. More dense than a LoL Silver 5 player.
    dense = tf.layers.dense(inputs=pool12_flat, units=1024, activation=tf.nn.relu)

    #droupout op. -> 0,6 prob. on keeping an element.
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    #input tensor - batchsize 1024, output tensor - batchsize 10
    logits = tf.layers.dense(inputs=dropout, units=10)

    #Softmax -> graph.
    predictions = {
        "classes": tf.argmax(inputs=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Loss calculation - train + eval mode
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #Training - TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step= tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Eval mode
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return  tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()













