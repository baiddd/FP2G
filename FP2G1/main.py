import tensorflow as tf
import numpy as np
# tensorflow API to retrieve the MNIST dataset
# More info about MNIST here : http://yann.lecun.com/exdb/mnist/
from tensorflow.examples.tutorials.mnist import input_data

def _net_params():
    weights = {
        # Example for conv1: 5 filters of size 5x5x1 -> shape (5, 5, 1, 5) 
        # (last dim is the number of filters)
        'conv1': tf.Variable(tf.random_normal([5, 5, 1, 5])),
        'conv2': tf.Variable(tf.random_normal([5, 5, 5, 50])),
        'fc1': tf.Variable(tf.random_normal([5 * 5 * 50, 100])),
        'fc2': tf.Variable(tf.random_normal([100, 10])),
    }


    # Biases are added to the current layer's neurons
    # Consequently they are of same shape than the number of neurons of the layer
    biases = {
        'conv1': tf.Variable(tf.random_normal([5])),
        'conv2': tf.Variable(tf.random_normal([50])),
        'fc1': tf.Variable(tf.random_normal([100])),
        'fc2': tf.Variable(tf.random_normal([10])),
    }

    return weights, biases


# Helper to create a FC layer before activation
def _fc_layer(inputs, weights, biases):
    return tf.add(tf.matmul(inputs, weights), biases)


# Helper to create a CONV+RELU layer
def _conv_layer(inputs, weights, biases, stride=2, padding='VALID'):
    layer = tf.nn.conv2d(
        input=inputs, 
        filter=weights, 
        strides=[1, stride, stride, 1], 
        padding=padding
    )
    # Adding biases to the filters fastforward transformations
    layer = tf.nn.bias_add(layer, biases)
    # Applying and returning RELU layer to CONV transformations + biases added
    return tf.nn.relu(layer)


# Model of the convolutional network
# x: batch of images; shape (batch_size, 29, 29, 1)
def conv_net(x):
    # Retrieving layers' parameters
    weights, biases = _net_params()

    # Reshaping the inputs to fit in convnet: 4D + the second and third dimensions should be of size 29
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    x = tf.pad(x, paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])

    # Conv layers
    conv1 = _conv_layer(x, weights['conv1'], biases['conv1'])
    conv2 = _conv_layer(conv1, weights['conv2'], biases['conv2'])

    # Flattening output of conv2 to be fed in fc1
    _fc1 = tf.reshape(conv2, [-1, weights['fc1'].get_shape().as_list()[0]])

    # Fully-connected layers
    fc1 = _fc_layer(_fc1, weights['fc1'], biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = _fc_layer(fc1, weights['fc2'], biases['fc2'])

    # Returning predicted output
    return fc2


# Defines the training environment for the network
def _training():
    # Placeholders used to feed the data with tensorflow
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    learning_rate_ = tf.placeholder(tf.float32)

    # Forward pass: calculating predictions of the batch x
    pred = conv_net(x)

    # Defining cost (= loss)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
    # Defining optimizer to compute gradient descent (backpropagation)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cost)

    # Evaluation of the model
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return x, y_, learning_rate_, optimizer, cost, accuracy


def main():
    # Retrieves the MNIST dataset (Python class)
    # The dataset will be downloaded on script launch and put in tmp/data by default
    mnist = input_data.read_data_sets('tmp/data', one_hot=True)

    # Defining hyperparameters for the learning sessions
    n_epochs = 300
    batch_size = 200
    learning_rate = 0.005
    learning_rate_discount = 0.3
    n_epochs_discount = 100

    x, y_, learning_rate_, optimizer, cost, accuracy = _training()

    # Defining saver to save versions of our network throughout training sessions
    saver = tf.train.Saver()
    
    # Starting training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        current_epoch = 0
        while current_epoch < n_epochs:
            current_epoch += 1

            print 'epoch %s' % (current_epoch,)

            # Adding decaying of the learning rate
            if current_epoch % n_epochs_discount == 0:
                learning_rate *= learning_rate_discount

            current_batch = 1
            while current_batch * batch_size <= len(mnist.train.images):
                current_batch += 1

                # Here we need to get the next batch of the dataset
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Here we need to get the next batch of the dataset
                # then update the network parameters according to the learning-policy (i.e. optimizer)
                sess.run(fetches=optimizer, feed_dict={
                     x: batch_x,
                     y_: batch_y,
                     learning_rate_: learning_rate,
                })
                
                # Printing loss and accuracy of the network each 100 batches
                if current_batch % 75 == 0:
                    # Retrieving the mean loss and accuracy of the previous training session
                    # The dropout is set to 1. : we want to see how the net behave so we dont want any neuron to be silenced
                    loss, acc = sess.run([cost, accuracy], feed_dict={
                        x: batch_x,
                        y_: batch_y,
                        learning_rate_: 0.,
                    })

                    print '  batch %s: batch_loss=%s, training_accuracy=%s' % (current_batch, loss, acc,)
            

        print 'Training complete !'
        print 'Final accuracy is %s' % sess.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels,
            learning_rate_: 0.
        })

        # Saving the final state of the network for further usage
        saver_path = saver.save(sess, '%sep_%sbatchsize.meta' % (n_epochs, batch_size,))
        print 'Model saved in file: %s' % (saver_path,)


if __name__ == '__main__':
    main()
