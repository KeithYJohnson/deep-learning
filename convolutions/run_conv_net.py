from __future__ import print_function
import numpy as np
import tensorflow as tf
from params import *
import sys
sys.path.append('/Users/keithjohnson/courses/deep_learning')
from accuracy import accuracy

batch_size = 16
patch_size = 5 #k subj ≡ kernel size along axis j.
depth = 16
num_hidden = 64

def run_conv_net(
    graph,
    train_dataset,
    train_labels,
    valid_dataset,
    valid_labels,
    test_dataset,
    test_labels):

        with graph.as_default():

          # Input data.
          tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
          tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
          keep_prob = tf.placeholder(tf.float32)
          tf_valid_dataset = tf.constant(valid_dataset)
          tf_test_dataset = tf.constant(test_dataset)
          global_step = tf.Variable(0, trainable=False)

          # Variables.
          layer1_weights = tf.Variable(tf.truncated_normal(
              [patch_size, patch_size, num_channels, depth], stddev=0.1))
          layer1_biases = tf.Variable(tf.zeros([depth]))
          layer2_weights = tf.Variable(tf.truncated_normal(
              [patch_size, patch_size, depth, depth], stddev=0.1))
          layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
          layer3_weights = tf.Variable(tf.truncated_normal(
              [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
          layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
          layer4_weights = tf.Variable(tf.truncated_normal(
              [num_hidden, num_labels], stddev=0.1))
          layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

          # Model.
          def model(data, keep_prob=keep_prob):
            print('keep_prob: ', keep_prob)
            conv = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            pool =  tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv = tf.nn.conv2d(pool, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            pool =  tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), keep_prob)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

          # Training computation.
          logits = model(tf_train_dataset)
          loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

          # Optimizer.
          starter_learning_rate = 0.05
          learning_rate = tf.train.exponential_decay(
            starter_learning_rate, global_step, 100000, 0.96, staircase=True
          )
                                           # Passing global_step to minimize() will increment it at each step.
          optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

          # Predictions for the training, validation, and test data.
          train_prediction = tf.nn.softmax(logits)
          valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
          test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))

        num_steps = 1001

        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
              print('Minibatch loss at step %d: %f' % (step, l))
              print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
              print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
          print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
