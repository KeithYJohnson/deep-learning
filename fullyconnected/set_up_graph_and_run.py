from params import *
from six.moves import range
import tensorflow as tf
import numpy as np

def set_up_graph_and_run(graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      # These are the parameters that we are going to be training. The weight
      # matrix will be initialized using random values following a (truncated)
      # normal distribution. The biases get initialized to zero.
      global w2, b2, w3, b3
      w2 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_layer_size]))
      b2 = tf.Variable(tf.zeros([hidden_layer_size]))
      w3 = tf.Variable(
        tf.truncated_normal([hidden_layer_size, num_labels]))
      b3 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
      # Training computation.
      def forward_propagate(training_set):
          a2 = tf.nn.relu(tf.matmul(training_set, w2) + b2)
          z3 = tf.matmul(a2, w3) + b3
          return z3

      train_z3 =forward_propagate(tf_train_dataset)

      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(train_z3, tf_train_labels))


      # Optimizer.
      # We are going to find the minimum of this loss using gradient descent.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      # Predictions for the training, validation, and test data.
      # These are not part of training, but merely here so that we can report
      # accuracy figures as we train.

        # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(train_z3)
      valid_prediction = tf.nn.softmax(forward_propagate(tf_valid_dataset))
      test_prediction =  tf.nn.softmax(forward_propagate(tf_test_dataset))





    def accuracy(predictions, labels):
      return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
