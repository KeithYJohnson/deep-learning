import tensorflow as tf
from params import *
import numpy as np
from six.moves import range
import random
import math
from generate_cbow_batch import *

def train_cbow(data, reverse_dictionary):
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size, cbow_window * 2])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )

        softmax_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)
            )
        )
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))


        #look up the vector for each of the source words in the batch
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)

        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                softmax_weights,
                softmax_biases,
                tf.reduce_sum(embed, 1),
                train_labels,
                num_sampled,
                vocabulary_size
            )
        )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset
        )
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      average_loss = 0
      for step in range(num_steps):
        batch_data, batch_labels = generate_cbow_batch(
          data, batch_size, cbow_window)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
          if step > 0:
            average_loss = average_loss / 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step %d: %f' % (step, average_loss))
          average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log = '%s %s,' % (log, close_word)
            print(log)
      final_embeddings = normalized_embeddings.eval()
