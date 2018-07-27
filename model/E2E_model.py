import tensorflow as tf
from data_provider import DataProvider
import logger
import argparse
import time
import math
import copy
import numpy as np

class E2EModel(object):
    def __init__(self, task_num, data_form):
        """

        :param data_path:
        :param task_num: train model on which task
        :param data_form: 1 denotes x_batch shape: [batch_size, num_utterance, sequence_max_len, vocab_size]
                          2 denotes x_batch shape: [batch_size, num_all_word_in_dialog, vocab_size]

                          Hierarchical LSTM, MLP use data form 1
                          LSTM, AttnNet use data form 2
        """
        self.output_dim = 20
        # self.path = data_path
        self.max_num_utterance = 25
        self.max_len_utterance = 30
        self.max_num_words_in_dialog = 180
        self.vocab_size = 205

        self.embed_matrix = None
        self.load_embed('../my_dataset/sub_glove_embedding_with_oov.txt')

        self.data_provider = DataProvider(data_form)

        if task_num == 1:
            self.task_num = 'task1'
            self.train_set = copy.deepcopy(self.data_provider.task1.train)
            self.val_set = copy.deepcopy(self.data_provider.task1.val)
            self.test_sets = [None,
                              copy.deepcopy(self.data_provider.task1.test1),
                              copy.deepcopy(self.data_provider.task1.test2),
                              copy.deepcopy(self.data_provider.task1.test3),
                              copy.deepcopy(self.data_provider.task1.test4)]
        elif task_num == 2:
            self.task_num = 'task2'
            self.train_set = copy.deepcopy(self.data_provider.task2.train)
            self.val_set = copy.deepcopy(self.data_provider.task2.val)
            self.test_sets = [None,
                              copy.deepcopy(self.data_provider.task2.test1),
                              copy.deepcopy(self.data_provider.task2.test2),
                              copy.deepcopy(self.data_provider.task2.test3),
                              copy.deepcopy(self.data_provider.task2.test4)]

        else:
            raise Exception('task num must be one of [1, 2]!')

    def load_embed(self, embed_file_path):
        embed_matrix = []
        with open(embed_file_path) as f:
            data = f.readlines()
            for word_embed in data:
                embed = [float(v_dim) for v_dim in word_embed.split(' ')[1:]]
                embed_matrix.append(embed)

        self.embed_matrix = embed_matrix

    def embedding_layer(self, x, use_glove=True, mask=False):
        if use_glove:
            embed_matrix = tf.Variable(self.embed_matrix)
        else:
            embed_matrix = tf.get_variable('embed_matrix', [self.vocab_size, 300],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           dtype=tf.float32)

        embed = tf.nn.embedding_lookup(embed_matrix, x)
        if mask:
            mask_index = 0
            mask_matrix = np.ones([1, 300])
            mask_matrix[mask_index] = 0
            return embed * mask_matrix
        else:
            return embed

    @staticmethod
    def get_position_encoding(length = 180,hidden_size = 300):
        """
        Cite from tensorflow
        Return positional encoding.
        Calculates the position encoding as a mix of sine and cosine functions with
        geometrically increasing wavelengths.
        Defined and formulized in Attention is All You Need, section 3.5.
        Args:
          length: Sequence length.
          hidden_size: Size of the
          min_timescale: Minimum scale that will be applied at each position
          max_timescale: Maximum scale that will be applied at each position
        Returns:
          Tensor with shape [length, hidden_size]
        """

        min_timescale = 1.0
        max_timescale = 1.0e4
        position = tf.to_float(tf.range(length))
        num_timescales = hidden_size // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return signal

    def apply_positional_encoding(self, x, length, hidden_size):
        """
        Add positional encoding to a batch data 'x' with shape [batch_size, max_words_in_dialog, embed_dim]
        :param x:
        length is the sequence leng
        hidden_size is the hidden dimension of each element in that sequence
        :return:
        """
        PE = self.get_position_encoding(length=length, hidden_size=hidden_size)
        xx = x + PE
        return xx

    @staticmethod
    def length(sequence):
        """

        :param sequence: Input shape: [batch_size, sequence_max_len, feature_dim]
        :return:
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def compute_loss(logits, desired):
        desired = tf.cast(desired, dtype=tf.int32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=desired, logits=logits)
        # cross_entropy = -tf.reduce_mean(desired * tf.log(pred))

        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def compute_accuracy(logits, desired):
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(desired, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return accuracy

    @classmethod
    def timer(cls, func):
        def wrapper(*arg, **kwargs):
            start_time = time.time()
            func(*arg, **kwargs)
            end_time = time.time()
            print('running time {0:f}'.format(end_time - start_time))

        return wrapper

    @staticmethod
    def attention_layer(x, attn_output_dim=1024):
        """
        :param x: inputs of attention layer, required shape: [batch_size, max_sequence_length, feature_dim
        :param attn_output_dim:
        :return: outputs of attention layer
        """

        # align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        # alignment = tf.nn.softmax(align_matrix / 30, 0)
        # context_vector = tf.matmul(x, alignment)
        # x_shape = x.get_shape().as_list()
        # attention_output = tf.layers.dense(tf.reshape(context_vector, [-1, x_shape[1] * x_shape[2]]),  # was 180*256
        #                                    attn_output_dim,
        #                                    activation=tf.nn.tanh)

        align_matrix = tf.matmul(x, tf.einsum('ijk->ikj', x))
        # align_vector = tf.reduce_sum(align_matrix, 1)
        alignment = tf.nn.softmax(align_matrix / 30, 0)
        context = tf.matmul(alignment, x)
        # context = tf.einsum('btk,bt->bk', x, alignment)

        return context


if __name__ == '__main__':

    model = E2EModel(1, 2)
    input = [[1,2,3,4,4,2],
                  [1,2,4,3,2,2],
             [0.1, 0, 0, 0, 0, 0],
                  [1, 2, 4, 3, 2, 2],
                  [0, 0, 0, 0, 0, 0]                  ]
    val_to_keep = 0
    print(np.array(input).shape)
    length = model.length(tf.convert_to_tensor([input, input]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(length))


