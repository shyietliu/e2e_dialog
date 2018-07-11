import tensorflow as tf
from data_provider import DataProvider
import logger
import argparse
import copy


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

        self.embed_matrix = None
        self.load_embed('../my_dataset/sub_glove_embedding.txt')

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

    def embedding_layer(self, x, use_glove=True):
        if use_glove:
            embed_matrix = tf.Variable(self.embed_matrix)
        else:
            embed_matrix = tf.get_variable('embed_matrix', [189, 300],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           dtype=tf.float32)

        embed = tf.nn.embedding_lookup(embed_matrix, x)
        return embed

    @staticmethod
    def length(sequence):
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

    @staticmethod
    def attention_layer(x, attn_output_dim=1024):
        """
        :param x: inputs of attention layer, required shape: [batch_size, max_sequence_length, feature_dim
        :param attn_output_dim:
        :return: outputs of attention layer
        """

        align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        alignment = tf.nn.softmax(align_matrix, 0)
        context_vector = tf.matmul(x, alignment)
        attention_output = tf.layers.dense(tf.reshape(context_vector, [-1, 180 * 128]),  # was 160*128
                                           attn_output_dim,
                                           activation=tf.nn.tanh)

        return attention_output


if __name__ == '__main__':

    model = E2EModel(1, 1)
    train = model.train_set
    val = model.val_set
    test = model.test_sets

    train.current_path()
    val.current_path()
    for i,tst in enumerate(test):
        if i>0:
            tst.current_path()

