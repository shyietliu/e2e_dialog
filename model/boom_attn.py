import tensorflow as tf
import data_provider
from E2E_model import E2EModel
import logger
import numpy as np
from logger import timer
import time


class AttnNet_1(E2EModel):
    def __init__(self, task_num, data_form=1):
        super(AttnNet_1, self).__init__(task_num, data_form)
        self.hidden_unit_num = 2048
        self.output_dim = 20

    @staticmethod
    def self_attention(x, keep_prob):
        """

        :param x: input with shape [batch_size, sequence_len, embedding_dim] e.g. [100, 180, 300]
        :return: self-attention output with the same shape with input 'x'
        """

        align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        alignment = tf.nn.softmax(align_matrix / 30, 0)
        context_vector = tf.matmul(x, alignment)

        return tf.nn.dropout(context_vector, keep_prob=keep_prob)

    @staticmethod
    def feed_forward(x, layer_num, keep_prob):
        """

        :param x: input for feed forward layer. with shape [batch_size, sequence_len, feature_dim]
        :param layer_num: the number of this feed forward layer
        :return:
        """
        x_shape = x.get_shape().as_list()
        print('!'*20)
        print(x_shape)

        # map to higher dim
        ff_weight_first = tf.get_variable('ff_weight_first_{0}'.format(layer_num),
                                          shape=[x_shape[2], 2000],
                                          initializer=tf.contrib.layers.xavier_initializer())
        ff_bias_first = tf.get_variable('ff_bias_first_{0}'.format(layer_num),
                                        shape=[2000],
                                        initializer=tf.contrib.layers.xavier_initializer())

        output1 = tf.nn.relu(tf.einsum('abc,cd->abd', x, ff_weight_first)+ff_bias_first)

        # map to lower dim
        ff_weight_second = tf.get_variable('ff_weight_second_{0}'.format(layer_num),
                                           shape=[2000, x_shape[2]],
                                           initializer=tf.contrib.layers.xavier_initializer())
        ff_bias_second = tf.get_variable('ff_bias_second_{0}'.format(layer_num),
                                         shape=[x_shape[2]],
                                         initializer=tf.contrib.layers.xavier_initializer())

        output2 = tf.nn.relu(tf.einsum('abc,cd->abd', output1, ff_weight_second)+ff_bias_second)
        # print(output2)
        return tf.nn.dropout(output2, keep_prob=keep_prob)

    @staticmethod
    def add_and_norm(x, x_out):
        """

        :param x: original input for a particular sub-layer
        :param x_out: the output of a sub-layer
        :return: LayerNorm(x + sublayer(x))
        """
        normed_x = tf.contrib.layers.layer_norm(x_out)
        output = normed_x + x
        return output

    def attention_network(self, x, keep_prob, seq_len=30, output_dim=512):
        """
        complete self-attention layer
        :param x: [batch_size, seq_len, feature_dim] e.g. [1000, 180, 300]
        :param keep_prob:
        :return:
        """
        x_shape = x.get_shape().as_list()

        attn_out_1 = self.add_and_norm(x, self.self_attention(x, keep_prob=keep_prob))
        ff_out_1 = self.add_and_norm(attn_out_1, self.feed_forward(attn_out_1, 1, keep_prob=keep_prob))
        print('-' * 20)
        print(ff_out_1)
        # attn_out_2 = self.add_and_norm(ff_out_1, self.self_attention(ff_out_1, keep_prob=keep_prob))
        # ff_out_2 = self.add_and_norm(attn_out_2, self.feed_forward(attn_out_2, 2, keep_prob=keep_prob))
        # pool_output = tf.layers.average_pooling1d(ff_out_1, 2, 2)
        output = tf.layers.dense(tf.reshape(ff_out_1, [-1, seq_len*x_shape[2]]), output_dim, tf.nn.relu)

        return output

    def train(self, epochs, exp_name, lr=1e-3, keep_prob=1.0, save_model=False, mask_input=0):

        # inputs & outputs format
        x = tf.placeholder(tf.int32, [None, 25, 30], name='x')
        y = tf.placeholder('float', [None, self.output_dim], name='y')
        dropout_rate = tf.placeholder('float', [])
        mask = tf.placeholder(tf.float32, [None, 25, 30])
        # construct computation graph
        single_self_attn_output = []
        for utterance_num in range(25):
            with tf.variable_scope('self_attention_{0}'.format(utterance_num)):
                embed_x = self.embedding_layer(x[:, utterance_num, :])
                if mask_input:
                    masked_embed = tf.einsum('abc,ab->abc', embed_x, mask[:, utterance_num, :])
                else:
                    masked_embed = embed_x
                pe_x = (self.apply_positional_encoding(tf.reshape(masked_embed, [-1, 30, 300]), length=30, hidden_size=300))
                single_self_attn_output.append(self.attention_network(pe_x, keep_prob=dropout_rate))  # [batch_size,512]

        context = tf.concat(single_self_attn_output, 1)
        # sentence_representation = tf.reshape(single_self_attn_output, [-1, 25, 512])  # [batch_size, 25, 512]
        #
        # print(sentence_representation)
        # context = self.attention_network(sentence_representation, keep_prob=dropout_rate, output_dim=1024, seq_len=25)

        hidden = tf.layers.dense(context, 1024, tf.nn.relu)
        logits = tf.layers.dense(hidden, self.output_dim)
        loss = self.compute_loss(logits, y)

        accuracy = self.compute_accuracy(logits, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, name='train_op')

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            log_saver = logger.LogSaver(exp_name)
            log_saver.set_log_cate(self.task_num)

            # train
            all_one_mask = np.ones([1000, 25, 30])
            for epoch in range(epochs):
                for i in range(int(8000/100)):
                    batch_x, batch_y, batch_mask = self.train_set.next_batch(100)
                    if mask_input:
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y,
                                                      dropout_rate: keep_prob, mask: batch_mask})
                    else:
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y,
                                                      dropout_rate: keep_prob, mask: all_one_mask})
                    # print validation information every 40 iteration (half epoch)
                    if i % 40 == 0 and i != 0:
                        if mask_input:
                            train_loss = loss.eval(
                                feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob, mask: batch_mask})
                            train_acc = accuracy.eval(
                                feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob, mask: batch_mask})
                        else:
                            train_loss = loss.eval(
                                feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob, mask: all_one_mask})
                            train_acc = accuracy.eval(
                                feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob, mask: all_one_mask})

                        val_x, val_y, _ = self.val_set.next_batch(1000)
                        val_acc = accuracy.eval(feed_dict={
                                        x: val_x,
                                        y: val_y,
                                        dropout_rate: 1.0,
                                        mask: all_one_mask})
                        print('Epoch, {0}, Train loss,{1:2f}, Train acc, {2:3f}, Val_acc,{3:3f}'.format(epoch,
                                                                                                        train_loss,
                                                                                                        train_acc,
                                                                                                        val_acc))
                        log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # evaluate on test set per epoch
                for index, test_set in enumerate(self.test_sets):
                    if index > 0:
                        test_x, test_y, _ = test_set.next_batch(1000)
                        test_acc = sess.run(
                            accuracy, feed_dict={
                                x: test_x,
                                y: test_y,
                                dropout_rate: 1.0,
                                mask: all_one_mask})
                        print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                        # save training log
                        log_saver.test_result_saver([test_acc], index)

            # Model save
            if save_model:
                log_saver.model_saver(sess)


if __name__ == '__main__':
    # DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset'
    #
    model = AttnNet_1(1)
    model.train(10, 'test')
    # tf.print_function(model.length())
    # model.train(100, 'test_save_model')
    pass


