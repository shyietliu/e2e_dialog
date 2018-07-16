import tensorflow as tf
import data_provider
from E2E_model import E2EModel
import logger


class AttnNet(E2EModel):
    def __init__(self, task_num, data_form=1):
        super(AttnNet, self).__init__(task_num, data_form)
        self.hidden_unit_num = 2048
        self.output_dim = 20

    @staticmethod
    def self_attention(x, keep_prob):
        """

        :param x: input with shape [batch_size, sequence_len, embedding_dim] e.g. [100, 180, 300]
        :return: self-attention output with the same shape with input 'x'
        """
        align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        alignment = tf.nn.softmax(align_matrix, 0)
        context_vector = tf.matmul(x, alignment)

        return tf.nn.dropout(context_vector, keep_prob=keep_prob)

    @staticmethod
    def feed_forward(x, layer_num, keep_prob):
        """

        :param x: input for feed forward layer.
        :param layer_num: the number of this feed forward layer
        :return:
        """
        # map to higher dim
        ff_weight_first = tf.get_variable('ff_weight_first_{0}'.format(layer_num),
                                          shape=[300, 2000],
                                          initializer=tf.contrib.layers.xavier_initializer())
        ff_bias_first = tf.get_variable('ff_bias_first_{0}'.format(layer_num),
                                        shape=[2000],
                                        initializer=tf.contrib.layers.xavier_initializer())

        output1 = tf.nn.relu(tf.einsum('abc,cd->abd', x, ff_weight_first)+ff_bias_first)

        # map to lower dim
        ff_weight_second = tf.get_variable('ff_weight_second_{0}'.format(layer_num),
                                           shape=[2000, 300],
                                           initializer=tf.contrib.layers.xavier_initializer())
        ff_bias_second = tf.get_variable('ff_bias_second_{0}'.format(layer_num),
                                         shape=[300],
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

    def hierarchical_attention_network(self, x, keep_prob):
        sentence_representation = []
        def word_level_attn_net(dialog):
            for i in range(self.max_num_utterance):
                utterance = dialog[:, i, :, :]
                with tf.variable_scope('word_level_attn_net' + str(i)):
                    attn_out = self.add_and_norm(utterance, self.self_attention(utterance, keep_prob=keep_prob))
                    ff_out = self.add_and_norm(attn_out_1, self.feed_forward(attn_out_1, 1, keep_prob=keep_prob))
                sentence_representation.append(ff_out)

            return tf.convert_to_tensor(sentence_representation)

        def sentence_level_attn_net(representation):
            pass




        attn_out_1 = self.add_and_norm(x, self.self_attention(x, keep_prob=keep_prob))
        ff_out_1 = self.add_and_norm(attn_out_1, self.feed_forward(attn_out_1, 1, keep_prob=keep_prob))

        hidden_1 = tf.layers.dense(tf.reshape(ff_out_1, [-1, 180*300]), 512, tf.nn.relu)

        logits = tf.layers.dense(hidden_1, self.output_dim, name='logits')

        return logits

    def train(self, epochs, exp_name, lr=1e-3, keep_prob=0.8, save_model=False):

        # inputs & outputs format
        x = tf.placeholder(tf.int32, [None, 25, 30], name='x')
        y = tf.placeholder('float', [None, self.output_dim], name='y')
        dropout_rate = tf.placeholder('float', [])

        # construct computation graph
        embed_x = self.embedding_layer(x)  # shape [Batch_size, 180, 300]

        # pe_x = self.apply_positional_encoding(embed_x)

        logits = self.hierarchical_attention_network(embed_x, keep_prob=dropout_rate)

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
            # train
            for epoch in range(epochs):
                for i in range(int(8000/1000)):
                    batch_x, batch_y = self.train_set.next_batch(1000)
                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})

                    # print validation information every 40 iteration (half epoch)
                    if i % 4 == 0 and i != 0:
                        train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})
                        train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})

                        val_x, val_y = self.val_set.next_batch(1000)
                        val_acc = accuracy.eval(feed_dict={
                                        x: val_x,
                                        y: val_y,
                                        dropout_rate: 1.0})
                        print('Epoch, {0}, Train loss,{1:2f}, Train acc, {2:3f}, Val_acc,{3:3f}'.format(epoch,
                                                                                                        train_loss,
                                                                                                        train_acc,
                                                                                                        val_acc))
                        log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # save evaluation result per epoch
                # train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})
                # train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                #
                # val_x, val_y = self.val_set.next_batch(1000)
                # val_acc = accuracy.eval(feed_dict={
                #     x: val_x,
                #     y: val_y})
                #
                # log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # evaluate on test set per epoch
                for index, test_set in enumerate(self.test_sets):
                    if index > 0:
                        test_x, test_y = test_set.next_batch(1000)
                        test_acc = sess.run(
                            accuracy, feed_dict={
                                x: test_x,
                                y: test_y,
                                dropout_rate: 1.0})
                        print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                        # save training log
                        log_saver.test_result_saver([test_acc], index)

            # Model save
            if save_model:
                log_saver.model_saver(sess)


if __name__ == '__main__':
    # DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset'
    #
    # model = AttnNet(1, 2)
    # model.train(100, 'test_save_model')
    pass


