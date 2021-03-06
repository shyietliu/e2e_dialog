import tensorflow as tf
import data_provider
import logger
import argparse
import numpy as np
from E2E_model import E2EModel


class HierarchicalLSTM(E2EModel):
    def __init__(self, task_num, data_form=1):
        super(HierarchicalLSTM, self).__init__(task_num, data_form)
        self.lstm_hidden_unit_num = 256
        self.max_num_utterance = 25

    def lstm_predictor(self, x):
        def word_level_lstm(inputs):
            lstm_cells = {}
            sentence_repre = []
            for i in range(self.max_num_utterance):
                lstm_cells[i] = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num,
                                                             forget_bias=1,
                                                             activation=tf.nn.relu)
                with tf.variable_scope('word_level_lstm'+str(i)):
                    lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cells[i], inputs[:, i, :, :], dtype="float32",
                                                                 sequence_length=self.length(inputs[:, i, :, :]))

                sentence_repre.append(last_state.h)
            return tf.reshape(tf.concat(sentence_repre, 1), [-1, self.max_num_utterance, self.lstm_hidden_unit_num])

        def sentence_level_lstm(inputs):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num,
                                                     forget_bias=1,
                                                     activation=tf.nn.relu)

            lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype="float32",
                                                         sequence_length=self.length(inputs))

            return last_state.h

        sentence_representation = word_level_lstm(x)
        h_lstm_outputs = sentence_level_lstm(sentence_representation)

        ff_inputs = h_lstm_outputs

        ff_outputs = tf.layers.dense(ff_inputs, 512, tf.nn.relu)

        logits = tf.layers.dense(ff_outputs, self.output_dim)

        return logits

    def train(self, epochs, exp_name, lr, save_model=False):

        print('-'*30)
        print('start training ...')
        print('Save model status: ', save_model)
        # inputs & outputs format
        x = tf.placeholder(tf.int32, [None, 25, 30])
        y = tf.placeholder('float', [None, self.output_dim])

        # construct computation graph
        embed_x = self.embedding_layer(x)

        pred = self.lstm_predictor(embed_x)

        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            # init logger
            log_saver = logger.LogSaver(exp_name)
            log_saver.set_log_cate(self.task_num)

            # train
            for epoch in range(epochs):
                for i in range(int(8000/100)):
                    batch_x, batch_y = self.train_set.next_batch(100)
                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

                    # print validation information every 40 iteration (half epoch)
                    if i % 40 == 0 and i != 0:
                        train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})
                        train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y})

                        val_x, val_y = self.val_set.next_batch(1000)
                        val_acc = accuracy.eval(feed_dict={
                                        x: val_x,
                                        y: val_y})
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

                # log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # evaluate on test set per epoch
                for index, test_set in enumerate(self.test_sets):
                    if index > 0:
                        test_x, test_y = test_set.next_batch(1000)
                        test_acc = sess.run(
                            accuracy, feed_dict={
                                x: test_x,
                                y: test_y})
                        print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                        # save training log
                        log_saver.test_result_saver([test_acc], index)

            # Model save
            if save_model:
                log_saver.model_saver(sess)


if __name__ == '__main__':

    pass


