import tensorflow as tf
import data_provider
import logger
import argparse
import numpy as np
from E2E_model import E2EModel


class HierarchicalLSTM(E2EModel):
    def __init__(self, task_num, data_form=1, attn_usage=False):
        super(HierarchicalLSTM, self).__init__(task_num, data_form)
        self.lstm_hidden_unit_num = 256
        self.max_num_utterance = 25
        self.attn_usage = attn_usage

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
                if self.attn_usage:

                    attn_context = self.attention_layer(lstm_outputs, attn_output_dim=256)
                    sentence_repre.append(attn_context)
                else:
                    sentence_repre.append(last_state.h)

            if self.attn_usage:
                return tf.reshape(tf.concat(sentence_repre, 1), [-1, self.max_num_utterance, 256])
            else:
                return tf.reshape(tf.concat(sentence_repre, 1), [-1, self.max_num_utterance, self.lstm_hidden_unit_num])

        def sentence_level_lstm(inputs):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num,
                                                     forget_bias=1,
                                                     activation=tf.nn.relu)

            lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype="float32",
                                                         sequence_length=self.length(inputs))

            output = self.attention_layer(lstm_outputs, attn_output_dim=512)
            # return last_state.h
            return output

        sentence_representation = word_level_lstm(x)
        h_lstm_outputs = sentence_level_lstm(sentence_representation)

        ff_inputs = h_lstm_outputs

        ff_outputs = tf.layers.dense(ff_inputs, 512, tf.nn.relu)

        logits = tf.layers.dense(ff_outputs, self.output_dim)

        # logits = tf.layers.dense(h_lstm_outputs, self.output_dim)

        return logits

    def train(self, epochs, exp_name, lr, save_model=False, mask_input=0):

        print('-'*30)
        print('start training ...')
        print('Save model status: ', save_model)
        # inputs & outputs format
        x = tf.placeholder(tf.int32, [None, 25, 30])
        y = tf.placeholder('float', [None, self.output_dim])

        # if mask_input:
        mask = tf.placeholder(tf.float32, [None, 25, 30])

        # construct computation graph
        embed_x = self.embedding_layer(x)

        if mask_input:
            embed_x = tf.einsum('abcd,abc->abcd', embed_x, mask)

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
            all_one_mask = np.ones([1000, 25, 30])
            #
            all_one_mask_for_training = np.ones([100, 25, 30])

            for epoch in range(epochs):
                for i in range(int(8000/100)):
                    if mask_input:
                        batch_x, batch_y, batch_mask = self.train_set.next_batch(100, mask_input=True)
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y, mask: batch_mask})
                    else:
                        batch_x, batch_y, _ = self.train_set.next_batch(100, mask_input=False)
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y, mask: all_one_mask_for_training})

                    # print validation information every 40 iteration (half epoch)
                    if i % 40 == 0 and i != 0:
                        if mask_input:
                            train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y, mask: batch_mask})
                            train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, mask: batch_mask})
                        else:
                            train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y, mask: all_one_mask_for_training})
                            train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, mask: all_one_mask_for_training})

                        val_x, val_y, _ = self.val_set.next_batch(1000)

                        val_acc = accuracy.eval(feed_dict={
                                        x: val_x,
                                        y: val_y,
                                        mask: all_one_mask})

                        # val_acc = accuracy.eval(feed_dict={
                        #                 x: val_x,
                        #                 y: val_y
                        #                 })
                        print('Epoch, {0}, Train loss,{1:2f}, Train acc, {2:3f}, Val_acc,{3:3f}'.format(epoch,
                                                                                                        train_loss,
                                                                                                        train_acc,
                                                                                                        val_acc))

                        log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # evaluate on test set per epoch
                for index, test_set in enumerate(self.test_sets):
                    if index > 0:
                        test_x, test_y, _ = test_set.next_batch(1000)
                        # test_acc = sess.run(
                        #     accuracy, feed_dict={
                        #         x: test_x,
                        #         y: test_y,
                        #         mask: all_one_mask})

                        test_acc = sess.run(
                            accuracy, feed_dict={
                                x: test_x,
                                y: test_y,
                                mask: all_one_mask
                                })
                        print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                        # save training log
                        log_saver.test_result_saver([test_acc], index)

            # Model save
            if save_model:
                log_saver.model_saver(sess)


if __name__ == '__main__':

    pass


