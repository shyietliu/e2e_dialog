import tensorflow as tf
import data_provider
import logger
import argparse
from E2E_model import E2EModel
import tqdm


class LSTM(E2EModel):
    def __init__(self, task_num, data_form=2, bidirection=False, attn_usage=True):
        super(LSTM, self).__init__(task_num, data_form)
        self.lstm_hidden_unit_num = 256
        self.output_dim = 20
        self.bidirection = bidirection
        self.attn_usage = attn_usage
        self.learning_rate = None
        self.epochs = None

    def bd_lstm_predictor(self, x):

        # lstm network
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)

        lstm_outputs, last_state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,
                                                                   bw_lstm_cell,
                                                                   x,
                                                                   dtype="float32",
                                                                   sequence_length=self.length(x))

        concated_last_state = tf.concat([last_state[0].h, last_state[1].h], 1)

        # attention (optional)
        if self.attn_usage:
            output = self.attention_layer(concated_last_state)
        else:
            output = concated_last_state

        ff_outputs = tf.layers.dense(output, 512, tf.nn.relu)

        logits = tf.layers.dense(ff_outputs, self.output_dim)

        return logits

    def lstm_predictor(self, x):

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)

        lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype="float32", sequence_length=self.length(x))

        if self.attn_usage:
            output = self.attention_layer(lstm_outputs, attn_output_dim=1024)
        else:
            output = tf.layers.dense(last_state.h, 256, tf.nn.relu)

        logits = tf.layers.dense(output, self.output_dim)

        # fc_inputs = tf.reshape(lstm_outputs, [-1, 160*self.lstm_hidden_unit_num])
        # fc_outputs = tf.layers.dense(fc_inputs, 1024, tf.nn.relu)
        #
        # logits = tf.layers.dense(fc_outputs, self.output_dim)

        return logits

    def train(self, epochs, exp_name, lr, save_model=False):

        print('Save model status: ', save_model)
        # inputs & outputs format
        x = tf.placeholder(tf.int32, [None, 180])
        y = tf.placeholder('float', [None, self.output_dim])

        embed_x = self.embedding_layer(x)

        # construct computation graph
        if self.bidirection:
            pred = self.bd_lstm_predictor(embed_x)
        else:
            pred = self.lstm_predictor(embed_x)

        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            # ini logger
            log_saver = logger.LogSaver(exp_name)
            log_saver.set_log_cate(self.task_num)

            # train
            for epoch in range(epochs):
                for i in range(int(8000/1000)):
                    batch_x, batch_y, _ = self.train_set.next_batch(1000)
                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

                    # print validation information every 40 iteration
                    if i % 4 == 0 and i != 0:
                        train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})
                        train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y})

                        val_x, val_y, _ = self.val_set.next_batch(1000)
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
                #
                # log_saver.train_process_saver([epoch, train_loss, train_acc, val_acc])

                # evaluate on test set per epoch
                for index, test_set in enumerate(self.test_sets):
                    if index > 0:
                        test_x, test_y, _ = test_set.next_batch(1000)
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

