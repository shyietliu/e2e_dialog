import tensorflow as tf
import data_provider
import logger
import argparse
import numpy as np
from E2E_model import E2EModel


class ConvolutionLSTM(E2EModel):
    def __init__(self, data_path, bidirection=False, attn_usage=True):
        super(ConvolutionLSTM, self).__init__(data_path)
        self.lstm_hidden_unit_num = 128
        self.output_dim = 19
        self.path = data_path
        self.bidirection = bidirection
        self.attn_usage = attn_usage
        self.learning_rate = None
        self.epochs = None

    def conv_lstm_predictor(self, x):
        """
        Convolution LSTM
        :param x: inputs in shape [batch_size, num_utterance, sequence_max_len, vocab_size]
        :return: logits
        """
        x = tf.transpose(x, [0, 2, 1, 3])
        pass
        tiled_x = tf.tile(x, [1, 6, 1, 1])

        xx = tf.concat([tiled_x, np.zeros([100, 1, 25, 181])], axis=1)

        p_input_list = tf.split(xx, 25, 2)
        # a, b, c, d, e = p_input_list = tf.split(xx, 5, 1)
        p_input_list = [tf.squeeze(p_input_, [2]) for p_input_ in p_input_list]

        conv_lstm_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1,
                                                     input_shape=[181, 181],
                                                     output_channels=2,
                                                     kernel_shape=[2, 2],
                                                     skip_connection=False
                                                     )

        # state = tf.constant(0.0, shape=[100, 181, 181])

        state = conv_lstm_cell.zero_state(100, dtype=tf.float32)

        with tf.variable_scope("ConvLSTM") as scope:  # as BasicLSTMCell # create the RNN with a loop
            for i, p_input_ in enumerate(p_input_list):
                if i > 0:
                    scope.reuse_variables()
                # ConvCell takes Tensor with size [batch_size, height, width, channel].
                t_output, state = conv_lstm_cell(p_input_, state)

        logits = tf.layers.dense(tf.reshape(state.h, [-1, 30*181*3]))

        return logits

    def bd_lstm_predictor(self, x):

        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)

        lstm_outputs, last_state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,
                                                                   bw_lstm_cell,
                                                                   x,
                                                                   dtype="float32",
                                                                   sequence_length=self.length(x))

        ff_inputs = tf.concat([last_state[0].h, last_state[1].h], 1)
        #
        ff_outputs = tf.layers.dense(ff_inputs, 512, tf.nn.relu)

        logits = tf.layers.dense(ff_outputs, self.output_dim)

        return logits

    def lstm_predictor(self, x):

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)

        lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype="float32", sequence_length=self.length(x))

        if self.attn_usage:
            output = self.attention_layer(lstm_outputs, attn_output_dim=1024)
        else:
            output = tf.layers.dense(last_state.h, 128, tf.nn.relu)

        logits = tf.layers.dense(output, self.output_dim)

        # fc_inputs = tf.reshape(lstm_outputs, [-1, 160*self.lstm_hidden_unit_num])
        # fc_outputs = tf.layers.dense(fc_inputs, 1024, tf.nn.relu)
        #
        # logits = tf.layers.dense(fc_outputs, self.output_dim)

        return logits

    def train(self, epochs, exp_name, lr, save_model=False):

        print(save_model)
        # inputs & outputs format
        x = tf.placeholder(tf.float32, [None, 25, 30, 181])
        y = tf.placeholder('float', [None, self.output_dim])

        self.learning_rate = lr
        self.epochs = epochs

        # construct computation graph
        pred = self.conv_lstm_predictor(x)

        pass
        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            dstc_data = data_provider.DataProvider(self.path, data_form=1)
            log_saver = logger.LogSaver(exp_name)
            log_saver.set_log_cate('task1')

            # train
            for epoch in range(epochs):
                batch_x, batch_y = dstc_data.train.task1.next_batch(100)
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

                # validating
                if epoch % 10 == 0 and epoch != 0:
                    train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})

                    val_x, val_y = dstc_data.test1.task1.next_batch(100)
                    val_acc = accuracy.eval(feed_dict={
                                    x: val_x,
                                    y: val_y})
                    print('Training {0} epoch, validation accuracy is {1}, training loss is {2}'.format(epoch,
                                                                                                        val_acc,
                                                                                                        train_loss))

                    log_saver.train_process_saver([epoch, train_loss, val_acc])

            # evaluate
            test_sets = [None,
                         dstc_data.test1,
                         dstc_data.test2,
                         dstc_data.test3,
                         dstc_data.test4]

            for index, test_set in enumerate(test_sets):
                if index > 0:
                    test_x, test_y = test_set.task1.next_batch(100)
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

    parser = argparse.ArgumentParser(description='Train BidirectionalLSTM saved_model')
    parser.add_argument('-exp_name', dest='exp_name')
    parser.add_argument('-train_epoch', dest='train_epoch', type=int)
    parser.add_argument('-lr', dest='learning_rate', type=float)

    args = parser.parse_args()

    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset'

    model = ConvolutionLSTM(DATA_PATH)
    train_epoch = args.train_epoch
    learning_rate = args.learning_rate
    exp_name = args.exp_name+'_'+str(train_epoch)+'_lr_'+str(learning_rate)
    model.train(train_epoch, exp_name, learning_rate)
    pass

