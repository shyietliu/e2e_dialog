import tensorflow as tf
import data_provider
import logger
import argparse
from E2E_model import E2EModel


class HierarchicalLSTM(E2EModel):
    def __init__(self, data_path, task_num, data_form=1):
        super(HierarchicalLSTM, self).__init__(data_path, task_num, data_form)
        self.lstm_hidden_unit_num = 128
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

    def train(self, train_epoch, exp_name, lr, save_model=False):

        dstc_data = data_provider.DataProvider(self.path, data_form=1)
        log_saver = logger.LogSaver(exp_name)

        # inputs & outputs format
        x = tf.placeholder(tf.float32, [None,
                                        dstc_data.max_num_utterance,
                                        dstc_data.sequence_max_len,
                                        181], name='x')

        y = tf.placeholder('float', [None, self.output_dim], name='y')

        # construct computation graph
        pred = self.lstm_predictor(x)
        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)
            log_saver.set_log_cate('task1')

            # train
            for epoch in range(train_epoch):
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
            for index, test_set in enumerate(self.test_sets):
                if index>0:
                    test_x, test_y = test_set.next_batch(100)
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

    model = HierarchicalLSTM(DATA_PATH)
    train_epoch = args.train_epoch
    learning_rate = args.learning_rate
    exp_name = args.exp_name+'_'+str(train_epoch)+'_lr_'+str(learning_rate)
    model.train(train_epoch, exp_name, learning_rate)

    pass


