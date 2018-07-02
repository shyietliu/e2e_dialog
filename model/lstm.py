import tensorflow as tf
import data_provider
import logger
import argparse
from E2E_model import E2EModel


class LSTM(E2EModel):
    def __init__(self, task_num, data_form=2, bidirection=False, attn_usage=True):
        super(LSTM, self).__init__(task_num, data_form)
        self.lstm_hidden_unit_num = 128
        self.output_dim = 20
        self.bidirection = bidirection
        self.attn_usage = attn_usage
        self.learning_rate = None
        self.epochs = None

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
        x = tf.placeholder(tf.float32, [None, 160, 188])
        y = tf.placeholder('float', [None, self.output_dim])

        self.learning_rate = lr
        self.epochs = epochs

        # construct computation graph
        if self.bidirection:
            pred = self.bd_lstm_predictor(x)
        else:
            pred = self.lstm_predictor(x)

        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            log_saver = logger.LogSaver(exp_name)
            log_saver.set_log_cate(self.task_num)

            # train
            for epoch in range(epochs):
                batch_x, batch_y = self.train_set.next_batch(100)
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

                # validating
                if epoch % 10 == 0 and epoch != 0:
                    train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})

                    val_x, val_y = self.val_set.next_batch(100)
                    val_acc = accuracy.eval(feed_dict={
                                    x: val_x,
                                    y: val_y})
                    print('Training {0} epoch, validation accuracy is {1}, training loss is {2}'.format(epoch,
                                                                                                        val_acc,
                                                                                                        train_loss))

                    log_saver.train_process_saver([epoch, train_loss, val_acc])

            # evaluate
            for index, test_set in enumerate(self.test_sets):
                if index > 0:
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

    @property
    def name(self):

        model_name = self.__class__.__name__
        attn_name = self.attn_usage
        bd_name = self.bidirection
        lr_name = self.learning_rate
        epoch_name = self.epochs
        ll = [model_name, attn_name, bd_name, lr_name, epoch_name]
        print('_'.join(str(ll)))
        return 'test'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train BidirectionalLSTM saved_model')
    parser.add_argument('-exp_name', dest='exp_name')
    parser.add_argument('-train_epoch', dest='train_epoch', type=int)
    parser.add_argument('-lr', dest='learning_rate', type=float)

    args = parser.parse_args()

    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset'

    model = LSTM(DATA_PATH)
    train_epoch = args.train_epoch
    learning_rate = args.learning_rate
    exp_name = args.exp_name+'_'+str(train_epoch)+'_lr_'+str(learning_rate)
    model.train(train_epoch, exp_name, learning_rate)
    pass

