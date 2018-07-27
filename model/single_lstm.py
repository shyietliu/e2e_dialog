import tensorflow as tf
import data_provider
import logger
import argparse
from E2E_model import E2EModel


class LSTM(E2EModel):
    def __init__(self, data_path):
        super(LSTM, self).__init__(data_path)
        self.lstm_hidden_unit_num = 128
        self.output_dim = 8
        self.path = data_path

    def lstm_predictor(self, x, lstm_hidden_unit_num, output_dim):
        """

        :param x: [batch_size, seq_len, feature_dim]
        :return: logits
        """

        # x_shape = x.get_shape().as_list()

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_unit_num, forget_bias=1, activation=tf.nn.relu)

        lstm_outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype="float32", sequence_length=self.length(x))

        # fc_inputs = tf.reshape(lstm_outputs, [-1, x_shape[1]*lstm_hidden_unit_num])
        #
        # fc_outputs = tf.layers.dense(fc_inputs, 1024, tf.nn.relu)

        # logits = tf.layers.dense(fc_outputs, output_dim)

        return last_state.h

    def train(self, train_epoch, exp_name, lr):

        # inputs & outputs format
        x = tf.placeholder(tf.float32, [None, 160, 180])
        y = tf.placeholder('float', [None, 8])

        # x_ = tf.unstack(x, 160, 1)

        # construct computation graph
        pred = self.lstm_predictor(x)
        loss = self.compute_loss(pred, y)

        accuracy = self.compute_accuracy(pred, y)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        with tf.Session() as sess:
            # initialization
            init = tf.global_variables_initializer()
            sess.run(init)

            dstc_data = data_provider.DataProvider(self.path, data_form=2)
            log_saver = logger.LogSaver(exp_name)
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
                    # self.save_log([epoch, train_loss, val_acc], exp_name)

            # evaluate
            test_sets = [None,
                         dstc_data.test1,
                         dstc_data.test2,
                         dstc_data.test3,
                         dstc_data.test4]

            for index, test_set in enumerate(test_sets):
                test_x, test_y = test_set.task1.next_batch(100)
                test_acc = sess.run(
                    accuracy, feed_dict={
                        x: test_x,
                        y: test_y})
                print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                # save training log
                log_saver.test_result_saver([test_acc], index)


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
    # dstc_data = data_provider.DataProvider(DATA_PATH, data_form=2)
    # batch_x, batch_y = dstc_data.train.task1.next_batch(10)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     length = saved_model.length(batch_x).eval()
    # print(length)
    pass



#
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.placeholder("float", [None, 10])
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print (sess.run(accuracy, feed_dict={x: mnist.test.images[0:1000], y_: mnist.test.labels[0:1000]}))
