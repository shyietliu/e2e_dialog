import tensorflow as tf
import data_provider_2
from E2E_model import E2EModel
import logger


class MultiLayerPerceptron(E2EModel):
    def __init__(self, task_num, data_form=1):
        """

        :param data_path:
        :param task_num: train model on which task
        """
        super(MultiLayerPerceptron, self).__init__(task_num, data_form)
        self.hidden_unit_num = 2048
        self.output_dim = 20
        # self.path = data_path

    def multi_layer_softmax_classifier(self, x):
        hidden = tf.layers.dense(x, self.hidden_unit_num, tf.nn.relu)
        # hidden = self.attention_layer(hidden, attn_output_dim=2048)
        logits = tf.layers.dense(hidden, self.output_dim, name='logits')

        return logits

    def train(self, epochs, exp_name, lr=1e-4, save_model=False):

        # inputs & outputs format
        x = tf.placeholder(tf.float32, [None, 25, 30, 188], name='x')
        y = tf.placeholder('float', [None, self.output_dim], name='y')

        # construct computation graph
        logits = self.multi_layer_softmax_classifier(tf.reshape(x, [-1, 25*30*188]))
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
            for epoch in range(epochs):
                batch_x, batch_y = self.train_set.next_batch(100)
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

                # validating
                if epoch % 10 == 0:
                    train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})

                    val_x, val_y = self.val_set.next_batch(100)
                    val_acc = accuracy.eval(feed_dict={
                                    x: val_x,
                                    y: val_y})
                    print('Training {0} epoch, validation accuracy is {1}, training loss is {2}'.format(epoch,
                                                                                                        val_acc,
                                                                                                        train_loss))
                    # save train process
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


if __name__ == '__main__':

    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
    pass


