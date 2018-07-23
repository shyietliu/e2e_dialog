import tensorflow as tf
import data_provider
from E2E_model import E2EModel
import logger


class MultiLayerPerceptron(E2EModel):
    def __init__(self, task_num, data_form=1):
        """

        :param data_path:
        :param task_num: train model on which task
        """
        super(MultiLayerPerceptron, self).__init__(task_num, data_form)
        self.hidden_unit_num = 512
        self.output_dim = 20
        self.data_form = data_form
        # self.path = data_path

    def multi_layer_softmax_classifier(self, x):
        hidden = tf.layers.dense(x, self.hidden_unit_num, tf.nn.relu)
        # hidden = self.attention_layer(hidden, attn_output_dim=2048)
        logits = tf.layers.dense(hidden, self.output_dim, name='logits')

        return logits

    def train(self, epochs, exp_name, lr=1e-4, keep_prob=0.8, save_model=False):

        print('-' * 20, 'save model state', '-' * 20)
        print(save_model)

        # inputs & outputs format
        if self.data_form == 1:
            x = tf.placeholder(tf.int32, [None, 25, 30], name='x')
        elif self.data_form == 2:
            x = tf.placeholder(tf.int32, [None, self.max_num_words_in_dialog], name='x')

        y = tf.placeholder('float', [None, self.output_dim], name='y')
        dropout_rate = tf.placeholder('float', [])

        # construct computation graph
        embed_x = self.embedding_layer(x)
        if self.data_form == 1:
            flatten_embed = tf.reshape(embed_x, [-1, 25 * 30 * 300])
        elif self.data_form == 2:
            flatten_embed = tf.reshape(embed_x, [-1,  self.max_num_words_in_dialog * 300])
        logits = self.multi_layer_softmax_classifier(flatten_embed)
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
            with tf.Session() as sess:
                # initialization
                init = tf.global_variables_initializer()
                sess.run(init)

                log_saver = logger.LogSaver(exp_name)
                log_saver.set_log_cate(self.task_num)

                # train
                for epoch in range(epochs):
                    for i in range(int(8000 / 1000)):
                        batch_x, batch_y, _ = self.train_set.next_batch(1000)
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})

                        # print validation information every 40 iteration (half epoch)
                        if i % 4 == 0 and i != 0:
                            train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})
                            train_acc = accuracy.eval(feed_dict={x: batch_x, y: batch_y, dropout_rate: keep_prob})

                            val_x, val_y,_ = self.val_set.next_batch(1000)
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
                            test_x, test_y,_ = test_set.next_batch(1000)
                            test_acc = sess.run(
                                accuracy, feed_dict={
                                    x: test_x,
                                    y: test_y,
                                    dropout_rate: 1.0})
                            print('test accuracy on test set {0} is {1}'.format(index, test_acc))
                            # save training log
                            log_saver.test_result_saver([test_acc], index)

                # Model save
                print('-'*20,'save model state', '-'*20)
                print(save_model)
                if save_model:
                    log_saver.model_saver(sess)


if __name__ == '__main__':

    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
    pass


