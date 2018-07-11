import tensorflow as tf
import data_provider
from E2E_model import E2EModel
import logger


class AttnNet(E2EModel):
    def __init__(self, data_path, task_num, data_form=2):
        super(AttnNet, self).__init__(task_num, data_form)
        self.hidden_unit_num = 2048
        self.output_dim = 20
        self.path = data_path

    @staticmethod
    def self_attention(x, attn_output_dim):
        align_matrix = tf.matmul(tf.einsum('ijk->ikj', x), x)
        alignment = tf.nn.softmax(align_matrix, 0)
        context_vector = tf.matmul(x, alignment)
        attention_output = tf.layers.dense(tf.reshape(tf.concat([x, context_vector], 1),
                                                      [-1, 2 * 160 * 188]),
                                           1024,
                                           activation=tf.nn.tanh)
        return attention_output

    def attention_network(self, x):

        # embed_x = self.embedding_layer(x)
        context_matrix = self.self_attention(x, attn_output_dim=1024)
        ff = tf.layers.dense(context_matrix, 512, tf.nn.relu)
        # hidden = self.attention_layer(hidden, attn_output_dim=2048)
        logits = tf.layers.dense(ff, self.output_dim, name='logits')

        return logits

    def train(self, epochs, exp_name, lr=1e-3, save_model=False):

        # inputs & outputs format
        x = tf.placeholder(tf.float32, [None, 160, 188], name='x')
        y = tf.placeholder('float', [None, self.output_dim], name='y')

        # construct computation graph
        logits = self.attention_network(x)
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
                if epoch % 10 == 0 and epoch != 0:
                    train_loss = loss.eval(feed_dict={x: batch_x, y: batch_y})

                    val_x, val_y = self.val_set.next_batch(1000)
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


if __name__ == '__main__':
    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset'

    model = AttnNet(DATA_PATH, 1)
    model.train(100, 'test_save_model')
    pass


