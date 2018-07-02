import tensorflow as tf
from data_provider import DataProvider

model_path = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/' \
             'exp_log/task1/test_save_model/saved_model/test_save_model.ckpt.meta'
saver = tf.train.import_meta_graph(model_path)
with tf.Session() as sess:

    saver.restore(sess, tf.train.latest_checkpoint('/afs/inf.ed.ac.uk/user/s17/'
                                                   's1700619/E2E_dialog/exp_log/task1/test_save_model/saved_model'))

    print('Model restored!')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')

    loss = graph.get_tensor_by_name('loss:0')
    acc = graph.get_tensor_by_name('accuracy:0')

    data_provider = DataProvider('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset', data_form=1)
    batch_x, batch_y = data_provider.test4.task1.next_batch(1000)
    accuracy = sess.run(acc, feed_dict={x: batch_x, y: batch_y})
    print(accuracy)
