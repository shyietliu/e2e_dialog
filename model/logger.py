import os
import tensorflow as tf

class LogSaver(object):
    def __init__(self, exp_name, log_path=None):
        if log_path is None:
            self.log_path = '../exp_log/'

        self.exp_name = exp_name
        self.file_path = None  # the path of training process log
        self.model_path = None  # the path of trained tf saved_model
        self.model_file_path = None

    def set_log_cate(self, task_cat):
        """
        Setting the path of saved log
        :param task_cat:
        :return:
        """

        if task_cat not in ['task1', 'task2', 'task3', 'task4', 'task5']:
            raise Exception('task category must be \'task#\' ')

        # if tst_set_cat not in ['test_set_1', 'test_set_2', 'test_set_3', 'test_set_4']:
        #     raise Exception('task category must be \'test_set_#\' ')

        # set log directory, default: '../dataset/exp_name/task#/'
        self.log_path = os.path.join(self.log_path, task_cat)
        # set log directory, default: '../dataset/exp_name/task#/exp_name'
        self.log_path = os.path.join(self.log_path, self.exp_name)
        # set log directory, default: '../dataset/exp_name/task#/exp_name/saved_model/'
        self.model_path = os.path.join(self.log_path, 'saved_model/')
        # set log directory, default: '../dataset/exp_name/task#/exp_name/log/'
        self.log_path = os.path.join(self.log_path, 'log/')

        # set log file path
        file_name = self.exp_name + '_log.txt'
        self.file_path = os.path.join(self.log_path, file_name)
        # print('\n', self.file_path, '\n')

        # set model file path
        model_file_name = self.exp_name + '.ckpt'
        self.model_file_path = os.path.join(self.model_path, model_file_name)

    def train_process_saver(self, information):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(self.file_path, 'a') as f:
            print('Epoch,{0}, Train loss,{1}, Val_acc,{2}'.format(information[0],
                                                                  information[1],
                                                                  information[2]), file=f)

    def test_result_saver(self, information, test_set_index):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(self.file_path, 'a') as f:
            print('Test acc on test set {0}, {1}'.format(test_set_index, information[0]), file=f)

    def model_saver(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.model_file_path)
