import json
import numpy as np
import tensorflow as tf
import os
import data_provider
import copy
import time


def reform_test_data(test_data_path, test_answer_path, reformed_data_saved_path):
    """
    Create a new json file that is compatible with the training data form
    :param test_data_path: file path of test set data
    :param test_answer_path: file path of the answer of test set
    :param reformed_data_saved_path: file path that the new data will be saved
    :return: None
    """
    with open(test_data_path) as f:
        test_data = json.load(f)
    with open(test_answer_path) as f:
        answers = json.load(f)

    for each_dialog, each_answer in zip(test_data, answers):
        if not each_answer['dialog_id'] == each_dialog['dialog_id']:
            raise Exception('Not match!!')
        for candidate in each_dialog['candidates']:
            if candidate['candidate_id'] == each_answer['lst_candidate_id'][0]['candidate_id']:
                # each_dialog['answer'] = {}
                each_dialog['answer'] = candidate
    with open(reformed_data_saved_path, 'w') as f:
        json.dump(test_data, f)


def count_answer_category(file_path, saved_path):
    """
    count answer category in 'file_path'
    :param file_path:
    :param saved_path:
    :return:
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    with open('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/task1_answer_category.txt') as f:
        answer_categories = f.readlines()
    print(answer_categories)
    print('--------------------')
    for dialog in data:
        answer = dialog['answer']['utterance']
        if answer[0:8] == 'api_call':
            answer = 'api_call'
        if answer[0:10] == 'here it is':
            answer = 'here it is'
        if answer[0:33] == 'what do you think of this option:':
            answer = 'what do you think of this option'
        if answer[0:14] == 'the option was':
            answer = 'the option was'

        if (answer+'\n') not in answer_categories:
            answer_categories.append(answer+'\n')

    for ans in answer_categories:
        print(ans)

    with open('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/task1_answer_category.txt', 'w') as f:
        for ele in answer_categories:
            f.write(ele)


def timer(func):
    def wrapper(*arg, **kwargs):
        start_time = time.time()
        func(*arg, **kwargs)
        end_time = time.time()
        print('running time {0:f}'.format(end_time-start_time))

    return wrapper



class MyClass(object):
    def __init__(self):
        pass



    @timer
    def inner_function(self):
        print('running inner function! {0}, {1}')




if __name__ == '__main__':
    # with open('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/task1/val/val_data.json', 'r') as f:
    #     train_data = json.load(f)
    #
    # for dialog in train_data:
    #     answer = dialog['answer']['utterance']
    #     # if answer[0:2] == 'do':
    #     print(answer)
    #
    # with open('../my_dataset/sub_glove_embedding.txt', 'a') as f:
    #     a = np.round(np.random.normal(0, 0.01, [1, 300]), decimals=5).tolist()
    #
    #     for ele in a:
    #         for v in ele:
    #             f.write(' '+str(v))
    #         f.write('\n')
    # reform_test_data('/Users/shyietliu/python/E2E/e2e_dialog/my_dataset/dataset-E2E-goal-oriented-test-v1.0/tst4/dialog-task2REFINE-kb2_atmosphere_restrictions-distr0.5-tst1000.json',
    #                  '/Users/shyietliu/python/E2E/e2e_dialog/my_dataset/dataset-E2E-goal-oriented-test-v1.0/tst4/dialog-task2REFINE-kb2_atmosphere_restrictions-distr0.5-tst1000.answers.json',
    #                  '/Users/shyietliu/python/E2E/e2e_dialog/my_dataset/dataset-E2E-goal-oriented-test-v1.0/tst4/task2.json')
    # dp = data_provider.DataProvider(1)
    #
    # train = copy.deepcopy(dp.task1.train)
    # val = copy.deepcopy(dp.task1.val)
    # #
    # train.current_path()
    # val.current_path()

    # >> > a = np.array([1, 0, 3])
    # >> > b = np.zeros((3, 4))
    # >> > b[np.arange(3), a] = 1
    # >> > b
    # array([[0., 1., 0., 0.],
    #        [1., 0., 0., 0.],
    #        [0., 0., 0., 1.]])

    obj = MyClass()
    obj.inner_function()
    # inner_function()
    # index_set_to_zer
    # o = np.random.random(1)
    # print(index_set_to_zero)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     a = tf.convert_to_tensor(np.arange(16).reshape(2, 2, 2, 2))
    #     b = tf.convert_to_tensor(np.array([[[0,1],[0,0]],[[1,0],[1,0]]]))
    #     print(sess.run(tf.einsum('abcd,abc->abcd', a, b)))
    # print(np.pad([1,2,3], (0, 10 - len([1,2,3])), 'constant'))
    pass
