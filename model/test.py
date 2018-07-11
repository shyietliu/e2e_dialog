import json
import numpy as np
import tensorflow as tf
import os
import data_provider
import copy


def reform_test_data(test_data_path, test_answer_path, reformed_data_saved_path):
    """
    Create a new json file that is compatible with the training data form
    :param test_data_path: file path of test set data
    :param test_answer_path: file path of the answer of test set
    :param reformed_data_saved_path: directory that the new data will be saved
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
    #     a = np.round(np.zeros([1, 300]), decimals=5).tolist()
    #
    #     for ele in a:
    #         for v in ele:
    #             f.write(' '+str(v))
    #         f.write('\n')

    # dp = data_provider.DataProvider(1)
    #
    # train = copy.deepcopy(dp.task1.train)
    # val = copy.deepcopy(dp.task1.val)
    # #
    # train.current_path()
    # val.current_path()
    print(np.pad([1,2,3], (0, 10 - len([1,2,3])), 'constant'))
    pass
