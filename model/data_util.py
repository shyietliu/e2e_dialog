import json
import numpy as np

class UtilsFn(object):
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def count_answer_category(file_path):
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

            if (answer + '\n') not in answer_categories:
                answer_categories.append(answer + '\n')

        for ans in answer_categories:
            print(ans)

        with open('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/task1_answer_category.txt', 'w') as f:
            for ele in answer_categories:
                f.write(ele)

    @staticmethod
    def split_data(full_train_data_file, train_dir, val_dir):
        """
        split original training data into train and validation set with proportion (9:1)
        And save new train and validation file at 'trian_dir' and 'val_dir' respectively
        :param full_train_data_file: original data file path
        :param train_dir: split data saved path
        :param val_dir: split data saved path
        :return: None
        """

        with open(full_train_data_file) as f:
            full_train_data = json.load(f)

        data_size = len(full_train_data)
        shuffled_index = np.random.permutation(data_size)

        train_data = []
        val_data = []
        for i in shuffled_index:
            if i < data_size*0.9:
                train_data.append(full_train_data[i])
            else:
                val_data.append(full_train_data[i])

        # print(len(train_data), len(val_data))

        with open(val_dir+'val_data.json', 'w') as f:
            json.dump(val_data, f)

        with open(train_dir+'train_data.json', 'w') as f:
            json.dump(train_data, f)


if __name__ == '__main__':
    util_function = UtilsFn()
    # util_function.split_data('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/task2/train/dialog-task2REFINE-kb1_atmosphere-distr0.5-trn10000.json',
    #                          '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/task2/train/',
    #                          '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/task2/val/')
    util_function.count_answer_category('/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/task1/train/train_data.json')