import json
import re
import os
import numpy as np
from data_util import UtilsFn


class DataProvider(object):
    def __init__(self,
                 data_form,
                 path='/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset',
                 vocab_path='/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset/vocab.txt'):
        """

        :param path: default : '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
        :param data_form: 1 denotes x_batch shape: [batch_size, num_utterance, sequence_max_len, vocab_size]
                          2 denotes x_batch shape: [batch_size, num_all_word_in_dialog, vocab_size]
        """
        self.path = path
        self.batch_count = 0

        self.data = None
        self.vocabulary = None
        self.ALREADY_LOAD_DATA = 0
        self.ALREADY_LOAD_VOCAB = 0
        self.sequence_max_len = 30
        self.max_num_utterance = 25  # maximal number of utterances in dialog
        self.max_num_words_in_dialog = 160
        self.shuffle_index = None
        self.data_form = data_form

        self.TASK_ALREADY_SPECIFY = 0
        self.CATEGORY_ALREADY_SPECIFY = 0

        self.load_vocab(vocab_path)
        np.random.seed(1000)

    @property
    def task1(self):
        if not self.TASK_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'task1')
            self.TASK_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('task\d', 'task1', self.path)
        return self

    @property
    def task2(self):
        if not self.TASK_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'task2')
            self.TASK_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('task\d', 'task2', self.path)
        return self

    @property
    def train(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'train')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|val', 'train', self.path)
        return self

    @property
    def val(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'val')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|train', 'val', self.path)
        return self

    @property
    def test1(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'tst1')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|train|val', 'tst1', self.path)
        return self

    @property
    def test2(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'tst2')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|train|val', 'tst2', self.path)
        return self

    @property
    def test3(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'tst3')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|train|val', 'tst3', self.path)
        return self

    @property
    def test4(self):
        if not self.CATEGORY_ALREADY_SPECIFY:
            self.path = os.path.join(self.path, 'tst4')
            self.CATEGORY_ALREADY_SPECIFY = 1
        else:
            self.path = re.sub('tst\d|train|val', 'tst4', self.path)
        return self

    def current_path(self):
        print(self.path)

    def shuffle_data(self, data_size):
        """
        Generate a shuffled index given data size
        :param data_size:
        :return: None
        """
        self.shuffle_index = np.random.permutation(data_size)

    def pre_process(self, vocab_path):
        """
        Clean data and Create a vocabulary, save them to $vocab_path$
        Only create vocab from task1 training set.
        """

        # read data file
        with open(self.path) as f:
            self.data = json.load(f)

        # create vocabulary
        for data_piece in self.data:
            dialog = data_piece['utterances']
            for sent in dialog:
                word_list = str(sent).split()
                for word in word_list:
                    word = re.sub('[,.]', '', word)
                    if word not in self.vocabulary:
                        self.vocabulary.append(word)

        # save vocabulary
        v_file_name = 'full_vocab.txt'

        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)

        with open(vocab_path+v_file_name, 'w') as f:
            for item in self.vocabulary:
                print(item, file=f)

    def create_vocab(self, vocab_path):
        """
        Create a vocabulary, save them to $vocab_path$
        :param vocab_path: exclude file name.
        :return:
        """

        data_tasks_path = [
            '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/train/dialog-task1API-kb1_atmosphere-distr0.5-trn10000.json',
            '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/train/dialog-task2REFINE-kb1_atmosphere-distr0.5-trn10000.json']
            # '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/train/dialog-task4INFOS-kb1_atmosphere-distr0.5-trn10000.json',
            # '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/dataset/train/dialog-task5FULL-kb1_atmosphere-distr0.5-trn10000.json']

        for i, data_path in enumerate(data_tasks_path):
            # read data file
            print('-'*40)
            print('start processing {0} task file'.format(i))
            print('-' * 40)
            with open(data_path) as f:
                data = json.load(f)

            # create vocabulary
            for index, data_piece in enumerate(data):

                if index % 50 == 0 and index != 0:
                    print('start processing {0}th dialog'.format(index))

                dialog = data_piece['utterances']
                for sent in dialog:
                    if sent[0:8] == 'api_call':
                        sent = 'api_call'
                    if sent[0:10] == 'here it is':
                        sent = 'here it is'
                    if sent[0:33] == 'what do you think of this option:':
                        sent = 'what do you think of this option'
                    if sent[0:14] == 'the option was':
                        sent = 'the option was'

                    word_list = str(sent).split()
                    for word in word_list:
                        word = re.sub('[,.]', '', word)
                        if word not in self.vocabulary:
                            self.vocabulary.append(word)

        # save vocabulary
        v_file_name = 'vocab.txt'

        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)

        with open(vocab_path + v_file_name, 'w') as f:
            for item in self.vocabulary:
                print(item, file=f)

    def replace_oov(self, oov_data_path):
        """
        Replace Out-Of-Vocabulary tokens with 'UNK' and save the replaced data in a new file
        :param oov_data_path: file path of the OOV file,
                              the file required to be a json file
                              and has the same format with training data

                              The oov data file must named in following form:
                              task#.json
                              where '#' is the number of task.
        :return: None
        """
        if not oov_data_path[-5:] == '.json' and oov_data_path[-10:-6] == 'task':
            raise Exception('The oov data file must named as task#.json')
        with open(oov_data_path, 'r') as f:
            oov_data = json.load(f)

            for oov_data_piece in oov_data:
                utterances = oov_data_piece['utterances']
                replaced_utterances = []
                replaced_answer = []
                answer = oov_data_piece['answer']['utterance']
                for utterance in utterances:
                    words = []
                    for word in utterance.split():
                        if word in self.vocabulary:
                            words.append(word)
                        else:
                            words.append('UNK')
                    replaced_utterances.append(" ".join(str(word) for word in words))
                oov_data_piece['utterances'] = replaced_utterances
                words = []
                for word in answer.split():
                    if word in self.vocabulary:
                        words.append(word)
                    else:
                        words.append('UNK')
                replaced_answer.append(" ".join(str(word) for word in words))
                oov_data_piece['answer']['utterance'] = replaced_answer[0]

        saved_path = oov_data_path[:-10]
        new_file_name = 'replaced_'+oov_data_path[-10:-5]+'.json'

        if not os.path.exists(saved_path):
            os.mkdir(saved_path)

        with open(os.path.join(saved_path, new_file_name), 'w') as f:
            json.dump(oov_data, f)

    def padding(self, seq):
        """
        Padding a sequence of indices
        :param seq: a batch of sequences of indices e.g. [[23, 12, 56, 891],
                                                          [1,  54 ]]
        :return: a padded sequence e.g. [[23, 12, 56, 891, 0, 0, 0, 0, 0, 0]
                                         [1,  54, 0,    0, 0, 0, 0, 0, 0, 0]]
        """
        padded_seq = []
        for row in seq:
            row = np.pad(row, (0, self.sequence_max_len-len(row)), 'constant')
            padded_seq.append(row)
        return np.array(padded_seq)

    def next_batch(self, batch_size,
                   data_type='one_hot',
                   label_type='one_hot'):
        """
        get batch data
        :returns x_batch: batch dialog history, stored in a 3D list: [batch number, utterance number, word number]
                 y_batch: batch correct candidate, stored in a 2D list: [batch number, word number]
                 if data_type = index, return word index in the vocab
                 if data_type = word,  return raw words.
        """
        # load vocabulary
        if not self.ALREADY_LOAD_VOCAB:
            if os.path.isfile('../dataset/vocab_task1_only.txt'):
                self.load_vocab('../dataset/vocab_task1_only.txt')
            else:
                raise Exception('No vocabulary file! Creating vocabulary by running $pre_process$ method first!')
            self.ALREADY_LOAD_VOCAB = 1

        # read data
        if not self.ALREADY_LOAD_DATA:
            json_file = os.path.join(self.path, os.listdir(self.path)[0])
            with open(json_file) as f:
                self.data = json.load(f)
                self.ALREADY_LOAD_DATA = 1

        data_size = len(self.data)  # the size of training data
        total_num_batch = int(data_size / batch_size)  # the number of batches

        # # shuffle data
        if self.shuffle_index is None:
            self.shuffle_data(data_size)

        # get batch data
        x_batch = []
        y_batch = []
        # for each data pair in a batch
        for index in range(self.batch_count*batch_size, self.batch_count*batch_size+batch_size):
            x = self.data[self.shuffle_index[index]]['utterances']  # list, each element is an utterance (string)
            y = self.data[self.shuffle_index[index]]['answer']['utterance']  # string, the correct candidate

            # --------------- y batch data ----------------- #
            # reform all possible api_call answers to the same
            if y[0:8] == 'api_call':
                y = 'api_call'
            if y[0:10] == 'here it is':
                y = 'here it is'
            if y[0:33] == 'what do you think of this option:':
                y = 'what do you think of this option'
            if y[0:14] == 'the option was':
                y = 'the option was'

            if label_type == 'word':
                y_batch.append(y)
            elif label_type == 'one_hot':
                # convert utterance into one-hot vector
                y = self.answer_category.index(y)
                y = self.one_hot([y],
                                 vocab_size=len(self.answer_category),
                                 pad=True,
                                 max_len=1)

                y_batch.append(y[0])

            # --------------- x batch data ----------------- #
            if self.data_form == 1:
                if data_type == 'word':
                    x_batch.append(x)
                    # y_batch.append(y)
                elif data_type == 'index' or data_type == 'one_hot':
                    dialog = []
                    # answer = None
                    if data_type == 'index':
                        for utterance in x:
                            utterance = re.sub('[,.?!]', '', utterance)
                            # words in each utterance, list of strings
                            dialog.append([self.word2idx(ele) for ele in utterance.split()])
                            # answer = [self.word2idx(ele) for ele in y.split()]
                    elif data_type == 'one_hot':
                        for i in range(self.max_num_utterance):
                            if i < len(x):
                                utterance = x[i]
                                utterance = re.sub('[,.?!]', '', utterance)
                                dialog.append(self.one_hot([self.word2idx(ele) for ele in utterance.split()],
                                                           vocab_size=self.vocab_size,
                                                           pad=True,
                                                           max_len=self.sequence_max_len))
                            else:
                                dialog.append(np.zeros([self.sequence_max_len, self.vocab_size]).tolist())
                        # answer = self.one_hot([self.word2idx(ele) for ele in y.split()])

                    x_batch.append(dialog)
                    # y_batch.append(answer)
                else:
                    raise Exception('data_type must be \'word\' or \'index\' or \'one_hot\'!')

            elif self.data_form == 2:

                if data_type == 'word':
                    x_batch.append(x)
                elif data_type == 'index':
                    for utterance in x:
                        utterance = re.sub('[,.?!]', '', utterance)
                        # words in each utterance, list of strings
                        for word in utterance.split():
                            x_batch.append(self.word2idx(word))

                elif data_type == 'one_hot':
                    dialog = []
                    for utterance in x:
                        utterance = re.sub('[,.?!]', '', utterance)
                        for word in utterance.split():
                            word = self.one_hot([self.word2idx(word)],
                                                vocab_size=self.vocab_size)
                            dialog.append(word[0])

                    # padding dialog to the number of all words in dialog
                    pad_num = self.max_num_words_in_dialog - len(dialog)
                    dialog = dialog + np.zeros([pad_num, self.vocab_size]).tolist()

                    x_batch.append(dialog)

                else:
                    raise Exception('data_type must be \'word\' or \'index\' or \'one_hot\'!')

        if self.batch_count == total_num_batch-1:
            self.batch_count = 0  # reset count
        else:
            self.batch_count = self.batch_count + 1

        # y_batch = [y[0] for y in y_batch]
        return x_batch, y_batch

    @property
    def answer_category(self):
        with open('/afs/inf.ed.ac.uk/'
                  'user/s17/s1700619/E2E_dialog/my_dataset/task1_answer_category.txt', 'r') as f:
            ans = f.readlines()
        return [line[:-1] for line in ans]

    @staticmethod
    def one_hot(list_of_index, vocab_size=None, pad=False, max_len=None):
        """
        Making one-hot encoding and padding (optional) for a sequence of indices
        :param list_of_index: a list of word index in vocabulary, e.g. [1, 3, 0]
        :param vocab_size:
        :param pad:
        :param max_len:
        :return: a list of one-hot form indices e.g. [[0, 1, 0, 0]  # 1
                                                      [0, 0, 0, 1]  # 3
                                                      [1, 0, 0, 0]  # 0
                                                      [0, 0, 0, 0]  # padding (optional)
                                                      [0, 0, 0, 0]] # padding (optional)
        """
        if vocab_size is None:
            raise Exception('\'vocab_size\' arg is required.')

        if pad:
            if max_len is None:
                raise Exception('If padding, \'max_len\' arg is required.')
            matrix = np.zeros([max_len, vocab_size])

        else:
            matrix = np.zeros([len(list_of_index), vocab_size])
        for i, word_index in enumerate(list_of_index):
            matrix[i][word_index] = 1

        return matrix.tolist()

    def load_vocab(self, vocab_path):
        """Load vocabulary"""
        with open(vocab_path, 'r') as f:
            self.vocabulary = f.read().splitlines()

    def word2idx(self, word):
        """Convert word to index"""
        return self.vocabulary.index(word)

    def idx2word(self, index):
        """Convert index to word"""
        return self.vocabulary[index]

    @property
    def vocab_size(self):
        """Return the size of vocabulary"""
        return len(self.vocabulary)


if __name__ == '__main__':
    DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'

    data_provider = DataProvider(path=DATA_PATH, data_form=2)

    x, y = data_provider.task1.train.next_batch(100, data_type='one_hot', label_type='one_hot')
    data_provider.task2.train.current_path()
    data_provider.task1.test1.current_path()
    xx, yy = data_provider.task2.val.next_batch(10)

    # a = re.sub('a|bc', '!!', '0000000a0bc00')
    # print(a)

    pass
