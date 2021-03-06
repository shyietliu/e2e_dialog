from h_lstm import HierarchicalLSTM
from classifier import MultiLayerPerceptron
from lstm import LSTM
from conv_lstm import ConvolutionLSTM
from attn_net import AttnNet
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
# tf.app.flags.DEFINE_string('data_path', DATA_PATH, "training data dir")
tf.app.flags.DEFINE_integer('epoch', 20, "Training epochs")
tf.app.flags.DEFINE_float('lr', 1e-4, "Learning rate")
tf.app.flags.DEFINE_float('keep_prob', 0.8, "Learning rate")
tf.app.flags.DEFINE_boolean('train', True, 'Training status')
tf.app.flags.DEFINE_string('exp_name', 'name', 'experiment name')
tf.app.flags.DEFINE_boolean('save_model', False, 'save model or not')
tf.app.flags.DEFINE_integer('task_num', 1, 'task category (1 or 2)')
tf.app.flags.DEFINE_string('model', 'h_lstm', 'experiment name')

if __name__ == '__main__':

    if FLAGS.model == 'h_lstm':
        model = HierarchicalLSTM(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'attn_net':
        model = AttnNet(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    keep_prob=FLAGS.keep_prob,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'lstm':
        model = LSTM(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    save_model=FLAGS.save_model)




    # task = 1,2,3,4,5
    # test = 1, 2, 3, 4
"""
Basic hyper-parameters
--task_num 1 --model attn_net --lr 1e-4  --epoch 20 --exp_name h_lstm --keep_prob 0.8 --save_model False
"""
# utterance = re.sub('!', '', utterance)
#                             utterance = re.sub(',', ' ,', utterance)
#                             utterance = re.sub('\.', ' .', utterance)
#                             utterance = re.sub('\?', ' ?', utterance)
#                             utterance = re.sub('\'', ' \'', utterance)