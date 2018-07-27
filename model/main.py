from h_lstm import HierarchicalLSTM
from classifier import MultiLayerPerceptron
from lstm import LSTM
from attn_net_data_form_1 import AttnNet_1
from conv_lstm import ConvolutionLSTM
from mix_model import MixModel
from attn_net import AttnNet
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# DATA_PATH = '/afs/inf.ed.ac.uk/user/s17/s1700619/E2E_dialog/my_dataset'
# tf.app.flags.DEFINE_string('data_path', DATA_PATH, "training data dir")
tf.app.flags.DEFINE_integer('epoch', 20, "Training epochs")
tf.app.flags.DEFINE_float('lr', 1e-4, "Learning rate")
tf.app.flags.DEFINE_float('keep_prob', 0.8, "Learning rate")
tf.app.flags.DEFINE_string('exp_name', 'name', 'experiment name')
tf.app.flags.DEFINE_boolean('save_model', False, 'save model or not')
tf.app.flags.DEFINE_integer('task_num', 1, 'task category (1 or 2)')
tf.app.flags.DEFINE_integer('data_form', 1, 'data form (1 or 2)')
tf.app.flags.DEFINE_integer('mask_input', 0, '1 for enable input masking')
tf.app.flags.DEFINE_string('model', 'h_lstm', 'experiment name')

if __name__ == '__main__':

    if FLAGS.model == 'h_lstm':
        model = HierarchicalLSTM(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    save_model=FLAGS.save_model,
                    mask_input=FLAGS.mask_input
                    )
    elif FLAGS.model == 'attn_net':
        model = AttnNet(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    keep_prob=FLAGS.keep_prob,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'attn_net_data_form_1':
        model = AttnNet_1(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    keep_prob=FLAGS.keep_prob,
                    mask_input=FLAGS.mask_input,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'mix_model':
        model = MixModel(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    keep_prob=FLAGS.keep_prob,
                    mask_input=FLAGS.mask_input,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'lstm':
        model = LSTM(FLAGS.task_num)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    save_model=FLAGS.save_model)
    elif FLAGS.model == 'mlp':
        model = MultiLayerPerceptron(FLAGS.task_num, data_form=2)
        model.train(FLAGS.epoch,
                    FLAGS.exp_name,
                    FLAGS.lr,
                    save_model=FLAGS.save_model,
                    keep_prob=FLAGS.keep_prob
                    )


"""
Basic hyper-parameters
--task_num 1 --model h_lstm --lr 1e-4  --epoch 20 --exp_name h_lstm --keep_prob 0.8 --save_model False
gcloud compute scp instance-2:/home/shyietliu/e2e_dialog/exp_log/task1/h_lstm_without_masking/log/h_lstm_without_masking_log.txt ../exp_log/h_lstm_without_masking_log.txt
"""
# utterance = re.sub('!', '', utterance)
#                             utterance = re.sub(',', ' ,', utterance)
#                             utterance = re.sub('\.', ' .', utterance)
#                             utterance = re.sub('\?', ' ?', utterance)
#                             utterance = re.sub('\'', ' \'', utterance
# )