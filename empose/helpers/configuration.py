"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    A singleton for some common constants.
    """

    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.DTYPE = torch.float32
            self.DATA_DIR = os.environ['EM_DATA_SYNTH']
            self.EXPERIMENT_DIR = os.environ['EM_EXPERIMENTS']
            self.SMPL_MODELS_DIR = os.environ['SMPL_MODELS']
            self.DATA_DIR_TEST = os.environ['EM_DATA_REAL']
            self.FPS = 60.0

            # Virtual tracker vertex IDs.
            self.VERTEX_IDS = [3027, 3748,
                               5430, 5178, 5006, 4447, 4559,
                               1961, 1391, 1535, 959, 1072]

            # Virtual tracker names and configuration.
            self.T_ROOT = 'root_0'
            self.T_HEAD = 'head_1'
            self.T_BACK = 'back_8'
            self.T_RLA = 'r_wrist_3'
            self.T_RUA = 'r_arm_5'
            self.T_RSH = 'r_shoulder_7'
            self.T_RUL = 'r_leg_9'
            self.T_RLL = 'r_low_leg_11'
            self.T_LLA = 'l_wrist_2'
            self.T_LUA = 'l_arm_4'
            self.T_LSH = 'l_shoulder_6'
            self.T_LUL = 'l_leg_10'
            self.T_LLL = 'l_low_leg_12'

            self.T_ORDER = [self.T_ROOT, self.T_BACK, self.T_HEAD,
                            self.T_RLA, self.T_RUA, self.T_RSH, self.T_RUL, self.T_RLL,
                            self.T_LLA, self.T_LUA, self.T_LSH, self.T_LUL, self.T_LLL]
            self.T_TO_IDX = {k: i for i, k in enumerate(self.T_ORDER)}
            self.T_TO_IDX_WO_ROOT = {k: i - 1 for i, k in enumerate(self.T_ORDER)}
            self.N_TRACKERS_WO_ROOT = len(self.T_ORDER) - 1  # The root is not a tracker

            self.T_SKELETON_W_ROOT = [[self.T_TO_IDX[self.T_ROOT], self.T_TO_IDX[self.T_BACK]],
                                      [self.T_TO_IDX[self.T_ROOT], self.T_TO_IDX[self.T_RUL]],
                                      [self.T_TO_IDX[self.T_ROOT], self.T_TO_IDX[self.T_LUL]],
                                      [self.T_TO_IDX[self.T_BACK], self.T_TO_IDX[self.T_HEAD]],
                                      [self.T_TO_IDX[self.T_BACK], self.T_TO_IDX[self.T_RSH]],
                                      [self.T_TO_IDX[self.T_BACK], self.T_TO_IDX[self.T_LSH]],
                                      [self.T_TO_IDX[self.T_RSH], self.T_TO_IDX[self.T_RUA]],
                                      [self.T_TO_IDX[self.T_RUA], self.T_TO_IDX[self.T_RLA]],
                                      [self.T_TO_IDX[self.T_LSH], self.T_TO_IDX[self.T_LUA]],
                                      [self.T_TO_IDX[self.T_LUA], self.T_TO_IDX[self.T_LLA]],
                                      [self.T_TO_IDX[self.T_RUL], self.T_TO_IDX[self.T_RLL]],
                                      [self.T_TO_IDX[self.T_LUL], self.T_TO_IDX[self.T_LLL]]]

            # Real tracker (sensor) names and configuration.
            self.S_HEAD = 'ID113.Set7.Num1'
            self.S_BACK = 'ID120.Set7.Num8'
            self.S_RLA = 'ID115.Set7.Num3'
            self.S_RUA = 'ID117.Set7.Num5'
            self.S_RSH = 'ID119.Set7.Num7'
            self.S_RUL = 'ID121.Set7.Num9'
            self.S_RLL = 'ID123.Set7.Num11'
            self.S_LLA = 'ID114.Set7.Num2'
            self.S_LUA = 'ID116.Set7.Num4'
            self.S_LSH = 'ID118.Set7.Num6'
            self.S_LUL = 'ID122.Set7.Num10'
            self.S_LLL = 'ID124.Set7.Num12'

            # This is the sensor order how the neural network expects it.
            self.S_ORDER = [self.S_BACK, self.S_HEAD,
                            self.S_RLA, self.S_RUA, self.S_RSH, self.S_RUL, self.S_RLL,
                            self.S_LLA, self.S_LUA, self.S_LSH, self.S_LUL, self.S_LLL]
            self.S_CONFIG_6 = [0, 1, 2, 6, 7, 11]
            self.S_TO_IDX_WO_ROOT = {k: i for i, k in enumerate(self.S_ORDER)}
            self.S_SKELETON_WO_ROOT = [[self.S_TO_IDX_WO_ROOT[self.S_BACK], self.S_TO_IDX_WO_ROOT[self.S_HEAD]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_BACK], self.S_TO_IDX_WO_ROOT[self.S_RSH]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_BACK], self.S_TO_IDX_WO_ROOT[self.S_LSH]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_BACK], self.S_TO_IDX_WO_ROOT[self.S_LUL]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_BACK], self.S_TO_IDX_WO_ROOT[self.S_RUL]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_RSH], self.S_TO_IDX_WO_ROOT[self.S_RUA]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_RUA], self.S_TO_IDX_WO_ROOT[self.S_RLA]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_LSH], self.S_TO_IDX_WO_ROOT[self.S_LUA]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_LUA], self.S_TO_IDX_WO_ROOT[self.S_LLA]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_RUL], self.S_TO_IDX_WO_ROOT[self.S_RLL]],
                                       [self.S_TO_IDX_WO_ROOT[self.S_LUL], self.S_TO_IDX_WO_ROOT[self.S_LLL]]]

            # SMPL constants
            self.N_JOINTS = 21  # not counting root
            self.MAX_INDEX_ROOT_AND_BODY = 66  # including root and assuming angle-axis
            self.N_JOINTS_HAND = 15
            self.N_SHAPE_PARAMS = 10

            # VISUALIZATION
            self.COLOR_PRED = (184 / 255, 130 / 255, 0 / 255, 1.0)
            self.COLOR_GT = (15 / 255, 127 / 255, 174 / 255, 1.0)
            self.COLOR_PRED_12 = (3 / 255, 180 / 255, 138 / 255, 1.0)
            self.COLOR_BIRNN = (116 / 255, 109 / 255, 144 / 255, 1.0)

            self.SMPL_JOINTS = ['root', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle',
                                'spine3', 'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder',
                                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
            self.SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()


class Configuration(object):
    """Configuration options for training/eval/test runs."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # General.
        parser.add_argument('--experiment_id', default=None, help='Use this experiment ID or create a new one.')
        parser.add_argument('--seed', type=int, default=None, help='Random generator seed.')
        parser.add_argument('--data_workers', type=int, default=4, help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=25, help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=700, help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        parser.add_argument('--test', action='store_true', help='Will tag this run as a test run.')

        # Model configurations.
        parser.add_argument('--m_type', default='rnn', choices=['rnn', 'resnet', 'ief', 'lgd'], help='The type of model.')
        parser.add_argument('--m_estimate_shape', action='store_true', help='The model estimates the body shape.')
        parser.add_argument('--m_shape_hidden_size', default=256, help='Size of the network estimating the shape.')  # Only used in RNN/ResNet.
        parser.add_argument('--m_fk_loss', type=float, default=0.0, help='Add an FK loss, requires shape estimate.')
        parser.add_argument('--m_dropout', type=float, default=0.0, help='Dropout applied on inputs.')
        parser.add_argument('--m_hidden_size', type=int, default=1024, help='Number of hidden units.')
        parser.add_argument('--m_num_layers', type=int, default=2, help='Number of layers.')
        parser.add_argument('--m_learn_init_state', action='store_true', help='Learn initial hidden state.')
        parser.add_argument('--m_bidirectional', action='store_true', help='Bidirectional RNN.')

        # IEF model specific.
        parser.add_argument('--m_num_iterations', type=int, default=4, help='Number of iterations for IEF.')
        parser.add_argument('--m_dropout_hidden', type=float, default=0.0, help='Dropout applied inside layers.')
        parser.add_argument('--m_step_size', type=float, default=0.1, help='Step size for IEF update.')
        parser.add_argument('--m_reprojection_loss_weight', type=float, default=0.01, help='Reprojection loss weight.')
        parser.add_argument('--m_shape_loss_weight', type=float, default=1.0, help='Loss for the shape weight.')
        parser.add_argument('--m_pose_loss_weight', type=float, default=1.0, help='Loss for the shape weight.')
        parser.add_argument('--m_average_shape', action='store_true', help='Average the shape per sequence.')
        parser.add_argument('--m_use_gradient', action='store_true', help='Feed dL/dtheta to the network.')
        parser.add_argument('--m_skip_connections', action='store_true', help='Skip connections in the MLP.')
        parser.add_argument('--m_no_batch_norm', action='store_true', help="Don't use batch norm.")
        parser.add_argument('--m_rnn_init', action='store_true', help="Initial estimate is provided by an RNN.")
        parser.add_argument('--m_rnn_denoiser', action='store_true', help="Use an RNN to de-noise the markers.")
        parser.add_argument('--m_rnn_bidirectional', action='store_true', help="BiRNN or not.")
        parser.add_argument('--m_rnn_hidden_size', type=int, default=512, help="Hidden size for the init RNN.")
        parser.add_argument('--m_rnn_num_layers', type=int, default=2, help="Number of layers for the init RNN.")

        # Input data.
        parser.add_argument('--use_marker_pos', action='store_true', help='Feed marker positions.')
        parser.add_argument('--use_marker_ori', action='store_true', help='Feed marker orientations.')
        parser.add_argument('--use_marker_nor', action='store_true', help='Feed marker normal instead of orientation.')
        parser.add_argument('--use_real_offsets', action='store_true', help='Sampling is informed by real offset distribution.')
        parser.add_argument('--offset_noise_level', type=int, default=0, help='How much noise to add to real offsets.')
        parser.add_argument('--n_markers', type=int, default=12, help='Subselect a number of markers for the input.')

        # Data augmentation.
        parser.add_argument('--noise_num_markers', type=int, default=1, help='How many markers are affected by the noise.')
        parser.add_argument('--spherical_noise_strength', type=float, default=0.0, help='Magnitude of noise in %.')
        parser.add_argument('--spherical_noise_length', type=float, default=0.0, help='Temporal length of noise in %.')
        parser.add_argument('--suppression_noise_length', type=float, default=0.0, help='Marker suppression length.')
        parser.add_argument('--suppression_noise_value', type=float, default=0.0, help='Marker suppression value.')

        # Learning configurations.
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for valid/test set.')
        parser.add_argument('--eval_window_size', type=int, default=None, help='Window size for evaluation on test set.')
        parser.add_argument('--window_size', type=int, default=120, help='Number of frames to extract per sequence.')
        parser.add_argument('--load', action='store_true', help='Whether to load the model with the given ID.')

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load a configuration object from JSON."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump this configuration object to JSON."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)
