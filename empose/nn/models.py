"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import torch
import torch.nn as nn

from empose.data.data import ABatch
from empose.data.virtual_sensors import VirtualMarkerHelper
from empose.helpers.configuration import CONSTANTS as C
from empose.nn.layers import FeedForwardResidualBlock
from empose.nn.layers import MLP
from empose.nn.layers import RNNLayer
from empose.nn.loss import mask_from_seq_lengths
from empose.nn.loss import padded_loss
from empose.nn.loss import reconstruction_loss
from empose.nn.loss import normal_mse


def create_model(config, *args):
    """Create the model according to the configuration object."""
    m_type = config.m_type
    if m_type == 'rnn':
        return SimpleRNN(config, *args)
    elif m_type == 'resnet':
        return FeedForwardResNet(config, *args)
    elif m_type == 'ief' or m_type == 'lgd':
        return IterativeErrorFeedback(config, *args)
    else:
        raise ValueError("Model type '{}' unknown.".format(m_type))


class BaseModel(nn.Module):
    """
    A base class to handle some tasks common to all models.
    """

    def __init__(self, config, smpl_model=None):
        super(BaseModel, self).__init__()
        self.n_markers = config.n_markers if hasattr(config, 'n_markers') and config.n_markers > -1 else C.N_TRACKERS_WO_ROOT
        self.config = config
        self.n_frames = config.window_size
        self.smpl = smpl_model

        self.estimate_shape = self.config.m_estimate_shape
        self.shape_avg = config.m_average_shape

        self.fk_loss_weight = config.m_fk_loss
        self.do_fk = self.fk_loss_weight > 0.0
        if self.do_fk:
            assert self.smpl is not None
            assert self.estimate_shape or isinstance(self, IterativeErrorFeedback)

        self.shape_weight = config.m_shape_loss_weight if hasattr(config, 'm_shape_loss_weight') else 1.0
        self.pose_weight = config.m_pose_loss_weight if hasattr(config, 'm_pose_loss_weight') else 1.0

        self.set_input_output_size()
        self.create_model()

    # noinspection PyAttributeOutsideInit
    def set_input_output_size(self):
        input_size = 0
        if self.config.use_marker_pos:
            input_size += self.n_markers * 3
        if self.config.use_marker_ori:
            input_size += self.n_markers * 9
            assert not self.config.use_marker_nor
        if self.config.use_marker_nor:
            input_size += self.n_markers * 3
            assert not self.config.use_marker_ori

        # Add input and output size to config.
        setattr(self.config, 'input_size', input_size)
        setattr(self.config, 'output_size', (C.N_JOINTS + 1) * 3)  # angle-axis including root ori

        self.input_size = self.config.input_size
        self.output_size = self.config.output_size

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        """A summary string of this model."""
        base_name = ''
        if self.estimate_shape is not None:
            base_name += '-shape{}{}'.format(self.config.m_shape_hidden_size,
                                             '-avg' if self.shape_avg else '')
        if self.do_fk:
            base_name += '-fk{}'.format(self.fk_loss_weight)
        base_name += '-n{}'.format(self.n_markers)
        base_name += '-lr{}'.format(self.config.lr)
        return base_name

    def forward(self, batch: ABatch, window_size=None, is_new_sequence=True):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: ABatch, model_out, writer=None, global_step=None):
        """The backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def prepare_inputs(self, batch_inputs):
        """Expects offsets in the input and reverts them."""
        n, f = batch_inputs['marker_pos'].shape[0], batch_inputs['marker_pos'].shape[1]
        m_pos = batch_inputs['marker_pos'].reshape((n, f, -1, 3))
        m_ori = batch_inputs['marker_oris'].reshape((n, f, -1, 3, 3))

        assert self.n_markers in [6, 12]
        if self.n_markers == 6:
            m_pos = m_pos[:, :, C.S_CONFIG_6]
            m_ori = m_ori[:, :, C.S_CONFIG_6]

        model_in = []
        if self.config.use_marker_pos:
            model_in.append(m_pos.reshape((n, f, -1)))
        if self.config.use_marker_ori:
            model_in.append(m_ori.reshape((n, f, -1)))
        if self.config.use_marker_nor:
            raise ValueError('Normals currently not supported.')

        return torch.cat(model_in, dim=-1)

    def log_loss_vals(self, loss_vals, writer, global_step):
        """Log loss values in tensorboard using `writer`."""
        mode_prefix = 'train' if self.training else 'valid'
        for k in loss_vals:
            prefix = '{}/{}'.format(k, mode_prefix)
            writer.add_scalar(prefix, loss_vals[k], global_step)

    def maybe_do_fk(self, pose_hat, shape_hat):
        """Forward pass through SMPL if configured. `pose_hat` should include the root pose."""
        joints_hat = None
        if self.do_fk:
            assert len(pose_hat.shape) == 3
            n, f = pose_hat.shape[0], pose_hat.shape[1]
            _, joints_hat = self.smpl(pose_hat[:, :, 3:].reshape(n * f, -1),
                                      shape_hat.reshape(n * f, -1),
                                      poses_root=pose_hat[:, :, :3].reshape(n * f, -1))
            joints_hat = joints_hat[:, :(C.N_JOINTS + 1)].reshape(n, f, -1)
        return joints_hat

    def window_generator(self, batch: ABatch, window_size):
        """Subdivide a batch into temporal windows of length `window_size`."""
        if window_size is not None:
            seq_len = batch.seq_length
            n_windows = seq_len // window_size + int(seq_len % window_size > 0)

            for i in range(n_windows):
                sf = i * window_size
                ef = min((i + 1) * window_size, seq_len)
                seq_lengths_new = torch.tensor([ef - sf]).to(dtype=batch.seq_lengths.dtype,
                                                             device=batch.seq_lengths.device)
                batch_inputs = batch.get_inputs(sf=sf, ef=ef)
                batch_inputs['seq_lengths'] = seq_lengths_new
                yield batch_inputs
        else:
            batch_inputs = batch.get_inputs()
            batch_inputs['seq_lengths'] = batch.seq_lengths
            yield batch_inputs


class FeedForwardResNet(BaseModel):
    """A frame-wise feed-forward model with residual connections, similar to Holden's "Robust Denoising". """

    def __init__(self, config, smpl_model):
        super(FeedForwardResNet, self).__init__(config, smpl_model)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        self.hidden_size = self.config.m_hidden_size
        self.num_layers = self.config.m_num_layers

        self.from_input = nn.Linear(self.input_size, self.hidden_size)
        self.blocks = nn.Sequential(
            *[FeedForwardResidualBlock(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])

        self.to_pose = nn.Linear(self.hidden_size, self.output_size)

        if self.estimate_shape:
            self.to_shape = MLP(input_size=self.hidden_size,
                                output_size=C.N_SHAPE_PARAMS, hidden_size=self.config.m_shape_hidden_size,
                                num_layers=2, dropout_p=self.config.m_dropout_hidden,
                                skip_connection=self.config.m_skip_connections, use_batch_norm=False)
        else:
            self.to_shape = None

        self.shape_loss = nn.L1Loss(reduction='none')

    def model_name(self):
        base_name = "ResNet-{}x{}".format(self.num_layers, self.hidden_size)
        base_name += super(FeedForwardResNet, self).model_name()
        return base_name

    def forward(self, batch: ABatch, window_size=None, is_new_sequence=True):
        inputs_ = self.prepare_inputs(batch.get_inputs())

        # Estimate pose.
        x = self.from_input(inputs_)
        x = self.blocks(x)
        pose_hat = self.to_pose(x)

        # Estimate shape if configured.
        if self.to_shape is not None:
            shape_hat = self.to_shape(x)
            if self.shape_avg:
                s = torch.mean(shape_hat, dim=1, keepdim=True)
                shape_hat = s.repeat((1, shape_hat.shape[1], 1))
        else:
            shape_hat = None

        joints_hat = self.maybe_do_fk(pose_hat, shape_hat)

        return {'pose_hat': pose_hat[:, :, 3:],
                'root_ori_hat': pose_hat[:, :, :3],
                'shape_hat': shape_hat,
                'joints_hat': joints_hat}

    def backward(self, batch: ABatch, model_out, writer=None, global_step=None):
        """The backward pass."""
        pose_hat, root_ori_hat, shape_hat = model_out['pose_hat'], model_out['root_ori_hat'], model_out['shape_hat']

        n, f = batch.batch_size, batch.seq_length
        # This is just MSE summed over joints, mean over length and batch size.
        pose_loss = normal_mse(batch.poses_body.reshape(n, f, -1, 3),
                               pose_hat.reshape(n, f, -1, 3),
                               batch.seq_lengths, batch.marker_masks)
        root_pose_loss = normal_mse(batch.poses_root.reshape(n, f, -1, 3),
                                    root_ori_hat.reshape(n, f, -1, 3),
                                    batch.seq_lengths, batch.marker_masks)

        if self.estimate_shape:
            shape_loss = padded_loss(batch.shapes.unsqueeze(1).repeat((1, shape_hat.shape[1], 1)),
                                     shape_hat, self.shape_loss, batch.seq_lengths)
        else:
            shape_loss = torch.zeros(1).to(device=C.DEVICE)

        if self.do_fk:
            joints_gt = batch.joints_gt.reshape(batch.batch_size, batch.seq_length, -1, 3)
            joints_hat = model_out['joints_hat'].reshape(batch.batch_size, batch.seq_length, -1, 3)
            fk_loss = reconstruction_loss(joints_gt, joints_hat, batch.seq_lengths, batch.marker_masks)
        else:
            fk_loss = torch.zeros(1).to(device=C.DEVICE)

        total_loss = pose_loss + root_pose_loss + shape_loss + self.fk_loss_weight * fk_loss

        loss_vals = {'pose': pose_loss.cpu().item(),
                     'root_pose': root_pose_loss.cpu().item(),
                     'shape': shape_loss.cpu().item(),
                     'fk': fk_loss.cpu().item(),
                     'total_loss': total_loss.cpu().item()}

        if writer is not None:
            self.log_loss_vals(loss_vals, writer, global_step)

        if self.training:
            total_loss.backward()

        return total_loss, loss_vals


class SimpleRNN(BaseModel):
    """A uni- or bidirectional RNN."""

    def __init__(self, config, smpl_layer=None):
        super(SimpleRNN, self).__init__(config, smpl_layer)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        hidden_size = self.config.m_hidden_size
        num_directions = 2 if self.config.m_bidirectional else 1

        self.rnn = RNNLayer(self.input_size, hidden_size, self.config.m_num_layers,
                            bidirectional=self.config.m_bidirectional, dropout=self.config.m_dropout,
                            learn_init_state=self.config.m_learn_init_state)
        self.to_pose = nn.Linear(hidden_size * num_directions, self.output_size)

        if self.estimate_shape:
            self.to_shape = MLP(input_size=hidden_size * num_directions,
                                output_size=C.N_SHAPE_PARAMS, hidden_size=self.config.m_shape_hidden_size,
                                num_layers=2, dropout_p=self.config.m_dropout_hidden,
                                skip_connection=self.config.m_skip_connections, use_batch_norm=False)
        else:
            self.to_shape = None

        self.shape_loss = nn.L1Loss(reduction='none')

    def model_name(self):
        """A summary string of this model."""
        base_name = "RNN-{}".format('-'.join([str(self.config.m_hidden_size)] * self.config.m_num_layers))
        if self.config.m_bidirectional:
            base_name = "Bi" + base_name
        base_name += super(SimpleRNN, self).model_name()
        return base_name

    def forward(self, batch: ABatch, window_size=None, is_new_sequence=True):
        if is_new_sequence:
            self.rnn.final_state = None
        self.rnn.init_state = self.rnn.final_state

        inputs_ = self.prepare_inputs(batch.get_inputs())
        lstm_out = self.rnn(inputs_, batch.seq_lengths)
        pose_hat = self.to_pose(lstm_out)  # (N, F, self.output_size)

        # Estimate shape if configured.
        shape_hat = None
        if self.estimate_shape:
            shape_hat = self.to_shape(lstm_out)  # (N, F, N_BETAS)
            if self.shape_avg:
                s = torch.mean(shape_hat, dim=1, keepdim=True)
                shape_hat = s.repeat((1, shape_hat.shape[1], 1))

        joints_hat = self.maybe_do_fk(pose_hat, shape_hat)

        return {'pose_hat': pose_hat[:, :, 3:],
                'root_ori_hat': pose_hat[:, :, :3],
                'shape_hat': shape_hat,
                'joints_hat': joints_hat}

    def backward(self, batch: ABatch, model_out, writer=None, global_step=None):
        """The backward pass."""
        pose_hat, root_ori_hat, shape_hat = model_out['pose_hat'], model_out['root_ori_hat'], model_out['shape_hat']

        n, f = batch.batch_size, batch.seq_length
        mask = mask_from_seq_lengths(batch.seq_lengths).to(dtype=pose_hat.dtype)

        p_body = batch.poses_body.reshape(n, f, -1, 3)
        p_root = batch.poses_root.reshape(n, f, -1, 3)

        # Joint-wise squared L2 norm for rotations.
        pose_loss = normal_mse(p_body, pose_hat.reshape(n, f, -1, 3),
                               batch.seq_lengths, batch.marker_masks)
        root_pose_loss = normal_mse(p_root, root_ori_hat.reshape(n, f, -1, 3),
                                    batch.seq_lengths, batch.marker_masks)

        if self.estimate_shape:
            shape_loss = padded_loss(batch.shapes.unsqueeze(1).repeat((1, shape_hat.shape[1], 1)),
                                     shape_hat, self.shape_loss, batch.seq_lengths)
        else:
            shape_loss = torch.zeros(1).to(device=C.DEVICE)

        if self.do_fk:
            joints_gt = batch.joints_gt.reshape(batch.batch_size, batch.seq_length, -1, 3)
            joints_hat = model_out['joints_hat'].reshape(batch.batch_size, batch.seq_length, -1, 3)
            fk_loss = reconstruction_loss(joints_gt, joints_hat, batch.seq_lengths, batch.marker_masks)
        else:
            fk_loss = torch.zeros(1).to(device=C.DEVICE)

        total_loss = pose_loss + root_pose_loss + shape_loss + self.fk_loss_weight * fk_loss

        loss_vals = {'pose': pose_loss.cpu().item(),
                     'root_pose': root_pose_loss.cpu().item(),
                     'shape': shape_loss.cpu().item(),
                     'fk': fk_loss.cpu().item(),
                     'total_loss': total_loss.cpu().item()}

        if writer is not None:
            self.log_loss_vals(loss_vals, writer, global_step)

        if self.training:
            total_loss.backward()

        return total_loss, loss_vals


class IterativeErrorFeedback(BaseModel):
    """The LGD RNN model."""

    def __init__(self, config, smpl_model):
        self.N = config.m_num_iterations
        self.step_size = config.m_step_size
        self.shape_avg = config.m_average_shape
        self.r_weight = config.m_reprojection_loss_weight
        self.use_gradient = config.m_use_gradient
        self.skip_connections = config.m_skip_connections
        self.rnn_init = config.m_rnn_init
        super(IterativeErrorFeedback, self).__init__(config, smpl_model)

        self.virtual_marker_helper = VirtualMarkerHelper(self.smpl)
        self.vertex_ids = C.VERTEX_IDS

        assert self.n_markers in [6, 12]
        self.marker_idxs = list(range(12)) if self.n_markers == 12 else C.S_CONFIG_6

        self.markers_gt = None
        self.markers_ori_gt = None
        self.markers_hat_history = None
        self.markers_ori_hat_history = None
        self.pose_hat_history = None
        self.shape_hat_history = None
        self.joints_hat_history = None

    # noinspection PyAttributeOutsideInit
    def set_input_output_size(self):
        self.pos_d_start, self.pos_d_end = 0, 0
        self.ori_d_start, self.ori_d_end = 0, 0
        input_size = 0
        if self.config.use_marker_pos:
            input_size += self.n_markers * 3
            self.pos_d_end = self.pos_d_start + self.n_markers * 3
            self.ori_d_start = self.pos_d_end
        if self.config.use_marker_ori:
            input_size += self.n_markers * 9
            self.ori_d_end = self.ori_d_start + self.n_markers * 9
            assert not self.config.use_marker_nor

        self.input_size = input_size
        self.pose_size = (C.N_JOINTS + 1) * 3  # angle-axis including root
        self.shape_size = C.N_SHAPE_PARAMS
        self.input_iter_size = input_size + self.pose_size + self.shape_size
        if self.use_gradient:
            self.input_iter_size += self.pose_size + self.shape_size

        # Add input and output size to config.
        setattr(self.config, 'input_size', self.input_size)
        setattr(self.config, 'pose_size', self.pose_size)
        setattr(self.config, 'shape_size', self.shape_size)
        setattr(self.config, 'input_iter_size', self.input_iter_size)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        if self.rnn_init:
            # The initial pose and shape are predicted by an RNN.
            self.rnn = RNNLayer(self.input_size, self.config.m_rnn_hidden_size, self.config.m_rnn_num_layers,
                                dropout=self.config.m_dropout, bidirectional=self.config.m_rnn_bidirectional)
            self.pose_net_init = nn.Linear(self.config.m_rnn_hidden_size, self.pose_size)
            self.shape_net_init = nn.Linear(self.config.m_rnn_hidden_size, self.shape_size)

        else:
            # The `init` networks produce a first estimate of pose and shape given only the inputs.
            # The `iter` networks take as input the previous pose/shape estimate, plus the inputs (plus ...)
            # One MLP to iteratively predict pose.
            self.pose_net_init = MLP(self.input_size, self.pose_size,
                                     self.config.m_hidden_size, self.config.m_num_layers,
                                     self.config.m_dropout_hidden, self.skip_connections,
                                     not self.config.m_no_batch_norm)

            # One MLP to iteratively predict shape.
            self.shape_net_init = MLP(self.input_size, self.shape_size,
                                      self.config.m_hidden_size, self.config.m_num_layers,
                                      self.config.m_dropout_hidden, self.skip_connections,
                                      not self.config.m_no_batch_norm)

        self.pose_net_iter = MLP(self.input_iter_size, self.pose_size,
                                 self.config.m_hidden_size, self.config.m_num_layers,
                                 self.config.m_dropout_hidden, self.skip_connections,
                                 not self.config.m_no_batch_norm)
        self.shape_net_iter = MLP(self.input_iter_size, self.shape_size,
                                  self.config.m_hidden_size, self.config.m_num_layers,
                                  self.config.m_dropout_hidden, self.skip_connections,
                                  not self.config.m_no_batch_norm)

        # Loss on the SMPL parameters.
        self.smpl_loss = nn.L1Loss(reduction='none')

    def model_name(self):
        """A summary string of this model."""
        name = "IEF-{}x{}-N{}".format(self.config.m_num_layers, self.config.m_hidden_size, self.config.m_num_iterations)
        if self.rnn_init:
            name += '-{}RNN-{}x{}'.format('Bi' if self.config.m_rnn_bidirectional else '',
                                          self.config.m_rnn_num_layers, self.config.m_rnn_hidden_size)
        name += '-r{}-ws{}-lr{}'.format(self.r_weight, self.config.window_size, self.config.lr)
        name += '-grad' if self.use_gradient else ''
        name += '-skip' if self.skip_connections else ''
        name += '-n{}'.format(self.n_markers)
        return name

    def get_estimated_real_markers(self, poses, shapes, offset_r, offset_t, vertex_ids):
        """Extract markers given SMPL pose and shape parameters and a list of vertex IDs."""
        # Pose and shape are expected to be in format (-1, dof).
        vertices, joints = self.smpl(poses_body=poses[:, 3:], betas=shapes, poses_root=poses[:, :3])
        marker_pos_synth, marker_ori_synth, _ = self.virtual_marker_helper.get_virtual_pos_and_rot(vertices, vertex_ids)

        # Apply the given offsets.
        marker_ori_corr = torch.matmul(marker_ori_synth, offset_r)
        marker_pos_corr = marker_pos_synth + torch.matmul(marker_ori_synth, offset_t.unsqueeze(-1)).squeeze(-1)

        joints_hat = joints[:, :(C.N_JOINTS + 1)]

        return marker_pos_corr, marker_ori_corr, joints_hat

    def forward(self, batch: ABatch, window_size=None, is_new_sequence=True):
        # We need to accumulate gradients for the reconstruction error.
        torch.set_grad_enabled(True)

        if self.rnn_init:
            if is_new_sequence:
                self.rnn.final_state = None
            self.rnn.init_state = self.rnn.final_state

        pose_hat_history = []
        shape_hat_history = []
        joints_hat_history = []
        markers_hat_history = []
        markers_ori_hat_history = []
        all_model_out = {'pose_hat': [], 'root_ori_hat': [], 'shape_hat': [], 'joints_hat': []}

        for batch_inputs in self.window_generator(batch, window_size=window_size):
            inputs_ = self.prepare_inputs(batch_inputs)
            dof = inputs_.shape[-1]
            batch_size, seq_length = inputs_.shape[0], inputs_.shape[1]

            offset_r = batch_inputs['offset_r']  # (N, M, 3, 3)
            offset_t = batch_inputs['offset_t']  # (N, M, 3)
            offset_r_flat = offset_r.unsqueeze(1).repeat(1, seq_length, 1, 1, 1).reshape(batch_size*seq_length, -1, 3, 3)
            offset_t_flat = offset_t.unsqueeze(1).repeat(1, seq_length, 1, 1).reshape(batch_size*seq_length, -1, 3)

            if self.rnn_init:
                self.rnn.init_state = self.rnn.final_state
                lstm_out = self.rnn(inputs_, batch_inputs['seq_lengths'])
                pose_hat = self.pose_net_init(lstm_out).reshape((batch_size * seq_length, -1))
                shape_hat = self.shape_net_init(lstm_out).reshape((batch_size * seq_length, -1))

                # Flatten everything.
                inputs_flat = inputs_.reshape((-1, dof))

            else:
                # Flatten everything.
                inputs_flat = inputs_.reshape((-1, dof))

                # Get initial estimate.
                pose_hat = self.pose_net_init(inputs_flat)
                shape_hat = self.shape_net_init(inputs_flat)

            # We only want one shape per sequence, so for now average the results and pad it again.
            def _to_single_shape(shapes):
                s = shapes.reshape(batch_size, seq_length, -1)
                s = torch.mean(s, dim=1, keepdim=True)
                return s.repeat((1, seq_length, 1)).reshape(seq_length * batch_size, -1)

            if self.shape_avg:
                shape_hat = _to_single_shape(shape_hat)

            marker_pos_hat, marker_ori_hat, joints_hat = self.get_estimated_real_markers(
                pose_hat, shape_hat, offset_r_flat, offset_t_flat, self.vertex_ids)

            # Keep track of history.
            pose_hat_history.append([pose_hat])
            shape_hat_history.append([shape_hat])
            joints_hat_history.append([joints_hat])
            markers_hat_history.append([marker_pos_hat])
            markers_ori_hat_history.append([marker_ori_hat])

            # Iterative Error Feedback.
            for i in range(self.N):
                input_params = [inputs_flat,
                                pose_hat_history[-1][-1].clone().detach(),
                                shape_hat_history[-1][-1].clone().detach()]

                if self.use_gradient:
                    pose_hat_history[-1][-1].retain_grad()
                    shape_hat_history[-1][-1].retain_grad()
                    joints_hat_history[-1][-1].retain_grad()
                    markers_hat_history[-1][-1].retain_grad()
                    markers_ori_hat_history[-1][-1].retain_grad()

                    reconstruction_error = torch.zeros([1]).to(dtype=inputs_.dtype, device=inputs_.device)

                    if self.config.use_marker_pos:
                        marker_pos_in = inputs_flat[:, self.pos_d_start:self.pos_d_end]
                        reconstruction_error += reconstruction_loss(
                            marker_pos_in.reshape(batch_size, seq_length, -1, 3),
                            markers_hat_history[-1][-1].reshape(batch_size, seq_length, -1, 3)[:, :, self.marker_idxs],
                            batch_inputs['seq_lengths'], batch_inputs['marker_masks'])

                    if self.config.use_marker_ori:
                        marker_ori_in = inputs_flat[:, self.ori_d_start:self.ori_d_end]
                        reconstruction_error += reconstruction_loss(
                            marker_ori_in.reshape(batch_size, seq_length, -1, 9),
                            markers_ori_hat_history[-1][-1].reshape(batch_size, seq_length, -1, 9)[:, :, self.marker_idxs],
                            batch_inputs['seq_lengths'], batch_inputs['marker_masks'])

                    reconstruction_error.backward(retain_graph=True)

                    pose_hat_grad = pose_hat_history[-1][-1].grad.clone().detach() * batch_size * seq_length
                    shape_hat_grad = shape_hat_history[-1][-1].grad.clone().detach() * batch_size * seq_length

                    input_params.append(pose_hat_grad)
                    input_params.append(shape_hat_grad)

                inputs_ = torch.cat(input_params, dim=-1)

                pose_hat_delta = self.pose_net_iter(inputs_)
                shape_hat_delta = self.shape_net_iter(inputs_)
                if self.shape_avg:
                    shape_hat_delta = _to_single_shape(shape_hat_delta)

                pose_hat = pose_hat_history[-1][-1] + pose_hat_delta * self.step_size
                shape_hat = shape_hat_history[-1][-1] + shape_hat_delta * self.step_size
                marker_pos_hat, marker_ori_hat, joints_hat = self.get_estimated_real_markers(
                    pose_hat, shape_hat, offset_r_flat, offset_t_flat, self.vertex_ids)

                pose_hat_history[-1].append(pose_hat)
                shape_hat_history[-1].append(shape_hat)
                joints_hat_history[-1].append(joints_hat)
                markers_hat_history[-1].append(marker_pos_hat)
                markers_ori_hat_history[-1].append(marker_ori_hat)

            pose_hat_final = pose_hat_history[-1][-1].reshape((batch_size, seq_length, -1))
            shape_hat_final = shape_hat_history[-1][-1].reshape((batch_size, seq_length, -1))
            joints_hat_final = joints_hat_history[-1][-1].reshape((batch_size, seq_length, -1))

            all_model_out['pose_hat'].append(pose_hat_final[:, :, 3:])
            all_model_out['root_ori_hat'].append(pose_hat_final[:, :, :3])
            all_model_out['shape_hat'].append(shape_hat_final)
            all_model_out['joints_hat'].append(joints_hat_final)

        # History is kept in nested list of size (n_windows, n_history), merge to list of size (n_history, ).
        def _reshape(in_, out_):
            for h in range(self.N + 1):
                tmp = []
                for k in range(len(in_)):
                    dof = in_[k][h].shape[-1]
                    tmp.append(in_[k][h].reshape((batch.batch_size, -1, dof)))
                out_.append(torch.cat(tmp, dim=1))

        self.pose_hat_history = []
        self.shape_hat_history = []
        self.joints_hat_history = []
        self.markers_hat_history = []
        self.markers_ori_hat_history = []
        _reshape(pose_hat_history, self.pose_hat_history)
        _reshape(shape_hat_history, self.shape_hat_history)
        _reshape(joints_hat_history, self.joints_hat_history)
        _reshape(markers_hat_history, self.markers_hat_history)
        _reshape(markers_ori_hat_history, self.markers_ori_hat_history)

        model_out = {k: torch.cat(all_model_out[k], dim=1) for k in all_model_out}
        return model_out

    def backward(self, batch: ABatch, model_out, writer=None, global_step=None):
        """The backward pass."""
        batch_size, seq_length = batch.batch_size, batch.seq_length
        inputs_ = self.prepare_inputs(batch.get_inputs())
        marker_pos_in = inputs_[:, :, self.pos_d_start:self.pos_d_end]
        marker_ori_in = inputs_[:, :, self.ori_d_start:self.ori_d_end]
        markers_in = marker_pos_in.reshape((batch_size, seq_length, -1, 3))
        markers_ori_in = marker_ori_in.reshape((batch_size, seq_length, -1, 9))

        reconstruction_loss_total = torch.zeros(1).to(device=C.DEVICE)
        shape_loss_total = torch.zeros(1).to(device=C.DEVICE)
        pose_loss_total = torch.zeros(1).to(device=C.DEVICE)
        fk_loss_total = torch.zeros(1).to(device=C.DEVICE)

        for i in range(len(self.pose_hat_history)):
            pose_hat = self.pose_hat_history[i].reshape((batch_size, seq_length, -1))
            shape_hat = self.shape_hat_history[i].reshape((batch_size, seq_length, -1))

            pose_loss_total += padded_loss(torch.cat([batch.poses_root, batch.poses_body], dim=-1),
                                           pose_hat, self.smpl_loss, batch.seq_lengths)
            shape_loss_total += padded_loss(batch.shapes.unsqueeze(1).repeat((1, seq_length, 1)),
                                            shape_hat, self.smpl_loss, batch.seq_lengths)

            if self.do_fk:
                joints_gt = batch.joints_gt.reshape(batch.batch_size, batch.seq_length, -1, 3)
                joints_hat = model_out['joints_hat'].reshape(batch.batch_size, batch.seq_length, -1, 3)
                fk_loss_total += reconstruction_loss(joints_gt, joints_hat, batch.seq_lengths, batch.marker_masks)

            if self.config.use_marker_pos:
                markers_hat = self.markers_hat_history[i].reshape((batch_size, seq_length, -1, 3))
                reconstruction_loss_total += reconstruction_loss(markers_in, markers_hat[:, :, self.marker_idxs],
                                                               batch.seq_lengths, batch.marker_masks)

            if self.config.use_marker_ori:
                markers_ori_hat = self.markers_ori_hat_history[i].reshape((batch_size, seq_length, -1, 9))
                reconstruction_loss_total += reconstruction_loss(markers_ori_in, markers_ori_hat[:, :, self.marker_idxs],
                                                               batch.seq_lengths, batch.marker_masks)

        total_loss = self.pose_weight * pose_loss_total + self.fk_loss_weight * fk_loss_total
        total_loss += self.shape_weight * shape_loss_total + self.r_weight * reconstruction_loss_total
        total_loss = total_loss / len(self.pose_hat_history)

        loss_vals = {'pose': pose_loss_total.cpu().item() / len(self.pose_hat_history),
                     'shape': shape_loss_total.cpu().item() / len(self.pose_hat_history),
                     'reconstruction': reconstruction_loss_total.cpu().item() / len(self.pose_hat_history),
                     'fk': fk_loss_total.cpu().item() / len(self.joints_hat_history),
                     'total_loss': total_loss.cpu().item()}

        if writer is not None:
            self.log_loss_vals(loss_vals, writer, global_step)

        if self.training:
            total_loss.backward()

        return total_loss, loss_vals
