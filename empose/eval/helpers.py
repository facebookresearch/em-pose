"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import collections
import numpy as np
import os
import torch

from empose.bodymodels.smpl import create_default_smpl_model
from empose.data.data import AMASSBatch
from empose.data.data import RealBatch
from empose.data.datasets import LMDBDataset, RealDataset
from empose.data.transforms import ExtractWindow
from empose.data.transforms import NormalizeRealMarkers
from empose.data.transforms import ToTensor
from empose.data.transforms import get_end_to_end_preprocess_fn
from empose.eval.metrics import MetricsEngine
from empose.helpers import utils as U
from empose.helpers.configuration import Configuration
from empose.helpers.configuration import CONSTANTS as C
from empose.nn.models import create_model
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def window_generator(batch: RealBatch, window_size):
    """Subdivide a batch into temporal windows of size `window_size`. This is helpful when GPU memory is limited."""
    if window_size is not None:
        assert isinstance(batch, RealBatch)
        seq_len = batch.seq_length
        n_windows = seq_len // window_size + int(seq_len % window_size > 0)

        for i in range(n_windows):
            sf = i * window_size
            ef = min((i + 1) * window_size, seq_len)
            seq_lengths_new = torch.from_numpy(np.array([ef - sf])).to(dtype=batch.seq_lengths.dtype,
                                                                       device=batch.seq_lengths.device)
            achunk = RealBatch(batch.ids, seq_lengths_new, batch.poses[:, sf:ef],
                               batch.shapes, batch.trans[:, sf:ef], batch.marker_pos_real[:, sf:ef],
                               batch.marker_ori_real[:, sf:ef], batch.marker_masks[:, sf:ef],
                               batch.offset_t, batch.offset_r)
            yield achunk
    else:
        yield batch


def evaluate(data_loader, net, preprocess_fn, metrics_engine, window_size=None):
    """
    Evaluate the model on a hold-out dataset.
    :param data_loader: The DataLoader object to loop over the validation set.
    :param net: The model.
    :param preprocess_fn: A function that preprocesses a batch.
    :param metrics_engine: Metrics engine to compute additional metrics other than the loss.
    :param window_size: Test Batches are too big to be evaluated in one go, so we use sliding windows.
    :return: All losses aggregated over the entire validation set.
    """
    # Put the model in evaluation mode.
    net.eval()
    loss_vals_agg = collections.defaultdict(float)
    n_samples = 0
    metrics_engine.reset()

    with torch.no_grad():

        for b, abatch in enumerate(data_loader):
            # We normalize here before we split into chunks because the normalization might be sequence dependent.
            abatch = preprocess_fn(abatch, mode='normalize_only')

            first_shape_hat = None
            loss_vals_seq = collections.defaultdict(float)
            for i, achunk in enumerate(window_generator(abatch, window_size)):
                # Move data to GPU.
                batch_gpu = achunk.to_gpu()

                # Preprocess.
                batch_gpu = preprocess_fn(batch_gpu, mode='after_normalize', reset_rng=(i+b == 0))

                # Get the predictions.
                model_out = net(batch_gpu, is_new_sequence=(i == 0))

                # Compute the loss.
                _, loss_vals = net.backward(batch_gpu, model_out)
                for k in loss_vals:
                    loss_vals_seq[k] += loss_vals[k]

                # Update the metrics.
                pose_hat = model_out['pose_hat'] if model_out['pose_hat'] is not None else batch_gpu.poses_body

                # If we have several chunks, we take the shape of the first chunk for the entire sequence.
                if i == 0:
                    shape_hat = model_out['shape_hat'][:, 0] if model_out['shape_hat'] is not None else None
                    first_shape_hat = shape_hat
                else:
                    shape_hat = first_shape_hat

                metrics_engine.compute(batch_gpu.poses_body, batch_gpu.shapes, pose_hat,
                                       shape_hat, batch_gpu.seq_lengths,
                                       batch_gpu.poses_root, model_out['root_ori_hat'],
                                       frame_mask=batch_gpu.marker_masks)

            for k in loss_vals_seq:
                loss_vals_agg[k] += loss_vals_seq[k] / (i+1) * batch_gpu.batch_size
            n_samples += batch_gpu.batch_size

    for k in loss_vals_agg:
        loss_vals_agg[k] /= n_samples
    return loss_vals_agg


def compute_loss_and_metrics(data_loader, net, preprocess_fn, model_id):
    """Loop over the given data set and compute loss and evaluation metrics."""
    # Prepare metrics engine.
    smpl_model = create_default_smpl_model()
    me = MetricsEngine(smpl_model)

    # Evaluate all validation samples.
    final_valid_loss = evaluate(data_loader, net, preprocess_fn, me)
    print('[LOSS] loss: {:.6f}'.format(final_valid_loss['total_loss']))

    # Compute metrics.
    valid_metrics = me.get_metrics()
    print(me.to_pretty_string(valid_metrics, model_id))

    return final_valid_loss, valid_metrics


def load_model_weights(checkpoint_file, net, state_key='model_state_dict'):
    """Loads a pre-trained model."""
    if not os.path.exists(checkpoint_file):
        raise ValueError("Could not find model checkpoint {}.".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    ckpt = checkpoint[state_key]
    net.load_state_dict(ckpt)


def get_model_config(model_id):
    """Load the configuration of the specified model."""
    model_id = model_id
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, model_id)
    model_config = Configuration.from_json(os.path.join(model_dir, 'config.json'))
    return model_config, model_dir


def load_model(model_id, is_valid=False):
    """Load a model and the corresponding preprocessing functions."""
    model_config, model_dir = get_model_config(model_id)
    smpl_model = create_default_smpl_model()

    preprocess_fn = get_end_to_end_preprocess_fn(model_config, smpl_model)
    net = create_model(model_config, smpl_model)

    net.to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(net)))

    # Load model weights.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    load_model_weights(checkpoint_file, net)
    print('Loaded weights from {}'.format(checkpoint_file))

    return net, model_config, model_dir, preprocess_fn


def load_model_and_eval_data(config, shuffle=False, partition='valid'):
    """Load model and the dataset."""
    assert partition in ['valid', 'test_real', 'test_real_0715']

    net, model_config, _, preprocess_fn = load_model(config.model_id, is_valid=(partition == 'valid'))

    ws = config.seq_length if hasattr(config, 'seq_length') else model_config.window_size
    bs = config.n_samples if hasattr(config, 'n_samples') else 6

    if partition == 'valid':
        transform = [ExtractWindow(ws, mode='middle'),
                     ToTensor()]
        transform = transforms.Compose(transform)
        valid_data = LMDBDataset(os.path.join(os.path.dirname(C.DATA_DIR), "3dpw_lmdb"), transform=transform)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=shuffle,
                                 num_workers=model_config.data_workers,
                                 collate_fn=AMASSBatch.from_sample_list)
    else:
        test_transform = transforms.Compose([NormalizeRealMarkers(),
                                             ToTensor()])
        partition_to_dir = {'test_real': C.DATA_DIR_TEST,
                            'test_real_0715': os.path.join(C.DATA_DIR_TEST, 'hold_out')}
        test_dir = partition_to_dir[partition]
        test_data = RealDataset(test_dir, transform=test_transform)
        eval_loader = DataLoader(test_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=RealBatch.from_sample_list)

    net.eval()
    return net, eval_loader, preprocess_fn, model_config
