"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import glob
import numpy as np
import os
import time
import sys
import torch
import torch.optim as optim

from empose.bodymodels.smpl import create_default_smpl_model
from empose.data.data import AMASSBatch
from empose.data.data import RealBatch
from empose.data.datasets import RealDataset
from empose.data.datasets import LMDBDataset
from empose.data.transforms import ExtractWindow
from empose.data.transforms import ToTensor
from empose.data.transforms import NormalizeRealMarkers
from empose.data.transforms import get_end_to_end_preprocess_fn
from empose.eval.helpers import evaluate
from empose.eval.helpers import load_model_weights
from empose.eval.metrics import MetricsEngine
from empose.helpers import utils as U
from empose.helpers.configuration import Configuration
from empose.helpers.configuration import CONSTANTS as C
from empose.nn.models import create_model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def main(config):
    # Fix seed.
    if config.seed is None:
        config.seed = int(time.time())

    # Create access to the data.
    rng_extractor = np.random.RandomState(4313)
    train_transform = transforms.Compose([ExtractWindow(config.window_size, rng_extractor, mode='random'),
                                          ToTensor()])
    valid_transform = transforms.Compose([ExtractWindow(config.window_size, mode='middle'),
                                          ToTensor()])
    test_transform = transforms.Compose([NormalizeRealMarkers(),
                                         ToTensor()])

    train_data = LMDBDataset(os.path.join(C.DATA_DIR, "amass_lmdb"), transform=train_transform)
    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "3dpw_lmdb"), transform=valid_transform)

    train_loader = DataLoader(train_data,
                              batch_size=config.bs_train,
                              shuffle=True,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)
    valid_loader = DataLoader(valid_data,
                              batch_size=config.bs_eval,
                              shuffle=False,
                              num_workers=config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)
    test_data = RealDataset(C.DATA_DIR_TEST, transform=test_transform)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=RealBatch.from_sample_list)

    # Create the SMPL model which is also a trainable layer.
    smpl_model = create_default_smpl_model()

    preprocess_fn = get_end_to_end_preprocess_fn(config, smpl_model, randomize_if_configured=True)
    preprocess_fn_valid = get_end_to_end_preprocess_fn(config, smpl_model, randomize_if_configured=False)
    preprocess_fn_test = preprocess_fn_valid
    net = create_model(config, smpl_model)

    # Prepare metrics engine.
    me = MetricsEngine(smpl_model)

    # Create or locate the experiment folder.
    experiment_id = config.experiment_id
    experiment_name = net.model_name()
    experiment_name += "{}{}{}".format('-pos' if config.use_marker_pos else '',
                                       '-ori' if config.use_marker_ori else '',
                                       '-nor' if config.use_marker_nor else '')
    experiment_name += "{}{}".format(
        '-noise-supp-{}'.format(config.suppression_noise_length) if config.suppression_noise_length > 0.0 else '',
        '-noise-spher-{}'.format(config.spherical_noise_strength) if config.spherical_noise_strength > 0.0 else '')
    if config.test:
        experiment_name += '--TEST'

    if experiment_id is None:
        experiment_id = int(time.time())
        model_dir = U.create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)
    else:
        model_dir = U.get_model_dir(C.EXPERIMENT_DIR, experiment_id)
        if config.load:
            if model_dir is None or not os.path.exists(model_dir):
                raise ValueError("Cannot find model directory for experiment ID {}".format(experiment_id))
        else:
            if model_dir is not None:
                raise ValueError("Model directory for experiment ID {} already exists. "
                                 "Did you mean to use --load?".format(experiment_id))
            else:
                model_dir = U.create_model_dir(C.EXPERIMENT_DIR, experiment_id, experiment_name)

    # Save code as zip into the model directory.
    code_files = glob.glob('./*.py', recursive=False)
    U.zip_files(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))
    checkpoint_file = os.path.join(model_dir, 'model.pth')

    # Save the command line that was used to the model directory.
    cmd = sys.argv[0] + ' ' + ' '.join(sys.argv[1:])
    with open(os.path.join(model_dir, 'cmd.txt'), 'w') as f:
        f.write(cmd)

    net.to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(net)))
    print('Saving checkpoints to {}'.format(checkpoint_file))

    # Optimizer.
    optimizer = optim.Adam(net.parameters(), lr=config.lr)

    # Tensorboard logger.
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    # Training loop.
    global_step = 0
    best_valid_loss = float('inf')
    for epoch in range(config.n_epochs):

        for i, abatch in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()

            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Remaining preprocessing.
            batch_gpu = preprocess_fn(batch_gpu)

            # Get the predictions.
            model_out = net(batch_gpu)

            # Compute gradients.
            loss, loss_vals = net.backward(batch_gpu, model_out, writer, global_step)

            # Update params.
            optimizer.step()

            elapsed = time.time() - start

            if i % (config.print_every - 1) == 0:
                loss_string = ' '.join(['{}: {:.6f}'.format(k, loss_vals[k]) for k in loss_vals])
                print('[TRAIN {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))
                me.reset()
                pose_hat = model_out['pose_hat'] if model_out['pose_hat'] is not None else batch_gpu.poses_body
                me.compute(batch_gpu.poses_body, batch_gpu.shapes, pose_hat, model_out['shape_hat'],
                           batch_gpu.seq_lengths)
                me.to_tensorboard_log(me.get_metrics(), writer, global_step, 'train')

            writer.add_scalar('lr', config.lr, global_step)

            if global_step % (config.eval_every - 1) == 0:
                # Evaluate on validation set.
                start = time.time()
                valid_losses = evaluate(valid_loader, net, preprocess_fn_valid, me)
                valid_metrics = me.get_metrics()
                elapsed = time.time() - start

                # Log to console.
                loss_string = ' '.join(['{}: {:.6f}'.format(k, valid_losses[k]) for k in valid_losses])
                print('[VALID {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed))

                # Evaluate on test set.
                start = time.time()
                test_losses = evaluate(test_loader, net, preprocess_fn_test, me, window_size=config.eval_window_size)
                test_metrics = me.get_metrics()
                elapsed = time.time() - start

                loss_string = ' '.join(['{}: {:.6f}'.format(k, test_losses[k]) for k in test_losses])
                print('[TEST {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    i + 1, epoch + 1, loss_string, elapsed), end='')

                current_eval_loss = test_losses['total_loss']
                if current_eval_loss < best_valid_loss:
                    print(' ***')
                    best_valid_loss = current_eval_loss

                    torch.save({
                        'iteration': i,
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': loss,
                        'valid_loss': valid_losses['total_loss'],
                        'test_eucl_mean': test_metrics['MPJPE [mm]'],
                        'test_angle_mean': test_metrics['MPJAE [deg]']
                    }, checkpoint_file)

                else:
                    print()

                # Pretty print metrics.
                print(me.to_pretty_string(valid_metrics, "{} {}".format(experiment_id, 'VALID')))
                print(me.to_pretty_string(test_metrics, "{} {}".format(experiment_id, 'TEST')))

                # Log to tensorboard.
                for k in valid_losses:
                    writer.add_scalar('{}/valid'.format(k), valid_losses[k], global_step)
                me.to_tensorboard_log(valid_metrics, writer, global_step, 'valid')

                for k in test_losses:
                    writer.add_scalar('{}/test'.format(k), test_losses[k], global_step)
                me.to_tensorboard_log(test_metrics, writer, global_step, 'test')

                # Some more book-keeping.
                net.train()
            global_step += 1

    # Training is done, evaluate on validation set again.
    load_model_weights(checkpoint_file, net)
    final_valid_losses = evaluate(valid_loader, net, preprocess_fn_valid, me)
    loss_string = ' '.join(['{}: {:.6f}'.format(k, final_valid_losses[k]) for k in final_valid_losses])
    print('[VALID FINAL] {}'.format(loss_string))

    print('FINAL VALIDATION METRICS')
    valid_metrics = me.get_metrics()
    print(me.to_pretty_string(valid_metrics, experiment_id))
    me.to_tensorboard_log(valid_metrics, writer, global_step, 'valid')

    # Evaluate on test again.
    final_test_losses = evaluate(test_loader, net, preprocess_fn_test, me, window_size=config.eval_window_size)
    loss_string = ' '.join(['{}: {:.6f}'.format(k, final_test_losses[k]) for k in final_test_losses])
    print('[TEST FINAL] {}'.format(loss_string))

    print('FINAL TEST METRICS')
    test_metrics = me.get_metrics()
    print(me.to_pretty_string(test_metrics, experiment_id))
    me.to_tensorboard_log(test_metrics, writer, global_step, 'test')


if __name__ == '__main__':
    main(Configuration.parse_cmd())
