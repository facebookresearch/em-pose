"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""

import argparse
import torch

from collections import defaultdict
from tabulate import tabulate

from empose.bodymodels.smpl import create_default_smpl_model
from empose.helpers.configuration import CONSTANTS as C
from empose.helpers import utils as U
from empose.eval.helpers import load_model_and_eval_data
from empose.eval.helpers import window_generator
from empose.eval.metrics import MetricsEngine
from empose.nn.models import IterativeErrorFeedback


def main(config):
    config.n_samples = 1  # Set batch size to 1 so that we can evaluate sequences individually.
    visualize = config.visualize != -1
    b_visualize = config.visualize

    partition = 'test_real_0715' if config.cross_subject else 'test_real'
    net, test_loader, preprocess_fn, model_config = load_model_and_eval_data(config, partition=partition)
    model_dir = U.get_model_dir(C.EXPERIMENT_DIR, config.model_id)
    net.eval()

    smpl_model = create_default_smpl_model(C.DEVICE)
    me_all = MetricsEngine(smpl_model)
    me_ind = MetricsEngine(smpl_model)

    is_lgd = isinstance(net, IterativeErrorFeedback)
    window_size = 256 if is_lgd else None

    with torch.no_grad():

        metric_vals_ind = []
        for i, batch in enumerate(test_loader):
            if visualize and i != b_visualize:
                continue

            print("Evaluate {} ({} frames)".format(batch.ids[0], batch.seq_lengths[0]))

            batch = preprocess_fn(batch, mode='normalize_only')
            first_shape_hat = None
            model_out_all = defaultdict(list)
            for c, achunk in enumerate(window_generator(batch, window_size=window_size)):
                # Move data to GPU.
                chunk_gpu = achunk.to_gpu()

                # Preprocess.
                chunk_gpu = preprocess_fn(chunk_gpu, mode='after_normalize', reset_rng=(c+i == 0))

                # Get the predictions.
                model_out = net(chunk_gpu, is_new_sequence=(c == 0))

                # If we have several chunks, we take the shape of the first chunk for the entire sequence.
                if c == 0:
                    shape_hat = model_out['shape_hat'][:, 0] if model_out['shape_hat'] is not None else None
                    first_shape_hat = shape_hat
                else:
                    shape_hat = first_shape_hat

                me_all.compute(chunk_gpu.poses_body, chunk_gpu.shapes, model_out['pose_hat'],
                               shape_hat, chunk_gpu.seq_lengths,
                               chunk_gpu.poses_root, model_out['root_ori_hat'],
                               frame_mask=chunk_gpu.marker_masks)

                if c == 0:
                    me_ind.reset()

                me_ind.compute(chunk_gpu.poses_body, chunk_gpu.shapes, model_out['pose_hat'],
                               shape_hat, chunk_gpu.seq_lengths,
                               chunk_gpu.poses_root, model_out['root_ori_hat'],
                               frame_mask=chunk_gpu.marker_masks)

                for k in model_out:
                    model_out_all[k].append(model_out[k])
                model_out_all['shape_hat'][-1] = shape_hat

            metrics = me_ind.get_metrics()
            metric_vals_ind.append([chunk_gpu.ids[0]] + [metrics[k] for k in metrics])

            if i == b_visualize and visualize:
                # TODO
                print("Visualization not yet implemented.")

        metrics = me_all.get_metrics()
        metric_vals_ind.append(['Overall average'] + [metrics[k] for k in metrics])
        for i, me in enumerate(metric_vals_ind):
            metric_vals_ind[i] = [i] + me
        headers = [k for k in metrics]
        summary_string = tabulate(metric_vals_ind,
                                  headers=['Nr', 'E2E {}'.format(config.model_id)] + headers)
        print(summary_string)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', required=True, type=int, help='Which end-to-end model to evaluate.')
    p.add_argument('--visualize', type=int, default=-1, help='Visualize a sample.')
    p.add_argument('--cross_subject', action='store_true', help='Evaluate on hold-out subject 0715.')
    args = p.parse_args()
    main(args)
