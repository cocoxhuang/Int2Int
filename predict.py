# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import pickle
import argparse

import src
from src.utils import bool_flag, initialize_exp
from src.envs import ENVS, build_env
from src.model import check_model_params, build_modules
from src.trainer import Trainer
from src.evaluator import Evaluator, idx_to_infix
from src.utils import to_cuda
from src.slurm import init_signal_handler, init_distributed_mode
from src.slurm import init_signal_handler, init_distributed_mode
from train import get_parser

def main(params):
    # load params from experiment folder
    pickle_path = os.path.join(params.eval_from_exp, "params.pkl")
    assert os.path.isfile(pickle_path), f"Missing file: {pickle_path}"
    loaded_params = pickle.load(open(pickle_path, "rb")).__dict__

    output_path = os.path.join(params.eval_from_exp, "output.txt")
    # remove output file if it exists
    if os.path.isfile(output_path):
        os.remove(output_path)
    with open(output_path, "w") as f:
        f.write("input\tpredict\tlabel\n")

    # override relevant fields
    for k in loaded_params:
        if not hasattr(params, k):
            setattr(params, k, loaded_params[k])
    params.eval_only = True
    params.reload_model = os.path.join(params.eval_from_exp, "checkpoint.pth")
    params.dump_path = params.eval_from_exp
    params.train_data = ""
    params.is_slurm_job = False
    params.local_rank = -1

    # distributed / CUDA
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()
    src.utils.CUDA = not params.cpu

    # build everything
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    if params.architecture != "decoder_only":
        encoder = (
            trainer.modules["encoder"].module
            if params.multi_gpu
            else trainer.modules["encoder"]
        )
        encoder.eval()
    if params.architecture != "encoder_only":
        decoder = (
            trainer.modules["decoder"].module
            if params.multi_gpu
            else trainer.modules["decoder"]
        )
        decoder.eval()

    # run predictions on valid set
    iterator = env.create_test_iterator(
            "valid",
            params.tasks[0],
            data_path=params.eval_data.split(',') if params.eval_data != "" else None,
            batch_size=params.batch_size_eval,
            params=params,
            size=params.eval_size,
        )
    eval_size = len(iterator.dataset)

    max_len = params.max_output_len + 2

    # print iterator size
    with torch.no_grad():
        for (x1, len1), (x2, len2), _ in iterator:
            x1_, len1_, x2_, len2_ = to_cuda(x1, len1, x2, len2)
            # target words to predict
            if params.architecture != "encoder_only":
                alen = torch.arange(len2_.max(), dtype=torch.long, device=len2_.device)
                pred_mask = (
                    alen[:, None] < len2_[None] - 1
                )  # do not predict anything given the last target word
                y = x2_[1:].masked_select(pred_mask[:-1])
                assert len(y) == (len2_ - 1).sum().item()
            else: 
                alen = torch.arange(len1_.max(), dtype=torch.long, device=len2_.device)
                pred_mask = (
                    (alen[:, None] < len2_[None]) # & (alen[:, None] > torch.zeros_like(len2)[None])
                )
                y= torch.cat((x2_,torch.full((len1_.max()-len2_.max(),len2_.size(0)),self.env.eos_index,device=len2_.device)),0)
                y = y.masked_select(pred_mask)

            bs = len(len1_)

            # forward / loss
            if params.architecture == "encoder_decoder":
                if params.lstm:
                    _, hidden = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                    decoded, _ = decoder(
                        "fwd",
                        x=x2_,
                        lengths=len2_,
                        causal=True,
                        src_enc=hidden,
                    )
                    word_scores, loss = decoder(
                        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                    )
                else:
                    encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                    decoded = decoder(
                        "fwd",
                        x=x2_,
                        lengths=len2_,
                        causal=True,
                        src_enc=encoded.transpose(0, 1),
                        src_len=len1_,
                    )
                    word_scores, loss = decoder(
                        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                    )
            elif params.architecture == "encoder_only":
                encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
                word_scores, loss = encoder(
                    "predict", tensor=encoded, pred_mask=pred_mask, y=y, get_scores=True
                )
            else:
                decoded = decoder("fwd", x=x2_, lengths=len2_, causal=True, src_enc=None, src_len=None)
                word_scores, loss = decoder(
                    "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
                )

            
            # # reshape x,y back to batch x sequence
            x1_ = x1_.transpose(0, 1)
            y = y.view(-1, bs).transpose(0, 1)
            # same for generated
            generated = word_scores.max(1)[1]
            if params.architecture == "encoder_only":
                generated = torch.cat((generated, torch.full((max_len - generated.size(0), bs), env.eos_index, device=generated.device)), 0)
            generated = generated.view(-1, bs).transpose(0, 1)

            for i in range(generated.size(0)):
                with open(output_path, "a") as f:
                    f.write(
                        f"{idx_to_infix(env, x1_[i].tolist())}\t{idx_to_infix(env,generated[i].tolist(),input = False)}\t{idx_to_infix(env, y[i].tolist(), input=False)}\n"
                    )

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    check_model_params(params)
    main(params)