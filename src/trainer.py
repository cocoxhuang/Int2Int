# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import io
import sys
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .optim import get_optimizer
from .utils import to_cuda

logger = getLogger()


class Trainer(object):
    def __init__(self, modules, env, params):
        """
        Initialize trainer.
        """
        # modules / params
        self.modules = modules
        self.params = params
        self.env = env

        assert self.params.report_loss_every > 0

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        self.total_samples = 0
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None
        if params.amp >= 0:
            self.init_amp()
        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m[1:], False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0)]
            + [("processed_w", 0)]
            + [("encoder_act_penalty", [])]
            + [("decoder_act_penalty", [])]
            + sum(
                [[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []
            )
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # file handler to export data
        if params.export_data:
            assert params.train_data == ""
            params.export_path_prefix = os.path.join(params.dump_path, "data.prefix")
            self.file_handler_prefix = io.open(
                params.export_path_prefix, mode="a", encoding="utf-8"
            )
            logger.info(
                f"Data will be stored in prefix in: {params.export_path_prefix} ..."
            )

        # reload exported data
        if params.train_data != "":
            assert params.num_workers in [0, 1]
            assert params.export_data is False
            self.data_path = params.train_data
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, self.data_path, params))
                for task in params.tasks
            }

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        # mod_names = sorted(self.modules.keys()) # unused
        self.scaler = torch.cuda.amp.GradScaler()

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.params.report_loss_every == 0:
            self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = (
            (" - LR: ")
            + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} examples/s - {:8.2f} words/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            v.load_state_dict(data[k])

        # reload optimizer
        # AMP checkpoint reloading is buggy, we cannot reload optimizer
        # instead, we only reload current iterations / learning rates
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])
        
        if self.params.fp16:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            assert self.scaler is None and "scaler" not in data

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch = next(self.dataloader[task])
        except Exception as e:
            logger.error(
                "An unknown exception of type {0} occurred in line {1} when fetching batch. "
                "Arguments:{2!r}. Restarting ...".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args
                )
            )
            if self.params.is_slurm_job:
                if int(os.environ["SLURM_PROCID"]) == 0:
                    logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    def export_data(self, task):
        """
        Export data to the disk.
        """
        env = self.env
        (x1, len1), (x2, len2), _ = self.get_batch(task)
        for i in range(len(len1)):
            # prefix assuming encoder decoder...
            # no bos
            prefix1 = [env.id2word[wid] for wid in x1[0: len1[i] - 1, i].tolist()]
            # bos 
            prefix2 = [env.id2word[wid] for wid in x2[1: len2[i] - 1, i].tolist()]
            # save
            prefix1_str = " ".join(prefix1)
            prefix2_str = " ".join(prefix2)
            self.file_handler_prefix.write(f"{prefix1_str}\t{prefix2_str}\n")
            self.file_handler_prefix.flush()
            # self.EQUATIONS[(prefix1_str, prefix2_str)] = self.EQUATIONS.get((prefix1_str, prefix2_str), 0) + 1

        # number of processed sequences / words
        self.n_equations += self.params.batch_size
        self.total_samples += self.params.batch_size
        self.stats["processed_e"] += len1.size(0)
        self.stats["processed_w"] += (len1 + len2 - 2).sum().item()

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        
        if params.architecture == "decoder_only":
            # batch
            (x2, len2), _ = self.get_batch(task)
            # cuda
            x2, len2 = to_cuda(x2, len2)
        else:
            # batch
            (x1, len1), (x2, len2), _ = self.get_batch(task)
            # cuda
            x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)

        # target words to predict
        if params.architecture != "encoder_only":
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
        else: 
            alen = torch.arange(len1.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                (alen[:, None] < len2[None])
            )
            y= torch.cat((x2,torch.full((len1.max()-len2.max(),len2.size(0)),self.env.eos_index,device=len2.device)),0)
            y = y.masked_select(pred_mask)
  
        if params.architecture != "decoder_only":
            encoder = (
                self.modules["encoder"].module
                if params.multi_gpu
                else self.modules["encoder"]
            )
            encoder.train()
        
        if params.architecture != "encoder_only":
            decoder = (
                self.modules["decoder"].module
                if params.multi_gpu
                else self.modules["decoder"]
            )
            decoder.train()

        # forward / loss
        if params.architecture == "encoder_decoder":
            if params.lstm:
                if params.amp == -1:
                    _, hidden = encoder("fwd", x=x1, lengths=len1, causal=False)
                    decoded, _ = decoder(
                        "fwd",
                        x=x2,
                        lengths=len2,
                        causal=True,
                        src_enc=hidden,
                    )
                    _, loss = decoder(
                        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False
                    )
                else:
                    with torch.cuda.amp.autocast():
                        _, hidden = encoder("fwd", x=x1, lengths=len1, causal=False)
                        decoded, _ = decoder(
                            "fwd",
                            x=x2,
                            lengths=len2,
                            causal=True,
                            src_enc=hidden,
                        )
                        _, loss = decoder(
                            "predict",
                            tensor=decoded,
                            pred_mask=pred_mask,
                            y=y,
                            get_scores=False,
                        )
            else:
                if params.amp == -1:
                    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                    decoded = decoder(
                        "fwd",
                        x=x2,
                        lengths=len2,
                        causal=True,
                        src_enc=encoded.transpose(0, 1),
                        src_len=len1,
                    )
                    _, loss = decoder(
                        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False
                    )
                else:
                    with torch.cuda.amp.autocast():
                        encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                        decoded = decoder(
                            "fwd",
                            x=x2,
                            lengths=len2,
                            causal=True,
                            src_enc=encoded.transpose(0, 1),
                            src_len=len1,
                        )
                        _, loss = decoder(
                            "predict",
                            tensor=decoded,
                            pred_mask=pred_mask,
                            y=y,
                            get_scores=False,
                        )
                if encoder.act:
                    loss = loss + encoder.layers[encoder.loop_idx].ponder_penalty
                if decoder.act:
                    loss = loss + decoder.layers[decoder.loop_idx].ponder_penalty
        elif params.architecture == "encoder_only":
            if params.amp == -1:
                encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                _, loss = encoder(
                    "predict", tensor=encoded, pred_mask=pred_mask, y=y, get_scores=False
                )
            else:
                with torch.cuda.amp.autocast():
                    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                    _, loss = encoder(
                        "predict",
                        tensor=encoded,
                        pred_mask=pred_mask,
                        y=y,
                        get_scores=False,
                    )
            if encoder.act:
                loss = loss + encoder.layers[encoder.loop_idx].ponder_penalty
        else:
            if params.amp == -1:
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=None,
                    src_len=None,
                )
                _, loss = decoder(
                    "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False
                )
            else:
                with torch.cuda.amp.autocast():
                    decoded = decoder(
                        "fwd",
                        x=x2,
                        lengths=len2,
                        causal=True,
                        src_enc=None,
                        src_len=None,
                    )
                    _, loss = decoder(
                        "predict",
                        tensor=decoded,
                        pred_mask=pred_mask,
                        y=y,
                        get_scores=False,
                    )
            if decoder.act:
                loss = loss + decoder.layers[decoder.loop_idx].ponder_penalty


        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)
        
        # number of processed sequences / words
        self.n_equations += params.batch_size
        if params.architecture == "decoder_only":
            self.stats["processed_e"] += len2.size(0)
            self.stats["processed_w"] += (len2 - 3).sum().item()
        else:
            self.stats["processed_e"] += len1.size(0)
            self.stats["processed_w"] += (len1 + len2 - 2).sum().item()
