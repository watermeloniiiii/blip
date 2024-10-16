#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authored by Chenxi
"""

import os
import math
import torch
import numpy as np
import torch.optim as optim
from typing import List
from accelerate.utils import DummyOptim, DummyScheduler
from collections import defaultdict

from torch.optim.lr_scheduler import (
    StepLR,
    CyclicLR,
    OneCycleLR,
)
from transformers import (
    MaskFormerImageProcessor,
    SegformerImageProcessor,
    Mask2FormerImageProcessor,
)
from transformers.image_utils import make_list_of_images

from typing import Optional
from common.logger import logger
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

from config.setup import SCHEDULER
from config.setup import default_setup

config = default_setup("./config/model_config.yaml")

IMAGE_PROCESSOR = {
    "segformer": SegformerImageProcessor(),
    "dinov2": SegformerImageProcessor(),
    "maskformer": MaskFormerImageProcessor(),
    "mask2former": Mask2FormerImageProcessor(),
}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def collate_fn(data):
    for i in range(0, len(data)):
        data[i]["label"] = data[i]["label"].sum(axis=0)
    patch = []
    name = []
    image = torch.stack([torch.from_numpy(b["image"]) for b in data], 0)
    label = torch.stack([torch.from_numpy(b["label"]) for b in data], 0)[
        :, np.newaxis, :, :
    ]
    patch = patch.append(b["patch"] for b in data)
    name = name.append(b["name"] for b in data)
    return {"image": image, "patch": patch, "name": name, "label": label}


def make_cuda_list(data: List):
    data = [d.cuda() for d in data]
    return data


class Trainer(object):
    def __init__(self, net):
        self.net = net

    def _select_optimizer(self):
        """
        initialize an optimizer from either the definition of deepspeed optimizer or user-defined optimizer
        NOTE that the user-defined optimizer would be prioritized
        """
        user_defined_optimizer = config.MODEL.optimization.optimizer is not None
        deepspeed_defined_optimizer = not (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        assert (
            user_defined_optimizer or deepspeed_defined_optimizer
        ), "Please provide at least one optimizer from either deepspeed-defined or user-defined"

        optimizer = None
        # if user-defined optimizer is available
        if user_defined_optimizer:
            if "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                del self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "optimizer"
                ]
            if config.MODEL.optimization.optimizer == "Adam":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=config.MODEL.optimization.base_lr,
                    weight_decay=config.MODEL.optimization.weight_decay,
                )
            elif config.MODEL.optimization.optimizer == "SGD":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=config.MODEL.optimization.base_lr,
                    weight_decay=config.MODEL.optimization.optimizerweight_decay,
                    momentum=config.MODEL.optimization.momentum,
                )
            elif config.MODEL.optimization.optimizer == "AdamW":
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=config.MODEL.optimization.base_lr,
                    weight_decay=config.MODEL.optimization.weight_decay,
                )
        # otherwise if there's no user-defined optimizer but a deepspeed optimizer
        if not user_defined_optimizer and deepspeed_defined_optimizer:
            optimizer_cls = DummyOptim
            lr = self.accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"][
                "params"
            ].get("lr", 1e-5)
            optimizer = optimizer_cls(self.net.parameters(), lr=lr)
        return optimizer

    def _makefolders(self):
        """
        This function is used to create necessary folders to save models, textbooks and images
        :return:
        """
        model_folder = config.PATH.model_outdir
        model_path = os.path.join(model_folder, config.MODEL_INFO.model_name)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        # os.makedirs(os.path.join(model_path, "latest"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "best"), exist_ok=True)
        self.model_folder = model_folder
        self.model_path = model_path

    def _select_scheduler(self):
        """
        initialize an optimizer from either the definition of deepspeed optimizer or user-defined optimizer
        NOTE that the user-defined optimizer would be prioritized
        """
        user_defined_scheduler = config.MODEL.optimization.scheduler is not None
        deepspeed_defined_scheduler = not (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        assert (
            user_defined_scheduler or deepspeed_defined_scheduler
        ), "Please provide at least one scheduler from either deepspeed-defined or user-defined"
        if user_defined_scheduler:
            if "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                del self.accelerator.state.deepspeed_plugin.deepspeed_config[
                    "scheduler"
                ]
            if config.MODEL.optimization.scheduler == "StepLR":
                return StepLR(
                    self.optimizer,
                    step_size=SCHEDULER["StepLR"]["step_size"] * len(self.train_loader),
                    gamma=SCHEDULER["StepLR"]["gamma"],
                )
            elif config.MODEL.optimization.scheduler == "CLR":
                return CyclicLR(
                    self.optimizer,
                    base_lr=SCHEDULER["CLR"]["base_lr"],
                    max_lr=SCHEDULER["CLR"]["max_lr"],
                    step_size_up=SCHEDULER["CLR"]["step_size"] * len(self.train_loader),
                )
            elif config.MODEL.optimization.scheduler == "ONECLR":
                return OneCycleLR(
                    self.optimizer,
                    max_lr=SCHEDULER["ONECLR"]["max_lr"],
                    steps_per_epoch=len(self.train_loader)
                    // self.accelerator.gradient_accumulation_steps,
                    pct_start=SCHEDULER["ONECLR"]["pct_start"],
                    div_factor=SCHEDULER["ONECLR"]["div_factor"],
                    epochs=self.epoch,
                )
        if not user_defined_scheduler and deepspeed_defined_scheduler:
            # remember to revise step-relevant parameters
            deepspeed_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if "warmup_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = (
                    math.ceil(
                        len(self.train_loader)
                        * self.epoch
                        * config.MODEL.optimization.warmup_steps_ratio
                    )
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            if "total_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["total_num_steps"] = (
                    math.ceil(
                        len(self.train_loader)
                        * self.epoch
                        * config.MODEL.optimization.total_steps_ratio
                    )
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            return DummyScheduler(self.optimizer)

    def process_model(self, net, inputs, input_values, tensor_type):
        image = input_values[inputs.index("image")].type(tensor_type)
        label = input_values[inputs.index("label")].squeeze()
        date = input_values[inputs.index("date")].squeeze()
        outputs = net(image, label, date)
        return outputs.loss, outputs

    def training(self, epoch):
        self.net.train()
        total_sample_met = 0
        for idx, sample in enumerate(self.train_loader, 0):
            total_sample_met += (
                self.train_loader.batch_sampler.batch_size * self.num_processes
            )
            with self.accelerator.accumulate(self.net):
                self.cur_step = (
                    (self.scheduler.scheduler.last_batch_iteration + 1)
                    * self.num_processes
                    if "scheduler"
                    in self.accelerator.state.deepspeed_plugin.deepspeed_config
                    else self.scheduler.scheduler.last_epoch
                )
                logger.info(
                    f"Batch: {idx}/{len(self.train_loader)} \
                    ----- Epoch: {epoch} \
                    ----- Rank: {self.accelerator.local_process_index}\
                    ----- Step: {self.cur_step}\
                    ----- lr: {get_lr(self.optimizer)}\
                    ----- sample_process: {self.train_loader.batch_sampler.batch_size * (self.accelerator.local_process_index + 1)}/{self.train_loader.batch_sampler.batch_size * self.num_processes}\
                    ----- sample_total: {total_sample_met}"
                )
                self.optimizer.zero_grad()
                inputs, input_values = zip(*sample.items())
                tensor_type = (
                    torch.cuda.FloatTensor
                    if self.accelerator.device.type == "cuda"
                    else torch.FloatTensor
                )
                loss, _ = self.process_model(
                    self.net, inputs, input_values, tensor_type
                )
                gathered_loss = self.accelerator.gather_for_metrics(loss)
                self.train_loss[epoch] += torch.mean(gathered_loss)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {
                            "train_loss": torch.mean(gathered_loss).item(),
                            "learning_rate": get_lr(self.optimizer),
                        },
                        step=self.cur_step,
                    )

    def evaluation(self, epoch):
        self.net.eval()
        total_sample_met = 0
        with torch.no_grad():
            for idx, sample in enumerate(self.vali_loader, 0):
                total_sample_met += (
                    self.train_loader.batch_sampler.batch_size * self.num_processes
                )
                logger.info(
                    f"Batch: {idx}/{len(self.vali_loader)} \
                    ----- Epoch: {epoch} \
                    ----- Rank: {self.accelerator.local_process_index}\
                    ----- Step: {self.cur_step} \
                    ----- sample_process: {self.train_loader.batch_sampler.batch_size * (self.accelerator.local_process_index + 1)}/{self.train_loader.batch_sampler.batch_size * self.num_processes} \
                    ----- sample_total: {total_sample_met}"
                )
                inputs, input_values = zip(*sample.items())
                tensor_type = (
                    torch.cuda.FloatTensor
                    if self.accelerator.device.type == "cuda"
                    else torch.FloatTensor
                )
                loss, outputs = self.process_model(
                    self.net, inputs, input_values, tensor_type
                )
                outputs = SegformerImageProcessor().post_process_semantic_segmentation(
                    outputs=outputs,
                    target_sizes=[
                        (
                            config.MODEL.optimization.img_size,
                            config.MODEL.optimization.img_size,
                        )
                    ]
                    * input_values[0].shape[0],
                )  # B, C, H, W
                prediction = torch.stack(outputs, dim=0)
                # prediction = (torch.nn.functional.sigmoid(outputs) > 0.5).to(torch.long)
                labels = input_values[inputs.index("label")]

                IOU_metric = MeanIoU(
                    num_classes=len(config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                Precision_metric = MulticlassPrecision(
                    num_classes=len(config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                Recall_metric = MulticlassRecall(
                    num_classes=len(config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                F1_metric = MulticlassF1Score(
                    num_classes=len(config.MODEL_INFO.class_of_interest) + 1
                ).cuda()
                if len(labels.shape) != len(prediction.shape):
                    labels = labels.squeeze()
                    prediction = prediction.squeeze()
                IOU = IOU_metric(prediction, labels.type(torch.cuda.LongTensor))
                Prec = Precision_metric(prediction, labels.type(torch.cuda.LongTensor))
                Rec = Recall_metric(prediction, labels.type(torch.cuda.LongTensor))
                F1 = F1_metric(prediction, labels.type(torch.cuda.LongTensor))
                gathered_metrics = self.accelerator.gather_for_metrics(
                    (IOU, Prec, Rec, F1)
                )
                for m_idx, m in enumerate(["IOU", "Precision", "Recall", "F1"]):
                    self.metric[m][epoch] += torch.mean(gathered_metrics[m_idx])

            gathered_loss = self.accelerator.gather_for_metrics(loss)
            self.vali_loss[epoch] += torch.mean(gathered_loss)
            cur_loss = self.vali_loss[epoch]

            if self.accelerator.is_local_main_process:
                self.accelerator.log(
                    {
                        "vali_loss": self.vali_loss[epoch].item()
                        / len(self.vali_loader),
                        "iou": self.metric["IOU"][epoch].item() / len(self.vali_loader),
                        "precision": self.metric["Precision"][epoch].item()
                        / len(self.vali_loader),
                        "recall": self.metric["Recall"][epoch].item()
                        / len(self.vali_loader),
                        "f1": self.metric["F1"][epoch].item() / len(self.vali_loader),
                    },
                    step=self.cur_step,
                )
            save_best_flag = False
            if self.best_loss is None or self.best_loss > cur_loss:
                self.best_loss = cur_loss
                save_best_flag = True

            self._makefolders()
            if save_best_flag:
                self.net.save_checkpoint(
                    save_dir=os.path.join(
                        config.PATH.model_outdir, config.MODEL_INFO.model_name
                    ),
                    tag="best",
                )

    def train_model(
        self,
        epoch: int,
        train_loader: DataLoader,
        vali_loader: Optional[DataLoader] = None,
        accelerator=None,
        gpu_id=None,
    ) -> None:
        """
        The main function to execute model training
        """

        self.epoch = epoch
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.gpu_id = gpu_id
        self.accelerator = accelerator
        self.num_processes = self.accelerator.state.num_processes
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()
        self.train_loss = np.zeros([epoch])
        self.vali_loss = np.zeros([epoch])
        self.metric = defaultdict(lambda: np.zeros([epoch]))
        self.best_loss = None
        self.login = False

        self.accelerator.init_trackers(
            project_name=f"blip2_pretrain",
            config=dict(config),
            init_kwargs={
                "wandb": {
                    "entity": "chenxilin",
                    "name": "test",
                }
            },
        )
        (
            self.net,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.vali_loader,
        ) = self.accelerator.prepare(
            self.net,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.vali_loader,
        )
        for i in range(epoch):
            self.training(epoch=i)
            self.evaluation(epoch=i)
        self.accelerator.end_training()
