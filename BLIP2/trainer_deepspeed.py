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

from collections import defaultdict

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

from base_trainer import Trainer

warnings.filterwarnings("ignore")

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


class blip2_trainer(Trainer):
    def __init__(self, net, config) -> None:
        super().__init__(self, net, config)
        
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
                        self.config.PATH.model_outdir, self.config.MODEL_INFO.model_name
                    ),
                    tag="best",
                )

    def train_model(
        self,
        train_loader: DataLoader,
        vali_loader: Optional[DataLoader] = None,
        accelerator=None,
        gpu_id=None,
    ) -> None:
        """
        The main function to execute model training
        """
        self.epoch = self.config.optimization.epoch
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.gpu_id = gpu_id
        self.accelerator = accelerator
        self.num_processes = self.accelerator.state.num_processes
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()
        self.train_loss = np.zeros(self.epoch)
        self.vali_loss = np.zeros(self.epoch)
        self.metric = defaultdict(lambda: np.zeros(self.epoch))
        self.best_loss = None
        self.login = False

        self.accelerator.init_trackers(
            project_name=f"blip2_pretrain",
            config=dict(self.config),
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
        for i in range(self.epoch):
            self.training(epoch=i)
            self.evaluation(epoch=i)
        self.accelerator.end_training()
