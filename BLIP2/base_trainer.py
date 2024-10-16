from accelerate.utils import DummyOptim, DummyScheduler
import math
import os
import torch.optim as optim

class Trainer(object):
    def __init__(self, net, config):
        self.net = net
        self.config = config

    def _select_optimizer(self):
        """
        initialize an optimizer from either the definition of deepspeed optimizer or user-defined optimizer
        NOTE that the user-defined optimizer would be prioritized
        """
        user_defined_optimizer = self.config.MODEL.optimization.optimizer is not None
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
            if self.config.MODEL.optimization.optimizer == "Adam":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=self.config.MODEL.optimization.base_lr,
                    weight_decay=self.config.MODEL.optimization.weight_decay,
                )
            elif self.config.MODEL.optimization.optimizer == "SGD":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=self.config.MODEL.optimization.base_lr,
                    weight_decay=self.config.MODEL.optimization.optimizerweight_decay,
                    momentum=self.config.MODEL.optimization.momentum,
                )
            elif self.config.MODEL.optimization.optimizer == "AdamW":
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, self.net.parameters()),
                    lr=self.config.MODEL.optimization.base_lr,
                    weight_decay=self.config.MODEL.optimization.weight_decay,
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
        model_folder = self.config.PATH.model_outdir
        model_path = os.path.join(model_folder, self.config.MODEL_INFO.model_name)
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
        user_defined_scheduler = self.config.MODEL.optimization.scheduler is not None
        deepspeed_defined_scheduler = not (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        assert (
            user_defined_scheduler or deepspeed_defined_scheduler
        ), "Please provide at least one scheduler from either deepspeed-defined or user-defined"
        if user_defined_scheduler:
            raise NotImplementedError
        if not user_defined_scheduler and deepspeed_defined_scheduler:
            # remember to revise step-relevant parameters
            deepspeed_config = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if "warmup_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = (
                    math.ceil(
                        len(self.train_loader)
                        * self.epoch
                        * self.config.MODEL.optimization.warmup_steps_ratio
                    )
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            if "total_num_steps" in deepspeed_config["scheduler"]["params"]:
                deepspeed_config["scheduler"]["params"]["total_num_steps"] = (
                    math.ceil(
                        len(self.train_loader)
                        * self.epoch
                        * self.config.MODEL.optimization.total_steps_ratio
                    )
                    // self.accelerator.gradient_accumulation_steps
                    // self.num_processes
                )
            return DummyScheduler(self.optimizer)

    def training(self, epoch):
        raise NotImplementedError