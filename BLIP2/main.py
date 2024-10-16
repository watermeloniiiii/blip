import os
import requests
from PIL import Image
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from BLIP2.dataset import coco_karpathy
from omegaconf import OmegaConf
from pathlib import Path
import utils
from trainer_deepspeed import Trainer


def main(args, config):
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",
        gradient_accumulation_steps=1,
        project_dir="blip2_pretrain",
    )
    train_data = coco_karpathy(config.PATH.data_root, coco_karpathy.Split["TRAIN"])
    vali_data = coco_karpathy(config.PATH.data_root, coco_karpathy.Split["VAL"])
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_loader = DataLoader(
        train_data, batch_size=config.MODEL.optimization.batch_size, drop_last=True
    )

    vali_loader = DataLoader(
        vali_data, batch_size=config.MODEL.optimization.batch_size, drop_last=True
    )

    blip2_trainer = Trainer(model)
    blip2_trainer.train_model(
        epoch=config.max_epoch,
        train_loader=train_loader,
        vali_loader=vali_loader,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    import argparse
    print ("hahah")
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip_config", default="./BLIP2/pretrain.yaml")
    args = parser.parse_args()

    # config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config = OmegaConf.load(args.blip_config)

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
