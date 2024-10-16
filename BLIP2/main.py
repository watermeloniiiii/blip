import os
import requests
from PIL import Image
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from data import create_dataset, create_sampler, create_loader
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
    datasets = [create_dataset("pretrain", config, min_scale=0.2)]
    print("number of training samples: %d" % len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)

    data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config.batch_size],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    blip2_trainer = Trainer(model)
    blip2_trainer.train_model(epoch=config.max_epoch, train_loader=data_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blip_config", default="./configs/pretrain.yaml")
    args = parser.parse_args()

    # config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config = OmegaConf.load(args.blip_config)

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
