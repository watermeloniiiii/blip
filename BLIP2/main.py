from PIL import Image
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from BLIP2.dataset import coco_karpathy
from omegaconf import OmegaConf
from pathlib import Path
import utils
from trainer_deepspeed import blip2_trainer

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment
from BLIP2.setup import default_setup


def collate_fn(data):
    keys = data[0].keys()
    for b in range(len(data)):
        data[b]["input_ids"] = data[b]["input_ids"][:5]
        data[b]["attention_mask"] = data[b]["attention_mask"][:5]
    batch = {}
    caption = []
    for key in keys:
        if key != "caption":
            batch[key] = torch.stack([d[key] for d in data], dim=0)
    batch["caption"] = [b["caption"] for b in data]
    return batch


def define_transforms(config):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.MODEL.optimization.image_size,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                (
                    config.MODEL.optimization.image_size,
                    config.MODEL.optimization.image_size,
                ),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform_train, transform_test


def main(args, config):
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.qformer.parameters():
        param.requires_grad = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb",
        gradient_accumulation_steps=1,
        project_dir="blip2_pretrain",
    )
    transform_train, _ = define_transforms(config)
    train_data = coco_karpathy(
        config.PATH.data_root,
        coco_karpathy.Split["TRAIN"],
        transforms=transform_train,
        processor=processor,
    )
    vali_data = coco_karpathy(
        config.PATH.data_root,
        coco_karpathy.Split["VAL"],
        transforms=transform_train,
        processor=processor,
    )
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_loader = DataLoader(
        train_data,
        batch_size=config.MODEL.optimization.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(
                0, len(train_data), (config.MODEL.optimization.num_train_samples,)
            )
        ),
    )

    vali_loader = DataLoader(
        vali_data,
        batch_size=config.MODEL.optimization.batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(
            torch.randint(
                0, len(vali_data), (config.MODEL.optimization.num_vali_samples,)
            )
        ),
        collate_fn=collate_fn,
    )

    trainer = blip2_trainer(model, config, processor)
    trainer.train_model(
        train_loader=train_loader,
        vali_loader=vali_loader,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--blip_config", default="./BLIP2/pretrain.yaml")
    args = parser.parse_args()

    # config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config = default_setup(args.blip_config)
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))
    main(args, config)
