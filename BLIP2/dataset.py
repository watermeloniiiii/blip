import os
import json
from enum import Enum
from PIL import Image
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Any

from torch.utils.data import Dataset


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 10000,
            _Split.VAL: 100,
            _Split.TEST: 1,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(
        self, actual_index: int, class_id: Optional[str] = None
    ) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class coco_karpathy(Dataset):
    Split = Union[_Split]

    def __init__(
        self, root, split: "coco_karpathy.Split", transforms=None, processor=None
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        with open(
            os.path.join(self.root, "coco_karpathy_" + split.value + ".json"), "r"
        ) as file:
            self.file_lst = json.load(file)
        if transforms:
            self.transforms = transforms
        if processor:
            self.processor = processor
        self._split = split

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        caption = self.file_lst[index]["caption"]
        image_dir = self.file_lst[index]["image"]
        if self._split.value == "train":
            image_id = self.file_lst[index]["image_id"]
        image_dir = os.path.join(self.root, image_dir)
        img = Image.open(image_dir).convert("RGB")
        encoding = self.processor(
            img,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["caption"] = caption
        return encoding


dataset = coco_karpathy(
    "/media/workspace/linchenxi/projects/blip", coco_karpathy.Split["VAL"]
)
