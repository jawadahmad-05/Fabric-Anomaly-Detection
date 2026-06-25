import json
import math
import os
import numpy as np
import cv2

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from .transforms import AddGaussianNoise
from .info import CLASS_NAMES, DATA_PATH, DOMAINS


class TextAndImageDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                self.meta.append(json.loads(line))

        self.transforms_list = [
            T.RandomApply(
                [T.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            T.RandomApply(
                [T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]

        transform_x = [
            AddGaussianNoise(std=1, p=0.7),
            T.RandomApply([T.ColorJitter(brightness=0.5)], p=0.7),
            T.RandomApply([T.ColorJitter(contrast=0.5)], p=0.7),
            T.RandomApply([T.ColorJitter(saturation=0.5)], p=0.7)
        ]
        self.transform_x = T.Compose(
            transform_x
            + [
                T.Resize((img_size, img_size), InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = T.Compose(
            [
                T.Resize((img_size, img_size), InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        data_path = self.data_path
        img_path = os.path.join(data_path, meta["image_path"])
        img = Image.open(img_path).convert("RGB")

        img = self.transform_x(img)
        if meta["label"] and meta["mask_path"]:
            mask_path = os.path.join(data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        random_transform = T.Compose(self.transforms_list)
        transform_tensor = torch.cat([img, mask], dim=0)
        assert transform_tensor.shape[0] == 4
        transform_tensor = random_transform(transform_tensor)
        img = transform_tensor[0:3, :, :]
        mask = transform_tensor[3:4, :, :]

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            meta_path: str,
            img_size: int,
            class_name: str
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        self.meta = []
        with open(meta_path, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                if m["class_name"] == class_name:
                    self.meta.append(m)

        # Define transforms
        self.transform_x = T.Compose(
            [
                T.Resize((img_size, img_size), Image.BICUBIC),
                T.ToTensor(),
                T.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = T.Compose(
            [
                T.Resize((img_size, img_size), Image.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        img_path = os.path.join(self.data_path, meta["image_path"])
        img_pil = Image.open(img_path).convert("RGB")

        # Apply Global Normalization & CLAHE if it's AITEX or Chenab
        if "AITEX" in self.data_path or "Chenab" in self.data_path:
            img_np = np.array(img_pil).astype(np.float32)
            # Min-Max Normalization to remove lighting gradients
            for i in range(3):
                img_np[:,:,i] = (img_np[:,:,i] - img_np[:,:,i].min()) / (img_np[:,:,i].max() - img_np[:,:,i].min() + 1e-5) * 255
            img_np = img_np.astype(np.uint8)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # Apply CLAHE to each channel
            for i in range(3):
                img_np[:,:,i] = clahe.apply(img_np[:,:,i])
            img_pil = Image.fromarray(img_np)

        img = self.transform_x(img_pil)
        if meta["label"] and meta["mask_path"]:
            mask_path = os.path.join(self.data_path, meta["mask_path"])
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_text_and_image_dataset(
        dataset_name: str,
        img_size: int,
        stage: str = "train"
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )
    if stage == "train":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + ".jsonl"
        )
        data_path = DATA_PATH[dataset_name.split("-")[0]]
        dataset = TextAndImageDataset(data_path, meta_path, img_size)
        return dataset
    elif stage == "test":
        meta_path = os.path.join(
            "./dataset/hub", dataset_name + "_test.jsonl"
        )
        if not os.path.exists(meta_path):
            meta_path = os.path.join(
                "./dataset/hub", dataset_name + ".jsonl"
            )
        datasets = {}
        for class_name in CLASS_NAMES[dataset_name]:
            if class_name == "Normal":
                continue
            
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name
            )
            
            # Special handling for Chenab: Add Normal images to every defect class for AUC
            if "Chenab" in dataset_name:
                normal_dataset = BaseSingleClassDataset(
                    data_path=DATA_PATH[dataset_name],
                    meta_path=meta_path,
                    img_size=img_size,
                    class_name="Normal",
                )
                image_dataset.meta.extend(normal_dataset.meta)
                
            # Special handling for DTD_Woven: Add woven images as normal reference
            if "DTD_Woven" in dataset_name:
                normal_dataset = BaseSingleClassDataset(
                    data_path=DATA_PATH[dataset_name],
                    meta_path=meta_path,
                    img_size=img_size,
                    class_name="woven",
                )
                image_dataset.meta.extend(normal_dataset.meta)
                
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
