import csv
import os
from collections import namedtuple
from typing import Optional, Tuple, Any, Union, List, Callable

import PIL
import torch
import torchvision
from torchvision.transforms import ToTensor

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebAHQ(torchvision.datasets.VisionDataset):
    def __init__(self, root,
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        attr = self._load_csv("CelebAMask-HQ-attribute-anno.txt", header=1)
        self.filename = attr.index
        self.attr = attr.data
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header



    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, "CelebA-HQ-img-orig", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            # elif t == "identity":
            #     target.append(self.identity[index, 0])
            # elif t == "bbox":
            #     target.append(self.bbox[index, :])
            # elif t == "landmarks":
            #     target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

dataset = CelebAHQ(root="/media/disk3/CelebAMaskHQ/CelebAMask-HQ/", transform=ToTensor())
for i in range(10):
    X, target = dataset[i]
    print(X.shape, target)