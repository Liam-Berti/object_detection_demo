import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset


class Imagenette(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=None,
        target_transform=None,
        train=True,
        label_noise=0,
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.valid = train

        self.img_labels = self.img_labels[self.img_labels["is_valid"] != train]

        # 0 = 0% noise, 1 = 1% noise, 2 = 5% noise,
        # 3 = 25% noise, 4 = 50% noise
        if label_noise < 0:
            label_noise = 0
        elif label_noise > 4:
            label_noise = 4
        self.label_noise = label_noise + 1

        self.classes = {
            label: i
            for i, label in enumerate(
                sorted(self.img_labels.iloc[:, self.label_noise].unique())
            )
        }

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, self.label_noise]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = self.classes[label]
        return image, label
