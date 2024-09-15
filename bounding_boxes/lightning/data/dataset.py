import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from bounding_boxes.lightning.scripts.utils import resize_and_pad


class BoundingBoxDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["environment"].str.contains("reddit", na=False)]
        # Group by image path and aggregate bounding boxes
        # self.image_groups = self.df.groupby("image")
        # self.images = list(self.image_groups.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]["image"]
        bbox = self.df.iloc[idx][["top", "left", "bottom", "right"]].values
        # Convert bbox to float
        bbox = bbox.astype("float")

        # Get all rows for this image
        # img_path = self.images[idx]
        # boxes_data = self.image_groups.get_group(img_path)
        # boxes = boxes_data[["top", "left", "bottom", "right"]].values
        # boxes = boxes.astype("float")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = resize_and_pad(image, output_size=(1024, 1024))

        sample = {"image": image, "bbox": bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors, including images and bounding boxes."""

    def __call__(self, sample):
        image, bbox = sample["image"], sample["bbox"]
        image = transforms.ToTensor()(image)
        bbox = torch.from_numpy(bbox).to(dtype=torch.float)
        return {"image": image, "bbox": bbox}


# Assuming you are using the BoundingBoxDataModule from the previous example
class BoundingBoxDataModule(pl.LightningDataModule):
    def __init__(
        self, csv_file, batch_size=32, transform=transforms.Compose([ToTensor()])
    ):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = BoundingBoxDataset(
            csv_file=self.csv_file, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        bboxes = torch.stack([item["bbox"] for item in batch])
        # center_points = [item["center_point"] for item in batch]
        return {
            "images": images,
            "bboxes": bboxes,
            # center_points: center_points,
        }


if __name__ == "__main__":
    # Example usage
    csv_file_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/static_comcrawl_no_btn_1_dataset.csv"
    data_module = BoundingBoxDataModule(csv_file=csv_file_path)
    data_module.setup()
    data_module.dataset[0]
    # Get one batch
    for batch in data_module.train_dataloader():
        breakpoint()
        print(batch)
        break

    print("tested")
