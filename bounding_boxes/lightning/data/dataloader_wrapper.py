from torch.utils.data import DataLoader, IterableDataset
from detectron2.data import MetadataCatalog


class DataloaderWrapper(IterableDataset):
    def __init__(self, detectron2_dataloader, dataset_name):
        self.detectron2_dataloader = detectron2_dataloader
        self.dataset_len = MetadataCatalog.get(dataset_name).get("num_images")

    def __iter__(self):
        return iter(self.detectron2_dataloader)

    def __len__(self):
        return self.dataset_len
