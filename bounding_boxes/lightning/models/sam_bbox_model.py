from torch import optim, nn, utils, Tensor
import torch
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import summarize
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.amg import build_all_layer_point_grids, batched_mask_to_box, box_xyxy_to_xywh
from segment_anything.utils.transforms import ResizeLongestSide
from bounding_boxes.lightning.data.dataset import BoundingBoxDataModule
import traceback


class BoundingBoxModel(pl.LightningModule):
    def __init__(
        self,
        checkpoint,
        model_type,
        points_per_side=32,
        crop_n_layers=1,  # Not sure how to do anything other than 1... hardcoded
        crop_n_points_downscale_factor=2,
        lr=1e-3,
        use_multimask=False,
        image_size=(2048, 1280)
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.lr = lr
        self.use_multimask = use_multimask

        checkpoint_path = hf_hub_download(
            "ybelkada/segment-anything", "checkpoints/{}".format(checkpoint)
        )
        self.sam_model: Sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.sam_model)
        self.image_encoder = self.sam_model.image_encoder

        self.loss_function = nn.MSELoss()

        for param in self.sam_model.parameters():
            param.requires_grad = False

        self.point_model = nn.Linear(256*64*64, image_size[0]*image_size[1])

        print(summarize(self))

        self.image_transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def get_image_embs(self, images):
        """
        Expects images BxCxHxW
        """
        breakpoint()
        input_images = self.image_transform.apply_image_torch(images)
        input_images = input_images.permute(2, 0, 1).contiguous()[1, :, :, :]
        image_embs = self.image_encoder(input_images)
        return image_embs

    def forward(
        self,
        images,
        use_multimask=False,
    ):
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        image_embs = self.image_encoder(images)

        x = image_embs.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, height, width)
        x = torch.tanh(x)

        flat_x = x.view(batch_size, -1)
        # Find indices of 10 values with the greatest absolute values
        _, indices = torch.topk(torch.abs(flat_x), 10, dim=1)
        # Gather the 10 values using the indices
        top_values = torch.gather(flat_x, 1, indices)
        # Convert flat indices to 2D indices (x, y locations)
        points = torch.stack((indices // self.output_width, indices % self.output_width), dim=-1)
        labels = (torch.sign(top_values) + 1) / 2

        breakpoint()
        # Predict points
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=points,
            point_labels=labels,
            multimask_output=use_multimask,
        )
        
        return masks, scores, logits

    def convert_masks_to_bboxes(self, masks):
        raw_boxes = batched_mask_to_box(masks)

        if self.use_multimask:
            # Get the indices of the max scores along dimension 1
            scores, max_indices = torch.max(scores, dim=1)

            # Select the boxes using the indices of the max scores
            selected_boxes = torch.zeros((raw_boxes.shape[0], raw_boxes.shape[2]))
            for i in range(raw_boxes.shape[0]):
                selected_boxes[i, :] = raw_boxes[i, max_indices[i], :]

            raw_boxes = selected_boxes
        else:
            raw_boxes = raw_boxes.squeeze(1)
            scores = scores.squeeze(1)


        breakpoint()
        boxes = box_xyxy_to_xywh(raw_boxes)

        return boxes, scores


    def training_step(self, batch, batch_idx):
        # Image is a tensor of shape (B, 3, H, W)
        # Boxes is a list of B tensors of shape (N, 4) where N is the number of bounding boxes in the image
        # Order is "top", "left", "bottom", "right"
        images = batch['images']
        gt_bboxes = batch['bboxes']

        masks, scores, logits = self(images)
        # Input BxHxW => Output Bx4
        pred_bboxes = self.convert_masks_to_bboxes(masks)
        loss = self.loss_function(pred_bboxes, gt_bboxes)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    model = BoundingBoxModel(checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b")
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=1, 
                         max_epochs=10,)
    csv_filepath = "datasets/comcrawl/static_comcrawl_no_btn_1_dataset.csv"
    data_module = BoundingBoxDataModule(csv_file=csv_filepath, batch_size=8)
    trainer.fit(model, data_module)
