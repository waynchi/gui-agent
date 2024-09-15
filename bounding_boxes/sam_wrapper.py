from huggingface_hub import hf_hub_download
from bounding_boxes.bounding_box_utils import (
    BoundingBox,
    get_csv_string_from_bounding_boxes,
)
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.amg import batched_mask_to_box, box_xyxy_to_xywh


class SegmentAnythingWrapper:
    def __init__(self, checkpoint, model_type, device="cuda"):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device

        checkpoint_path = hf_hub_download(
            "ybelkada/segment-anything", "checkpoints/{}".format(checkpoint)
        )
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=device)

        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=5,  # Requires open-cv to run post-processing
        )

    def get_bounding_boxes(self, image, points, labels, use_multimask=False):
        points = points.to(self.device)
        points = self.predictor.transform.apply_coords_torch(points, image.shape[:2])
        self.predictor.set_image(image)

        try:
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=points,
                point_labels=labels,
                multimask_output=use_multimask,
            )
        except torch.cuda.OutOfMemoryError:
            print("OUT OF MEMORY ERROR. Defaulting to single mask")
            use_multimask = False
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=points,
                point_labels=labels,
                multimask_output=use_multimask,
            )
        

        raw_boxes = batched_mask_to_box(masks)

        if use_multimask:
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

        boxes = []
        try:
            for box in raw_boxes:
                box = box.cpu().numpy()
                box = box_xyxy_to_xywh(box).tolist()
                boxes.append(box)
        except Exception as e:
            import traceback
            print(e)
            print(traceback.format_exc())
            breakpoint()

        try:
            bounding_boxes = [
                BoundingBox(
                    top=boxes[i][1],
                    left=boxes[i][0],
                    bottom=boxes[i][1] + boxes[i][3],
                    right=boxes[i][0] + boxes[i][2],
                    class_id=0,
                    confidence=scores[i].item(),
                    interactable=True,
                )
                for i in range(len(boxes))
            ]
        except Exception as e:
            import traceback
            print(e)
            print(traceback.format_exc())
            breakpoint()

        return bounding_boxes

    def get_all_bounding_boxes(self, image):
        masks = self.mask_generator.generate(image)

        # bbox is in XYWH format
        bounding_boxes = [
            BoundingBox(
                top=mask["bbox"][1],
                left=mask["bbox"][0],
                bottom=mask["bbox"][1] + mask["bbox"][3],
                right=mask["bbox"][0] + mask["bbox"][2],
                class_id=0,
                confidence=mask["predicted_iou"],
                interactable=True,
            )
            for mask in masks
        ]

        return bounding_boxes

    def get_csv_string_from_image(self, image):
        bounding_boxes = self.get_all_bounding_boxes(image)
        return get_csv_string_from_bounding_boxes(bounding_boxes)


if __name__ == "__main__":
    import cv2
    from PIL import Image
    from bounding_box_utils import draw_bounding_boxes

    # checkpoint = "sam_vit_h_4b8939.pth"
    checkpoint = "sam_vit_b_01ec64.pth"
    # model_type = "vit_h"
    model_type = "vit_b"

    cv2_image = cv2.imread("example_images/screenshot.png")
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(cv2_image)

    sam_wrapper = SegmentAnythingWrapper(checkpoint, model_type)
    bounding_boxes = sam_wrapper.get_all_bounding_boxes(cv2_image)

    csv_string = get_csv_string_from_bounding_boxes(bounding_boxes)
    bbox_image, _, _, _ = draw_bounding_boxes(csv_string, pil_image, 1)
    bbox_image.save("sam_output.png")
