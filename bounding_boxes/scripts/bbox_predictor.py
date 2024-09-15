from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from bounding_boxes.bounding_box_utils import (
    compute_bounding_boxes_webui,
    BoundingBox,
    get_csv_string_from_bounding_boxes,
    draw_bounding_boxes,
    calculate_iou,
)
import easyocr


class BoundingBoxPredictor:
    def __init__(self, model_path, cfg_path, env="vwa", caption=True, ocr_only=False):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.VAL = ("my_dataset_val",)
        self.cfg.DATASETS.TEST = ("my_dataset_test",)
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        if env == "vwa":
            CLASS_TYPE_TO_ID = {"IMG": 0, "A": 1}
        elif env == "omniact":
            CLASS_TYPE_TO_ID = {"UI_ELEMENT": 0}
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_TYPE_TO_ID.keys())
        self.cfg.OUTPUT_DIR = model_path
        self.cfg.MODEL.WEIGHTS = self.cfg.OUTPUT_DIR
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

        self.predictor = DefaultPredictor(self.cfg)
        self.ocr_reader = easyocr.Reader(["en"])
        self.env = env
        self.caption = caption
        self.ocr_only = ocr_only

    def predict(
        self,
        cv2_image,
        score_threshold=0.3,
        captioning_fn=None,
        use_all_static_text=False,
    ):
        bounding_boxes = []
        if self.ocr_only:
            ocr_results = self.ocr_reader.readtext(cv2_image)
            # Sort through static text bounding boxes
            for idx in range(len(ocr_results)):
                box, text, score = ocr_results[idx]
                if score < 0.1:
                    continue
                bounding_boxes.append(
                    BoundingBox(
                        top=box[0][1],
                        left=box[0][0],
                        right=box[2][0],
                        bottom=box[2][1],
                        interactable=False,
                        confidence=score,
                        text=text,
                        class_id=-1,
                        class_type="StaticText",
                    )
                )
            return bounding_boxes

        outputs = self.predictor(cv2_image)
        ocr_results = self.ocr_reader.readtext(cv2_image)

        ocr_used_indices = set([])
        for idx in range(0, outputs["instances"].pred_boxes.tensor.shape[0]):
            # Each row is (x1, y1, x2, y2)
            bbox = outputs["instances"].pred_boxes.tensor[idx].cpu().numpy().astype(int)
            score = outputs["instances"].scores[idx].cpu().numpy()
            if score < score_threshold:
                continue
            pred_class_id = int(outputs["instances"].pred_classes[idx].cpu().numpy())
            if self.env == "vwa":
                CLASS_ID_TO_TYPE = {0: "IMG", 1: "A"}
            elif self.env == "omniact":
                CLASS_ID_TO_TYPE = {0: "UI_ELEMENT"}
            pred_class_type = CLASS_ID_TO_TYPE[pred_class_id]

            if self.env == "vwa":
                if pred_class_type == "IMG" and captioning_fn is not None:
                    cropped_image = cv2_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    text = ", description: {}".format(
                        captioning_fn([cropped_image])[0].strip()
                    )
                else:
                    pred_bbox = BoundingBox(
                        top=bbox[1],
                        left=bbox[0],
                        bottom=bbox[3],
                        right=bbox[2],
                    )
                    text, used_indices = self.get_ocr_text_for_bounding_box(
                        pred_bbox, ocr_results
                    )
                    ocr_used_indices.update(used_indices)
            elif self.env == "omniact":
                if self.caption:
                    cropped_image = cv2_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                    caption_text = ", description: {}".format(
                        captioning_fn([cropped_image])[0].strip()
                    )
                else:
                    caption_text = ""
                # text = caption_text
                pred_bbox = BoundingBox(
                    top=bbox[1],
                    left=bbox[0],
                    bottom=bbox[3],
                    right=bbox[2],
                )
                ocr_text, used_indices = self.get_ocr_text_for_bounding_box(
                    pred_bbox, ocr_results
                )
                ocr_used_indices.update(used_indices)

                text = "{}{}".format(ocr_text, caption_text)

            bounding_boxes.append(
                BoundingBox(
                    top=bbox[1],
                    left=bbox[0],
                    bottom=bbox[3],
                    right=bbox[2],
                    interactable=True,
                    confidence=score,
                    text=text,
                    class_id=pred_class_id,
                    class_type=pred_class_type,
                )
            )

        bounding_boxes = self.merge_overlapping_boxes(bounding_boxes)

        # Sort through static text bounding boxes
        for idx in range(len(ocr_results)):
            if idx in ocr_used_indices and not use_all_static_text:
                continue
            box, text, score = ocr_results[idx]
            if score < 0.1:
                continue
            bounding_boxes.append(
                BoundingBox(
                    top=box[0][1],
                    left=box[0][0],
                    right=box[2][0],
                    bottom=box[2][1],
                    interactable=False,
                    confidence=score,
                    text=text,
                    class_id=-1,
                    class_type="StaticText",
                )
            )

        # sort bounding boxes by top left
        # bounding_boxes = sorted(
        #     bounding_boxes, key=lambda box: (box.top**2 + box.left**2) ** 0.5
        # )
        # bounding_boxes.sort(key=lambda x: (x.top, x.left))

        return bounding_boxes

    def get_csv_string(
        self,
        image,
        score_threshold=0.3,
        captioning_fn=None,
        use_all_static_text=False,
    ):
        bounding_boxes = self.predict(
            image,
            score_threshold,
            captioning_fn,
            use_all_static_text,
        )
        return get_csv_string_from_bounding_boxes(bounding_boxes)

    def get_ocr_text_for_bounding_box(self, pred_bbox, ocr_results):
        # Filter OCR bounding boxes based on IoU and concatenate their texts
        filtered_texts = []
        used_indices = set([])
        for idx in range(len(ocr_results)):
            box, text, score = ocr_results[idx]
            if score < 0.1:
                continue
            ocr_bbox = BoundingBox(
                top=box[0][1],
                left=box[0][0],
                right=box[2][0],
                bottom=box[2][1],
            )

            # if self.is_bbox_within_with_margin(ocr_bbox, pred_bbox, margin=25) or self.is_bbox_within_with_margin(pred_bbox, ocr_bbox, margin=25):
            if self.is_bbox_within_with_margin(ocr_bbox, pred_bbox, margin=5):
                filtered_texts.append(
                    (box[0][0], text)
                )  # Use left coordinate for sorting
                used_indices.add(idx)

        # Sort filtered OCR texts by horizontal position (left coordinate)
        filtered_texts.sort(key=lambda x: x[0])

        # Concatenate filtered texts
        concatenated_text = "".join(text for _, text in filtered_texts)

        return concatenated_text, used_indices

    def is_bbox_within_with_margin(self, inner_bbox, outer_bbox, margin=0):
        """
        Check if inner_bbox is within outer_bbox with an optional margin.
        """
        return (
            inner_bbox.left >= outer_bbox.left - margin
            and inner_bbox.top >= outer_bbox.top - margin
            and inner_bbox.right <= outer_bbox.right + margin
            and inner_bbox.bottom <= outer_bbox.bottom + margin
        )

    def merge_into_smaller_box(self, box1, box2):
        # Determine the smaller box based on area
        area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        area2 = (box2.right - box2.left) * (box2.bottom - box2.top)

        return box1 if area1 < area2 else box2

    def merge_overlapping_boxes(self, boxes):
        merged_boxes = []
        skip_indices = set()

        for i in range(len(boxes)):
            if i in skip_indices:
                continue
            for j in range(i + 1, len(boxes)):
                if j in skip_indices:
                    continue

                iou = calculate_iou(boxes[i], boxes[j])
                if iou > 0.5:
                    # Merge boxes into the smaller one and mark the other for skipping
                    merged_box = self.merge_into_smaller_box(boxes[i], boxes[j])
                    boxes[i] = merged_box  # Update the box at i to be the merged box
                    skip_indices.add(j)  # Skip processing of box j in the future

        # Add non-skipped boxes to the result
        for i, box in enumerate(boxes):
            if i not in skip_indices:
                merged_boxes.append(box)

        return merged_boxes
