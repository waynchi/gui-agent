from bounding_boxes.scripts.bbox_predictor import BoundingBoxPredictor
from PIL import Image


bbox_predictor = BoundingBoxPredictor(
    model_path="/home/waynechi/dev/gui-agent/bounding_boxes/outputs/vwa_pl_100k_classes_9/model_final.pth",
    cfg_path="/home/waynechi/dev/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
)
image = Image.open()
csv_content = bbox_predictor.get_csv_string(
    image, score_threshold=0.3, captioning_fn=None
)
