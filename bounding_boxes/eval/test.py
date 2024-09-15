import numpy as np
from mean_average_precision import MetricBuilder

# [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
gt = np.array(
    [
        [439, 157, 556, 241, 0, 0, 0],
        [437, 246, 518, 351, 0, 0, 0],
        [515, 306, 595, 375, 0, 0, 0],
        [407, 386, 531, 476, 0, 0, 0],
        [544, 419, 621, 476, 0, 0, 0],
        [609, 297, 636, 392, 0, 0, 0],
    ]
)

# [xmin, ymin, xmax, ymax, class_id, confidence]
preds = np.array(
    [
        [439, 157, 556, 241, 0, 0],
        [437, 246, 518, 351, 0, 0],
        [515, 306, 595, 375, 0, 0],
        [407, 386, 531, 476, 0, 0],
        [544, 419, 621, 476, 0, 0],
        [609, 297, 636, 392, 0, 0],
    ]
)

# print list of available metrics
print(MetricBuilder.get_metrics_list())

# create metric_fn
metric_fn = MetricBuilder.build_evaluation_metric(
    "map_2d", async_mode=True, num_classes=1
)

# add some samples to evaluation
# for i in range(10):
metric_fn.add(preds, gt)

# compute PASCAL VOC metric
result = metric_fn.value(iou_thresholds=0.5)
breakpoint()
print(
    f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}"
)

# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

# compute metric COCO metric
print(
    f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}"
)
