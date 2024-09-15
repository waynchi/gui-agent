import os
import logging
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List

import logging
import torch
import pytorch_lightning as pl
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

import detectron2.utils.comm as comm
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning.tuner.tuning import Tuner
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    default_writers,
    DefaultTrainer,
    hooks,
    SimpleTrainer,
)
from detectron2.evaluation import print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver.build import (
    build_lr_scheduler,
    build_optimizer,
    maybe_add_gradient_clipping,
    get_default_optimizer_params,
)
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from torchinfo import summary
from detectron2.config import CfgNode
from detectron2.utils.env import TORCH_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/waynechi/dev/gui-agent/bounding_boxes")
from bounding_boxes.lightning.data.detectron_dataset import Detectron2DataModule


def build_adam_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )
    adam_args = {
        "params": params,
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }
    if TORCH_VERSION >= (1, 12):
        adam_args["foreach"] = True
    return maybe_add_gradient_clipping(cfg, torch.optim.AdamW(**adam_args))


def save_bounding_box_images(batch, outputs, output_dir, epoch, device_id):
    """
    Saves images with ground truth and predicted bounding boxes.

    Args:
    - batch: The input batch containing images and ground truth annotations.
    - outputs: The output from the model, containing the predictions.
    - output_dir: Directory to save the output images.
    """
    # Assuming batch[0] and outputs[0] contains the data and prediction for the first sample
    image_tensor = batch[0]["image"]

    # Convert image tensor to numpy for visualization (assuming standard normalization)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = image_np[:, :, [2, 1, 0]]

    # Setup visualizer for predictions (assuming predictions are stored like this)
    pred_instances = outputs[0]["instances"]
    v_pred = Visualizer(
        image_np[:, :, ::-1],
        metadata=MetadataCatalog.get("my_dataset_train"),
        scale=1.2,
    )
    out_pred = v_pred.draw_instance_predictions(pred_instances.to("cpu"))

    # Ensure unique filename for predictions image
    os.makedirs(os.path.join(output_dir, f"epoch_{epoch}"), exist_ok=True)
    base_filename_pred = f"{output_dir}/epoch_{epoch}/predictions{device_id}"
    counter = 0
    pred_filename = f"{base_filename_pred}.png"
    while os.path.exists(pred_filename):
        pred_filename = f"{base_filename_pred}_{counter}.png"
        counter += 1

    # Save the predicted image
    plt.figure(figsize=(10, 10))
    plt.imshow(out_pred.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.savefig(pred_filename, bbox_inches="tight")
    plt.close()


class FasterRCNNLightningModule(pl.LightningModule):
    def __init__(self, cfg, lr=0.0025, freeze_backbone=False, optimizer_name="sgd"):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        self.lr = lr
        cfg.SOLVER.BASE_LR = lr
        cfg.DATASETS.TRAIN = ("my_dataset_train",)
        cfg.DATASETS.VAL = ("my_dataset_val",)
        cfg.DATASETS.TEST = "my_dataset_test"  # Empty if no validation/test dataset

        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)
        if freeze_backbone:
            for param in self.model.backbone.bottom_up.parameters():
                param.requires_grad = False
        summary(self.model)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.optimizer_name = optimizer_name

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["iteration"] = self.storage.iter

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        self.start_iter = checkpointed_state["iteration"]
        self.storage.iter = self.start_iter

    def setup(self, stage: str):
        if self.cfg.MODEL.WEIGHTS:
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                self.cfg.OUTPUT_DIR,
            )
            logger.info(
                f"Load model weights from checkpoint: {self.cfg.MODEL.WEIGHTS}."
            )
            # Only load weights, use lightning checkpointing if you want to resume
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
                if comm.is_main_process()
                else {}
            )
        else:
            self.storage.__enter__()

        loss_dict = self.model(batch)
        SimpleTrainer.write_metrics(
            loss_dict,
            data_time,
            # batch_idx,
            (self.current_epoch * self.trainer.limit_train_batches) + batch_idx,
        )

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr",
            opt.param_groups[self._best_param_group_id]["lr"],
            smoothing_hint=False,
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 20 == 0:
            for writer in self.writers:
                writer.write()

        return sum(loss_dict.values())

    def training_step_end(self, training_step_outputs):
        self.data_start = time.perf_counter()
        return training_step_outputs

    def on_train_epoch_end(self):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
        for writer in self.writers:
            writer.write()
            # writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self) -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.cfg.DATASETS.VAL):
            results[dataset_name] = self._evaluators[idx].evaluate()
            if comm.is_main_process():
                print_csv_format(results[dataset_name])

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def _reset_dataset_evaluators(self):
        self._evaluators = []
        for dataset_name in self.cfg.DATASETS.VAL:
            evaluator = COCOEvaluator(dataset_name=dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self):
        self._reset_dataset_evaluators()

    def on_validation_epoch_end(self):
        results = self._process_dataset_evaluation_results()

        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        self.storage.put_scalars(**flattened_results, smoothing_hint=False)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]

        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
                if comm.is_main_process()
                else {}
            )

        outputs = self.model(batch)

        with torch.no_grad():
            save_bounding_box_images(
                batch,
                outputs,
                self.cfg.OUTPUT_DIR,
                self.current_epoch,
                self.device.index,
            )
        self._evaluators[dataloader_idx].process(batch, outputs)

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = build_adam_optimizer(self.cfg, self.model)
        else:
            # Default to SGD
            optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_lr", type=float, default=0.00025)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    # parser.add_argument("--max_epochs", type=int, default=10)
    # parser.add_argument("--image_batches_per_epoch", type=int, default=3331)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="None")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/waynechi/dev/gui-agent/bounding_boxes/cfg/faster_rcnn.yaml",
    )
    parser.add_argument("--optimizer", type=str, default="sgd")

    args = parser.parse_args()

    # Initialize W&B logger
    # wandb_logger = WandbLogger(name="FasterRCNN_Training", project="ObjectDetection")

    cfg = get_cfg()
    config_path = args.config_path
    cfg.merge_from_file(config_path)
    # cfg.SOLVER.MAX_ITER = args.max_epochs * args.image_batches_per_epoch
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.OUTPUT_DIR = args.output_dir
    cfg.IMS_PER_BATCH = args.batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open("{}/config.yaml".format(cfg.OUTPUT_DIR), "w") as f:
        f.write(cfg.dump())

    # Example setup
    # dataset_path = '/home/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_2/static_comcrawl_no_btn_1_coco_{}.json'
    dataset_path = args.dataset_path
    json_files = {
        "train": dataset_path.format("_train"),
        "val": dataset_path.format("_val"),
        "test": dataset_path.format("_test"),
        # "all": dataset_path.format(""),
    }

    # img_path = "/home/waynechi/dev/gui-agent/bounding_boxes/datasets/comcrawl/comcrawl_no_btn_1/images"
    img_path = args.img_path
    image_roots = {
        "train": img_path,
        "val": img_path,
        "test": img_path,
        # "all": img_path,
    }

    data_module = Detectron2DataModule(
        cfg,
        json_files=json_files,
        image_roots=image_roots,
        batch_size=cfg.IMS_PER_BATCH,
    )

    model = FasterRCNNLightningModule(
        cfg,
        lr=args.base_lr,
        freeze_backbone=args.freeze_backbone,
        optimizer_name=args.optimizer,
    )
    max_epochs = args.epochs
    if args.strategy == "ddp":
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            max_epochs=max_epochs,
            # num_sanity_val_steps=2,
            limit_train_batches=int(args.max_iters / max_epochs),
            num_nodes=args.num_nodes,
            strategy="ddp",
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.devices,
            max_epochs=max_epochs,
            # num_sanity_val_steps=2,
            limit_train_batches=int(args.max_iters / max_epochs),
        )

    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model)
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # breakpoint()
    # fig.savefig(f"lr_finder.png")

    trainer.fit(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)
