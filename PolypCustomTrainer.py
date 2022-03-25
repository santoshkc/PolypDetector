import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper

from detectron2.evaluation import COCOEvaluator

from LossEvalHook import LossEvalHook
from TensorBoardLossWriter import CustomTensorboardXWriter

from detectron2.utils.events import CommonMetricPrinter, JSONWriter

class PolypCustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True),
                num_workers=4,
                batch_size=4
            )
        ))
        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks
                     
    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1,LossEvalHook(
    #         self.cfg.TEST.EVAL_PERIOD,
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             DatasetMapper(self.cfg,True),
    #             num_workers=4
    #         )
    #     ))
    #     return hooks    
    
    # def build_writers(self):
    #     """
    #     Overwrites the default writers to contain our custom tensorboard writer

    #     Returns:
    #         list[EventWriter]: a list of :class:`EventWriter` objects.
    #     """
    #     return [
    #         CommonMetricPrinter(self.max_iter),
    #         JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
    #         CustomTensorboardXWriter(self.cfg.OUTPUT_DIR),
    #     ]      