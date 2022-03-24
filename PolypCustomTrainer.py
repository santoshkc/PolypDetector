import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper

from detectron2.evaluation import COCOEvaluator

from LossEvalHook import LossEvalHook

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
                num_workers=4
            )
        ))
        return hooks          