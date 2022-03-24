import pathlib
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
from matplotlib import pyplot as plt

from PolypCustomTrainer import PolypCustomTrainer
setup_logger()


# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from PolypDataLoader import get_polyp_metadata, parse_data, register_dataset

class PolypDetector:
	def __init__(self, training_dataset, testing_dataset) -> None:
		self.polyp_metadata = None
		self.cfg = get_cfg()
		self.cfg.MODEL.DEVICE = "cuda"

		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

		# minimum image size for the train set
		self.cfg.INPUT.MIN_SIZE_TRAIN = (256,)
		# maximum image size for the train set
		self.cfg.INPUT.MAX_SIZE_TRAIN = 256
		#  minimum image size for the test set
		self.cfg.INPUT.MIN_SIZE_TEST = 256
		#  maximum image size for the test set
		self.cfg.INPUT.MAX_SIZE_TEST = 512

		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
		self.cfg.DATASETS.TRAIN = (training_dataset,)
		self.cfg.DATASETS.TEST = (testing_dataset,)
		
		self.cfg.TEST.EVAL_PERIOD = 400
		
		self.cfg.DATALOADER.NUM_WORKERS = 4
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
		self.cfg.SOLVER.IMS_PER_BATCH = 4
		self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
		self.cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
		self.cfg.SOLVER.STEPS = []        # do not decay learning rate
		self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
		# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

		os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

	def train(self, training_source: str, path_prefix: str, validation_source: str, validation_path_prefix: str):

		register_dataset(self.cfg.DATASETS.TRAIN[0], training_source, path_prefix)
		register_dataset(self.cfg.DATASETS.TEST[0], validation_source, validation_path_prefix)

		#self.polyp_metadata = get_polyp_metadata(self.cfg.DATASETS.TRAIN[0], training_source, path_prefix )
		#self.cfg.DATASETS.TRAIN = ("polyp_train",)
		self.trainer = PolypCustomTrainer(self.cfg) 
		self.trainer.resume_or_load(resume=False)
		self.trainer.train()

	def infer(self, data_set: str, training_source: str, path_prefix: str,image_output_folder: str):
		# Inference should use the config with parameters that are used in training
		# cfg now already contains everything we've set previously. We changed it a little bit for inference:

		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
		predictor = DefaultPredictor(self.cfg)

		from detectron2.utils.visualizer import ColorMode

		register_dataset(data_set, training_source, path_prefix)

		polyp_metadata = get_polyp_metadata(data_set,training_source, path_prefix)
		
		dataset_dicts = parse_data(training_source, path_prefix)

		for d in dataset_dicts:    
			im = cv2.imread(d["file_name"])
			image_id = d["image_id"]
			outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
			v = Visualizer(im[:, :, ::-1],
						metadata=polyp_metadata, 
						scale=0.5, 
						instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
			)
			out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

			#cv2_imshow(out.get_image()[:, :, ::-1])
			cv2.imwrite(f'{image_output_folder}/{image_id}_{pathlib.Path(d["file_name"]).stem}.jpg', out.get_image())

			#cv2.imwrite("abc.jpg", out.get_image()[:, :, ::-1])

	def evaluate(self,training_souce:str, path_prefix: str, output_dir ,data_set):
		from detectron2.evaluation import PascalVOCDetectionEvaluator,COCOEvaluator, inference_on_dataset
		from detectron2.data import build_detection_test_loader, build_detection_train_loader

		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
		predictor = DefaultPredictor(self.cfg)

		if self.polyp_metadata is None:
			self.polyp_metadata = get_polyp_metadata(data_set,training_souce, path_prefix)

		evaluator = COCOEvaluator(data_set, output_dir=output_dir)
		val_loader = build_detection_test_loader(self.cfg, data_set,num_workers=4)
		print(inference_on_dataset(predictor.model, val_loader, evaluator))
		# another equivalent way to evaluate the model is to use `trainer.test`