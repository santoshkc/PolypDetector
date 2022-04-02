import torch
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
from detectron2.utils.visualizer import Visualizer,_create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from detectron2.data import detection_utils as utils

from PolypDataLoader import get_polyp_metadata, parse_data, register_dataset

class PolypDetector:
	def __init__(self, training_dataset, testing_dataset,
			default_output_dir,
			checkpoint_per_epoch:int,
			validation_per_epoch:int,

			min_train_image_size: int = 512,
			max_train_image_size: int = 512,
			min_test_image_size: int = 512,
			max_test_image_size: int = 512,
			total_epoch = 5,
			image_count_per_batch = 32,
			total_image_samples: int = 0) -> None:
		
		
		self.polyp_metadata = None
		self.cfg = get_cfg()
		self.cfg.MODEL.DEVICE = "cuda"

		self.cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
		self.cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]

		#Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
		self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.6]
		

		# minimum image size for the train set
		self.cfg.INPUT.MIN_SIZE_TRAIN = (min_train_image_size,)
		# maximum image size for the train set
		self.cfg.INPUT.MAX_SIZE_TRAIN = max_train_image_size
		#  minimum image size for the test set
		self.cfg.INPUT.MIN_SIZE_TEST = min_test_image_size
		#  maximum image size for the test set
		self.cfg.INPUT.MAX_SIZE_TEST = max_test_image_size

		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
		self.cfg.DATASETS.TRAIN = (training_dataset,)
		self.cfg.DATASETS.TEST = (testing_dataset,)

		self.cfg.SOLVER.IMS_PER_BATCH = image_count_per_batch
		num_epochs = total_epoch

		total_images = total_image_samples
		one_epoch = int(total_images / self.cfg.SOLVER.IMS_PER_BATCH)
		max_iter = one_epoch * num_epochs

		self.cfg.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset

		# Save a checkpoint after every this number of iterations
		# run validation every x steps		
		self.cfg.SOLVER.CHECKPOINT_PERIOD = int(one_epoch/checkpoint_per_epoch)
		self.cfg.TEST.EVAL_PERIOD = int(one_epoch/validation_per_epoch)
		
		self.cfg.DATALOADER.NUM_WORKERS = 4
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
		self.cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
		self.cfg.SOLVER.STEPS = []        # do not decay learning rate
		self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
		# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

		# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
		#self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

		self.cfg.OUTPUT_DIR = default_output_dir
		
		os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

		# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
		# balance obtaining high recall with not having too many low precision
		# detections that will slow down inference post processing steps (like NMS)
		# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
		# inference.
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
		# Overlap threshold used for non-maximum suppression (suppress boxes with
		# IoU >= this threshold)
		self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
		# If True, augment proposals with ground-truth boxes before sampling proposals to
		# train ROI heads.
		#self.cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True


	def train(self, training_source: str, path_prefix: str, validation_source: str, validation_path_prefix: str, resume:bool=False):

		register_dataset(self.cfg.DATASETS.TRAIN[0], training_source, path_prefix)
		register_dataset(self.cfg.DATASETS.TEST[0], validation_source, validation_path_prefix)

		#self.polyp_metadata = get_polyp_metadata(self.cfg.DATASETS.TRAIN[0], training_source, path_prefix )
		#self.cfg.DATASETS.TRAIN = ("polyp_train",)
		self.trainer = PolypCustomTrainer(self.cfg) 
		self.trainer.resume_or_load(resume)
		self.trainer.train()

	def infer(self, data_set: str, training_source: str, path_prefix: str,image_output_folder: str, score_threshold: float = 0.1,model_weight_file="model_final.pth"):
		self.cfg.INPUT.MIN_SIZE_TEST = 0
		#maximum image size for the test set
		self.cfg.INPUT.MAX_SIZE_TEST = 0

		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_weight_file)  # path to the model we just trained
		
		# Inference should use the config with parameters that are used in training
		# cfg now already contains everything we've set previously. We changed it a little bit for inference:

		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
		# Overlap threshold used for non-maximum suppression (suppress boxes with
		# IoU >= this threshold)
		self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01

		#self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
		predictor = DefaultPredictor(self.cfg)

		from detectron2.utils.visualizer import ColorMode

		register_dataset(data_set, training_source, path_prefix)

		polyp_metadata = get_polyp_metadata(data_set)
		
		dataset_dicts = parse_data(training_source, path_prefix)

		for d in dataset_dicts:    
			im = utils.read_image(d["file_name"], format="BGR")
			#im = cv2.imread(d["file_name"])
			image_id = d["image_id"]
			outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
			v = Visualizer(
						im,
						#im[:, :, ::-1],
						metadata=polyp_metadata, 
						scale=1,
						instance_mode= ColorMode.IMAGE
			)

			predictions = outputs["instances"].to("cpu")
			#print(type(predictions),predictions)
			if predictions.has("pred_boxes") is not None:

				np_scores = predictions.scores.numpy()
				np_pred_boxes = predictions.pred_boxes.tensor.detach().numpy()
				np_pred_classes = predictions.pred_classes.numpy()

				total_items = range(np_scores.shape[0])
				final_box_array = np.array([np_pred_boxes[i][:] for i in total_items if np_scores[i] >= score_threshold ])

				desired_boxes = torch.from_numpy(final_box_array)
				desired_scores = np.array([np_scores[i] for i in total_items if np_scores[i] >= score_threshold ])
				desired_classes = np.array([np_pred_classes[i] for i in total_items if np_scores[i] >= score_threshold ])

				labels = _create_text_labels(desired_classes, desired_scores, polyp_metadata.get("thing_classes", None))

				if(len(desired_boxes) > 0):
					#out = v.draw_instance_predictions(predictions)
					out = v.overlay_instances(boxes=desired_boxes,labels= labels )

					prediction_result = out.get_image()
					#print(prediction_result.shape)
					annotation = d["annotations"][0]
					box_coordinates = annotation['bbox']
					#print("Result", annotation, box_coordinates,d["annotations"])

					xmin, ymin, xmax, ymax = tuple(box_coordinates)

					cv2.rectangle(prediction_result,(int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0),2)

					#cv2_imshow(out.get_image()[:, :, ::-1])
					cv2.imwrite(f'{image_output_folder}/{image_id}_{pathlib.Path(d["file_name"]).stem}.jpg', prediction_result)

			#cv2.imwrite("abc.jpg", out.get_image()[:, :, ::-1])

	def evaluate(self,training_source:str, path_prefix: str, output_dir ,data_set,model_weight_file):
		from detectron2.evaluation import PascalVOCDetectionEvaluator,COCOEvaluator, inference_on_dataset
		from detectron2.data import build_detection_test_loader, build_detection_train_loader

		self.cfg.INPUT.MIN_SIZE_TEST = 256
		#  maximum image size for the test set
		self.cfg.INPUT.MAX_SIZE_TEST = 256

		self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, model_weight_file)  # path to the model we just trained
		#self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
		predictor = DefaultPredictor(self.cfg)

		register_dataset(data_set, training_source, path_prefix)

		evaluator = COCOEvaluator(data_set, output_dir=output_dir)
		val_loader = build_detection_test_loader(self.cfg, data_set,num_workers=2,batch_size=6)
		print(inference_on_dataset(predictor.model, val_loader, evaluator))
		# another equivalent way to evaluate the model is to use `trainer.test`