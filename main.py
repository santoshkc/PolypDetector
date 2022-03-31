from email.policy import default
import random

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import torch
import numpy as np

import zipfile
import os

import pathlib

from PolypDataLoader import get_polyp_metadata, parse_data
from Detector import PolypDetector

def extract_required_data(): 
	current_dir = pathlib.Path.cwd()

	zip_path = str(pathlib.Path.joinpath(current_dir,"balloon_dataset.zip"))
	dest_path = str(pathlib.Path.joinpath(current_dir,"balloon_dataset"))

def extract_zip_file(zip_file,destination_path):
	with zipfile.ZipFile(zip_file,"r") as zip_ref:
		zip_ref.extractall(destination_path)

#extract_zip_file(zip_path, dest_path)

import matplotlib.pyplot as plt

categories = { 'adenomatous': 0, 'hyperplastic': 1 }

import os
import time

import argparse

def get_parser():
	parser = argparse.ArgumentParser(description="Polyp configs")

	parser.add_argument("--train", action="store_true", help="use for training")
	parser.add_argument("--infer", action="store_true", help="use for inference.")
	parser.add_argument("--evaluate", action="store_true", help="use for evaluation.")
	
	parser.add_argument("--epoch", type= int , default=5, help="total epoch")
	parser.add_argument("--checkpoint-per-epoch", type= int , default=5, help="total checkpoints during epoch")
	parser.add_argument("--validation-per-epoch", type= int , default=2, help="total validations during epoch")

	parser.add_argument("--min-train-image-size", type=int, default=512, help="training image size")
	parser.add_argument("--max-train-image-size", type=int, default=512, help="training image size")

	parser.add_argument("--min-test-image-size", type=int, default=512, help="test image size")
	parser.add_argument("--max-test-image-size", type=int, default=512, help="test image size")

	parser.add_argument("--total-image-samples", type=int, default=27048, help="total image count in training dataset.")
	parser.add_argument("--image-count-per-batch", type=int, default=32, help="total image acceepted in batch")
	
	parser.add_argument("--unique-run", action="store_true", help="clean run for train/infer/evaluate.")
    
	parser.add_argument(
		"--weights-output-dir",
		default="./detectron_output",
		help="Place to save training weights and logs"
	)

	parser.add_argument(
		"--model-weight-file",
		default="model_final.pth",
		help="Model name to be used for training/resuming weights"
	)

	parser.add_argument(
		"--detection-output-dir",
		default="./detection_result",
		help="Place to store image when generated during inference"
	)

	parser.add_argument(
		"--path-prefix",
		default="../PolypsSet",
		help="source image prefix to be used for training samples"
	)
	return parser


def parse_arguments():
	pass

if __name__ == "__main__":
	print(list(categories.keys()))

	args = get_parser().parse_args()

	#print("Before: ", args)

	if args.path_prefix is None:
		args.path_prefix = r"../PolypsSet"

	if args.unique_run is True:
		args.weights_output_dir = f"{args.weights_output_dir}_{time.time()}"

	print(f"Training Weights/logs folder:  {args.weights_output_dir}")

	if args.unique_run is True:
		args.detection_output_dir = f"{args.detection_output_dir}_{time.time()}"

	print(f"Detection result folder:  {args.detection_output_dir}")

	if args.model_weight_file is None:
		args.model_weight_file = "model_final.pth"
	
	#args.path_prefix = r"C:/Users/Dev2/Desktop/Jupyter Notebook/PolypsSet/PolypsSet"
	print("After: ", args)

	def detection_test():
		#output_folder = '/content/drive/MyDrive/detectron_train_2'
		#output_folder = './detectron_train_pcampus'
		#output_folder = '/kaggle/working/detectron_train_pcampus'
		polyp_detector = PolypDetector("polyp_train", "polyp_validation", 
			default_output_dir=args.weights_output_dir,
			total_epoch=args.epoch,
			image_count_per_batch= args.image_count_per_batch,
			min_train_image_size = args.min_train_image_size,
			max_train_image_size = args.max_train_image_size,
			min_test_image_size = args.min_test_image_size,
			max_test_image_size = args.max_test_image_size,
			total_image_samples=args.total_image_samples,
			checkpoint_per_epoch = args.checkpoint_per_epoch,
			validation_per_epoch = args.validation_per_epoch,

			)

		if args.train == True:
			polyp_detector.train("train_data.txt", args.path_prefix,"val_data_final.txt", args.path_prefix,
			resume = not args.unique_run
			)

		if args.infer == True:
			if os.path.exists(args.detection_output_dir) == False:
				os.mkdir(args.detection_output_dir)
			#polyp_detector.infer("polyp_train", "train_data.txt", args.path_prefix, args.detection_output_dir)
			polyp_detector.infer("polyp_validation", "val_data_final.txt", args.path_prefix, 
			args.detection_output_dir,
			model_weight_file=args.model_weight_file,
			)

		if args.evaluate == True:
			new_output_dir = f"./output_{time.time()}"
			#new_output_dir = "output_1648726312.2160532"
			if os.path.exists(new_output_dir) == False:
				os.mkdir(new_output_dir)
			polyp_detector.evaluate("train_data.txt", args.path_prefix,new_output_dir,data_set="polyp_train",
			model_weight_file=args.model_weight_file,)

	detection_test()
