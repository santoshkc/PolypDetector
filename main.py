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
    parser.add_argument("--infer", action="store_true", help="infer.")
    
    parser.add_argument(
        "--output-dir",
        help="Save training weights and logs"
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def parse_arguments():
	pass

if __name__ == "__main__":
	print(list(categories.keys()))
	path_prefix = r"../PolypsSet"
	#path_prefix = r"C:/Users/Dev2/Desktop/Jupyter Notebook/PolypsSet/PolypsSet"

	def detection_test():
		#output_folder = f'./result_{time.time()}'
		#output_folder = './result_1648384510.6090004'
		#output_folder = '/content/drive/MyDrive/detectron_train_2'
		#output_folder = './detectron_train_pcampus'
		output_folder = '/kaggle/working/detectron_train_pcampus'
		polyp_detector = PolypDetector("polyp_train", "polyp_validation", default_output_dir=output_folder)

		should_train = True
		
		if should_train == True:
			polyp_detector.train("train_data.txt", path_prefix,"val_data_final.txt", path_prefix)

		should_infer = False
		if should_infer == True:
			output_folder = f'./result_{time.time()}'
			#output_folder = './detectron_train_pcampus'
			if os.path.exists(output_folder) == False:
				os.mkdir(output_folder)
			#polyp_detector.infer("polyp_train", "train_data.txt", path_prefix, output_folder)
			polyp_detector.infer("polyp_validation", "val_data_final.txt", path_prefix, output_folder)

		should_evaluate = False
		if should_evaluate == True:
			new_output_dir = f"./output_{time.time()}"
			if os.path.exists(new_output_dir) == False:
				os.mkdir(new_output_dir)
			polyp_detector.evaluate("train_data.txt", path_prefix,new_output_dir,data_set="polyp_train")

	detection_test()
