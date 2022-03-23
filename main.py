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

if __name__ == "__main__":
	print(list(categories.keys()))
	path_prefix = r"C:/Users/Dev2/Desktop/Jupyter Notebook/PolypsSet/PolypsSet"

	def detection_test():
		polyp_detector = PolypDetector("polyp_train", "polyp_test")

		should_train = False
		if should_train == True:
			polyp_detector.train("train_data.txt", path_prefix)

		should_infer = True
		if should_infer == True:
			output_folder = f'./result_{time.time()}'
			if os.path.exists(output_folder) == False:
				os.mkdir(output_folder)
			polyp_detector.infer("val_data.txt", path_prefix,output_folder)

		should_evaluate = False
		if should_evaluate == True:
			new_output_dir = f"./output_{time.time()}"
			if os.path.exists(new_output_dir) == False:
				os.mkdir(new_output_dir)
			polyp_detector.evaluate("train_data.txt", path_prefix,new_output_dir,data_set="polyp_train")

	detection_test()

	# polyp_metadata = get_polyp_metadata()
	# dataset_dicts = parse_data()

	# for d in random.sample(dataset_dicts, 3):
	#     img = cv2.imread(d["file_name"])
	#     visualizer = Visualizer(img[:, :, ::-1], metadata=polyp_metadata, scale=0.5)
	#     out = visualizer.draw_dataset_dict(d)
	#     print(">>",d,pathlib.Path(d["file_name"]).stem)
	#     cv2.imwrite(f'{pathlib.Path(d["file_name"]).stem}.jpg', out.get_image())

	#extract_required_data()


	# print(torch.cuda.is_available())

	# np_array = np.array([1,3,2])
	# print(torch.from_numpy(np_array))

	# detector = Detector()

	# detector.onImage(r"C:\Users\Dev2\Desktop\PUL076MSDSA016.jpg")
