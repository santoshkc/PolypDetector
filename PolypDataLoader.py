
from os import popen
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import cv2
from detectron2.structures import BoxMode

categories = { 'adenomatous': 0, 'hyperplastic': 1 }

def parse_data(input_source_file: str, pathPrefix = "./polypsSet" ):
    input_filename = "val_data.txt"

    fileprefix = r"C:/Users/Dev2/Desktop/Jupyter Notebook/PolypsSet/PolypsSet"
    dataset_dicts = []

    with open(input_source_file) as data:
        for index, line in enumerate(data.readlines()):

            # if index > 10:
            #     break
            tokens = line.strip().split(',')
            filepath, xmin, ymin, xmax, ymax, class_name = tuple(tokens)

            new_path = filepath.replace("../PolypsSet", pathPrefix)
            record = {}
            height, width = cv2.imread(new_path).shape[:2]
            
            record["file_name"] = new_path
            record["image_id"] = index
            record["height"] = height
            record["width"] = width

            annotations = [{
                "bbox": [float(xmin),float(ymin),float(xmax),float(ymax)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": categories[class_name],
            }]

            record["annotations"] = annotations
            dataset_dicts.append(record)
            #print(record)
    return dataset_dicts
#parse_data()

def get_polyp_metadata(meta_type, input_source: str, path_prefix):

    if meta_type not in DatasetCatalog.keys():
        DatasetCatalog.register(meta_type, lambda d= meta_type: parse_data(input_source, path_prefix) )
    
    MetadataCatalog.get(meta_type).set(thing_classes=list(categories.keys()))

    polyp_metadata = MetadataCatalog.get(meta_type)
    #print(polyp_metadata)

    return polyp_metadata

