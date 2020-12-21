#!/usr/bin/python3

import argparse

import cv2
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer

from model import get_predictor

parser = argparse.ArgumentParser(description='Inference football helmets on image')
required = parser.add_argument_group('required')
required.add_argument('-i', '--input', type=str, help='path to an image', required=True)
required.add_argument('-o', '--output', type=str, help='output path', required=True)
args = parser.parse_args()

predictor = get_predictor()
image = read_image(args.input, "BGR")
model_output = predictor(image)

model_output = model_output["instances"].to("cpu")
confident_predictions = model_output[model_output.scores > 0.8]

img = cv2.imread(args.input)
v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("nflimpact"))
img = v.draw_instance_predictions(confident_predictions).get_image()

success = cv2.imwrite(args.output, img)
if not success:
    raise ValueError("couldn't write image successfully, sorry opencv does not give helpful error messages")
