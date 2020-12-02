#!/usr/bin/python3
import argparse
import os
import subprocess

import cv2
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer

from model import get_predictor

parser = argparse.ArgumentParser(description='Inference football helmets on a mp4 video')
required = parser.add_argument_group('required')
required.add_argument('-i', '--input', type=str, help='path to a video', required=True)
required.add_argument('-o', '--output', type=str, help='output path', required=True)
args = parser.parse_args()

predictor = get_predictor()

video_codec = "MP4V"
color = (255, 255, 255)
input_path = args.input
output_path = args.output

vid_cap = cv2.VideoCapture(input_path)
tmp_output_path = output_path + ".tmp.mp4"
fps = 60
width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*video_codec), fps, (width, height))

while True:
    more_frames, img = vid_cap.read()
    if not more_frames:
        break

    model_output = predictor(img)

    model_output = model_output["instances"].to("cpu")
    filter_mask = model_output.scores > 0.8
    ni = Instances(model_output.image_size, **{
        "scores": model_output.scores[filter_mask],
        "pred_boxes": model_output.pred_boxes[filter_mask],
        "pred_classes": model_output.pred_classes[filter_mask]
    })

    v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("nflimpact"))
    img = v.draw_instance_predictions(ni).get_image()
    output_video.write(img)

output_video.release()

if os.path.exists(output_path):
    os.remove(output_path)

subprocess.run(
    ["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", output_path])
os.remove(tmp_output_path)
