#!/usr/bin/python3

import os
from multiprocessing import Pool

import cv2


def write_frames(video_path):
    video_name = os.path.basename(video_path)
    output_base_path = "data/nfl_impact_images_train"
    os.makedirs(os.path.join(output_base_path, video_name), exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    frame = 0
    while True:
        more_frames, img = vidcap.read()
        if not more_frames:
            break

        frame += 1
        img_name = "{}".format(frame).zfill(6) + ".png"
        success = cv2.imwrite(os.path.join(output_base_path, video_name, img_name), img)
        if not success:
            raise ValueError("couldn't write image successfully")


if __name__ == '__main__':
    train_videos = os.listdir("data/train")
    pool = Pool()
    pool.map(write_frames, map(lambda video_name: f"data/train/{video_name}", train_videos))
