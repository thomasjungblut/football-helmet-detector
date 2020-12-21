#!/usr/bin/python3
import os
import pickle

import numpy as np
import pandas as pd
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader, \
    DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

from model import new_model_cfg

print(torch.__version__, torch.cuda.is_available())


def convert_video_labels(all_videos, video_labels: pd.DataFrame, train_videos):
    dataset_dicts = []
    videos_processed = 0
    for video in all_videos:
        is_train = True if video in train_videos else False
        frames = video_labels.query("video == @video")
        min_frame = frames['frame'].min()
        max_frame = frames['frame'].max()

        for i in range(min_frame, max_frame + 1):
            player_frames = frames.query("frame == @i")
            img_path = os.path.join("data", "nfl_impact_images_train", video, str.format("{}", i).zfill(6) + '.png')
            if not os.path.exists(img_path):
                continue
            record = {"image_id": img_path,
                      "video_name": video,
                      "file_name": img_path,
                      "frame": i,
                      "height": 720,
                      "width": 1280}
            objects = []
            if not player_frames.empty:
                for _, row in player_frames.iterrows():
                    obj = {
                        "bbox": [row['left'],
                                 row['top'],
                                 row['left'] + row['width'],
                                 row['top'] + row['height']],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                    objects.append(obj)

            record["annotations"] = objects
            if is_train:
                if objects:
                    dataset_dicts.append(record)
            else:
                dataset_dicts.append(record)

        print("conversion progress %d/%d" % (videos_processed, len(all_videos)))
        videos_processed += 1

    return dataset_dicts


class NflImpactTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[])
        return build_detection_test_loader(DatasetCatalog.get(dataset_name), mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator("nflimpact_test", ('bbox',), use_fast_impl=False, output_dir="output")


if __name__ == '__main__':
    pickle_path = 'output/dataset_dict.pkl'
    if not os.path.exists(pickle_path):
        video_labels = pd.read_csv('data/train_labels.csv')
        video_labels = video_labels.replace({np.nan: None})
        all_videos = video_labels['video'].unique().tolist()
        train_vids, test_vids = train_test_split(all_videos, test_size=0.05)
        dataset_dicts = convert_video_labels(all_videos, video_labels, train_vids)

        train = []
        test = []

        for r in dataset_dicts:
            if r['video_name'] in train_vids:
                train.append(r)
            else:
                test.append(r)

        print("train: {} test: {}".format(len(train), len(test)))
        with open(pickle_path, 'wb') as f:
            pickle.dump((train, test), f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_path, 'rb') as f:
            (train, test) = pickle.load(f)

    DatasetCatalog.register("nflimpact_train", lambda: train)
    DatasetCatalog.register("nflimpact_test", lambda: test)

    classes = ["helmet"]
    MetadataCatalog.get("nflimpact_train").set(thing_classes=classes)
    MetadataCatalog.get("nflimpact_test").set(thing_classes=classes)

    cfg = new_model_cfg()
    cfg.DATASETS.TRAIN = ("nflimpact_train",)
    cfg.DATASETS.TEST = ("nflimpact_test",)

    trainer = NflImpactTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
