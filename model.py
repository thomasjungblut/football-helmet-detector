from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo


def new_model_cfg():
    cfg = get_cfg()
    model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.MODEL.MASK_ON = False
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.OUTPUT_DIR = "output"

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.STEPS = (21000, 50000)
    cfg.SOLVER.MAX_ITER = 200000
    cfg.SOLVER.BASE_LR = 0.001
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    return cfg


def get_predictor():
    classes = ['helmet']
    MetadataCatalog.get("nflimpact").set(thing_classes=classes)
    cfg = new_model_cfg()
    cfg.MODEL.WEIGHTS = "model/helmet_detector_model_0093999.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)
