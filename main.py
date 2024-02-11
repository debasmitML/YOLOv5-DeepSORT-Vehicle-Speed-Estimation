import argparse
import os
import sys
import os.path as osp
import json
from infer import detect
import torch
from Config import Config as cfg


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    with torch.no_grad():
        detect(cfg.SOURCE, cfg.MODEL_WEIGHT, cfg.MODEL_DEEPSORT, args.device, cfg.SHOW_VIDEO, cfg.SAVE_VIDEO,
                args.imgsz, cfg.MODEL_CONFIG_DEEPSORT, args.half, args.dnn,cfg.MODEL_ROI, args.augment,cfg.MODEL_CONF_THRESHOLD,cfg.MODEL_IOU_THRESHOLD, args.classes, args.agnostic_nms, args.max_det, cfg.DIRECTORY_SAVE_VIDEO)

