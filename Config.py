import os

class Config:
    
    SOURCE = os.environ.get("SOURCE", "./input_video/input.mp4")
    SAVE_VIDEO = str(os.environ.get("SAVE_VIDEO", "yes")).lower() in ("yes", "true", "t", "1")
    SHOW_VIDEO = bool(os.environ.get("SHOW_VIDEO", True))
    DIRECTORY_SAVE_VIDEO = os.environ.get("DIRECTORY_SAVE_VIDEO", "OUT")
    MODEL_WEIGHT = os.environ.get('MODEL_WEIGHT', './weight/yolov5l.pt')
    MODEL_DEEPSORT = os.environ.get('MODEL_DEEPSORT', 'osnet_x0_25')
    MODEL_CONFIG_DEEPSORT = os.environ.get('MODEL_CONFIG_DEEPSORT', os.path.join('deep_sort', 'configs' , 'deep_sort.yaml'))
    MODEL_CONF_THRESHOLD = os.environ.get('MODEL_CONF_THRESHOLD', 0.75)
    MODEL_IOU_THRESHOLD = os.environ.get('MODEL_IOU_THRESHOLD', 0.5)
    MODEL_ROI = os.environ.get("ROI", [(589.0138854980469,323.3055281154927),(952.0138854980469,312.3055290479741),(942.0138854980469,312.3055290479741),(1006.0138854980469,390.30552243583276),(576.0138854980469,415.3055203165567)]) 