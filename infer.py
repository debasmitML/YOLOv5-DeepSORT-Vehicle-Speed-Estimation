

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.insert(0, './yolov5')

import os
import shutil
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn


from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

dict_info = dict()
#list_conf = []
ini_count = 0
alert_dict = {}


    
def detect(source, yolo_model, deep_sort_model, device, show_vid, save_vid, imgsz, config_deepsort, half,dnn,ROI,augment,conf_thres,iou_thres,classes,agnostic_nms,max_det,save_dir):
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    imgsz *= 2 if len(imgsz) == 1 else 1 
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA


    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    os.makedirs(save_dir, exist_ok=True)
    video_writer = cv2.VideoWriter(os.path.join(save_dir,"output.avi"), cv2.VideoWriter_fourcc(*'MJPG'), 29, (1280, 720))

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
   

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    frame_skip_deletion = 20
    initial_frame_info = dict()
    ini = 0
   
    
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        
        TrackId = []
        
    
        t1 = time_sync()

       
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1


        
        pred = model(img, augment=augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred,conf_thres,iou_thres,classes,agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
        
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            im0 = cv2.resize(im0,(1280,720))
            p = Path(p) 
            s += '%gx%g ' % img.shape[2:]  

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
           
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    print()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

            
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
    
                if len(outputs) > 0 :
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                    
                        global cx,cy
                        cx, cy = (int(bboxes[0]+(bboxes[2]-bboxes[0])/2) , int(bboxes[1]+(bboxes[3]-bboxes[1])/2))    
                        c = int(cls)  # integer class
                    
                        if c==2 or c== 5 or c==7:
                    
                            result = cv2.pointPolygonTest(np.array(ROI,np.int32),(int(cx),int(cy)),False)
                            if result>=0:
                                TrackId.append(id) 
                                annotator.box_label(bboxes,f'track_id: {id}', color=(0,255,0))        
                                    
                            if len(alert_dict)!=0:   
                                for id3 in list(alert_dict.keys()):
                                    if id3 == id:
                                        annotator.box_label(bboxes,alert_dict[id3], color=(0,255,0))
                    
                            else:
                                annotator.box_label(bboxes,f'track id: {id}', color=(0,255,0))
                                    
                                
                        for key11 in list(initial_frame_info.keys()):
                            if initial_frame_info[key11] == 20:
                                if dict_info[key11] >= 19.44:
                                    alert = f'{round((5/(int(dict_info[key11])/50)) * 3.6)} km/h'
                                    alert_dict[key11] = alert

                                    


                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
               

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
            
            
           
                ############# IMP ################
            try:
                for i in TrackId:
                    if i not in list(dict_info.keys()):
                        dict_info[i]=ini_count
                
                for key in list(dict_info.keys()):
                    
                    if key not in TrackId:
                        
                        if key not in list(initial_frame_info.keys()):
                            initial_frame_info[key] = ini
                            
                        if initial_frame_info[key] == frame_skip_deletion:
                            dict_info.pop(key)
                            initial_frame_info.pop(key)
                            
                        elif len(TrackId) == 0:
                            initial_frame_info[key] = initial_frame_info[key] + 1    
                        else:
                            initial_frame_info[key] = initial_frame_info[key] + 1
                                    
                        
                    else:
                        dict_info[key] = dict_info[key] + 1 
            except:
                print('not starting')
            
            
            # Stream results
            im0 = annotator.result()
            
            if show_vid:
                cv2.polylines(im0,[np.array(ROI,np.int32)],True,(0,255,0),3)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == 13:
                    break
                

            # Save results (Inference Video)
            if save_vid:
                video_writer.write(im0)
    
    
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    




