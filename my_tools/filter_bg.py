import os
import pickle
import cv2
import numpy as np
import mmcv


TXT_ROOT = '/data/ChaiJM/Competition/CVPPA-DMOT/Code/ByteTrack-main/results/4ep'
DEPTH_ROOT = '/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT_CVPPA24_DATA/test/depth'
PKL_ROOT = '/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT_CVPPA24_DATA/test/mask2former_output'
DEPTH_Thr = 1200
Valid_Rate = 0.5
SAVE_ROOT = TXT_ROOT + '_filter_mask2former_1200_0.5_0.3'
os.makedirs(SAVE_ROOT, exist_ok=True)
MAX_IoU_Thr = 0.5
MIN_IoU_Thr = 0.3

def load_txt(file_path):
    """加载txt文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    detections = []
    for line in lines:
        parts = line.strip().split(',')
        detection = {
            'image_id': parts[0],
            'target_id': parts[1],
            'x': float(parts[2]),
            'y': float(parts[3]),
            'w': float(parts[4]),
            'h': float(parts[5]),
            'confidence': float(parts[6])
        }
        detections.append(detection)
    return detections

def save_txt(detections, file_path):
    """保存txt文件"""
    with open(file_path, 'w') as f:
        for det in detections:
            line = f"{det['image_id']},{det['target_id']},{det['x']},{det['y']},{det['w']},{det['h']},{det['confidence']}\n"
            f.write(line)

def load_pkl(pkl_file_path):
    """加载pkl文件"""
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_depth_map(depth_file_path):
    """加载深度图"""
    depth_map = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
    return depth_map

def calculate_iou(boxA, boxB):
    """计算两个bbox之间的IoU"""
    xA = max(boxA['x'], boxB['x'])
    yA = max(boxA['y'], boxB['y'])
    xB = min(boxA['x'] + boxA['w'], boxB['x'] + boxB['w'])
    yB = min(boxA['y'] + boxA['h'], boxB['y'] + boxB['h'])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA['w'] * boxA['h']
    boxBArea = boxB['w'] * boxB['h']

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_background(mask, depth_map):
    """判断是否属于背景"""
    # mask_pixels = mask > 0
    # depth_pixels = depth_map[mask_pixels]
    # foreground_pixels = depth_pixels > 1200
    # return np.sum(foreground_pixels) / np.sum(mask_pixels) < 0.5
    indices = np.where(mask > 0)
    valid_indices_number = np.logical_and(depth_map[indices] <= DEPTH_Thr, depth_map[indices] > 0)
    valid_indices_number = np.count_nonzero(valid_indices_number)
    if valid_indices_number / len(indices[0]) > Valid_Rate:
        # it's not bg object
        return False
    else:
        # bg object
        return True


def process_one_scene(scene_id):
    """遍历目录下的所有图片序列，处理每一张图"""
    scene_id = str(scene_id)
    txt_path = os.path.join(TXT_ROOT, scene_id + '.txt')
    new_txt_path = os.path.join(SAVE_ROOT, scene_id + '.txt')
    detections = load_txt(txt_path)
    
    filtered_detections = []
    for det in detections:
        image_id = det['image_id']
        pkl_path = os.path.join(PKL_ROOT, scene_id, image_id + '.pkl')
        depth_path = os.path.join(DEPTH_ROOT, scene_id, image_id + '.tiff')
        pkl_data = load_pkl(pkl_path)
        depth_map = load_depth_map(depth_path)
        boxA = {'x': det['x'], 'y': det['y'], 'w': det['w'], 'h': det['h']}
        keep = True
        max_iou = 0.0
        for instance_id, instance_data in pkl_data.items():
            boxB = {'x': instance_data['bbox'][0], 'y': instance_data['bbox'][1], 'w': instance_data['bbox'][2], 'h': instance_data['bbox'][3]}
            iou = calculate_iou(boxA, boxB)
            max_iou = max(max_iou, iou)
            
            if iou > MAX_IoU_Thr:
                det['x'], det['y'], det['w'], det['h'] = instance_data['bbox'][0], instance_data['bbox'][1], instance_data['bbox'][2], instance_data['bbox'][3]
                mask = instance_data['instance_mask']
                if is_background(mask, depth_map):
                    keep = False
                    break

        if max_iou < MIN_IoU_Thr:
            keep = False

        if keep:
            filtered_detections.append(det)

    save_txt(filtered_detections, new_txt_path)


if __name__ == "__main__":
    scene_list = os.listdir(DEPTH_ROOT)
    mmcv.track_parallel_progress(process_one_scene, scene_list, 16)
