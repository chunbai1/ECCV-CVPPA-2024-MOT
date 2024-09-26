import os
import shutil
import cv2
import pickle
import tqdm
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--subset', type=str, default='train', choices=['train', 'val', 'test'])
args = parser.parse_args()

if args.subset == 'train':
    """
    train:
    """
    def create_seqinfo(dataset_dir, sequence_name, frame_rate, seq_length, im_width, im_height, im_ext=".jpg"):
        seqinfo_content = f"""[Sequence]
    name={sequence_name}
    imDir=img1
    frameRate={frame_rate}
    seqLength={seq_length}
    imWidth={im_width}
    imHeight={im_height}
    imExt={im_ext}
    """
        seqinfo_path = os.path.join(dataset_dir, "train", sequence_name, "seqinfo.ini")
        with open(seqinfo_path, "w") as f:
            f.write(seqinfo_content)

    def convert_to_mot20_format(dataset_root, output_dir, frame_rate=25, im_width=720, im_height=1280):
        os.makedirs(output_dir, exist_ok=True)
        
        train_dir = os.path.join(dataset_root, "train", "rgb")
        for scene_id in tqdm.tqdm(os.listdir(train_dir)):
            if not scene_id.isdigit():  # 跳过非数字文件夹
                continue
            
            scene_dir = os.path.join(train_dir, scene_id)
            
            # Create output directories
            sequence_name = f"MOT20-{int(scene_id):02d}"
            output_sequence_dir = os.path.join(output_dir, "train", sequence_name)
            os.makedirs(os.path.join(output_sequence_dir, "img1"), exist_ok=True)
            os.makedirs(os.path.join(output_sequence_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(output_sequence_dir, "det"), exist_ok=True)
            
            # Copy images to the img1 directory and convert them to .jpg
            # rgb_dir = os.path.join(scene_dir, "rgb", scene_id)
            rgb_dir = scene_dir
            # img_files = sorted(os.listdir(rgb_dir))
            img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.tiff', '.jpg', '.png')) and f != 'gt.txt'])
            for i, img_file in enumerate(img_files):
                img_path = os.path.join(rgb_dir, img_file)
                img = cv2.imread(img_path)
                output_img_path = os.path.join(output_sequence_dir, "img1", f"{i+1:06d}.jpg")
                cv2.imwrite(output_img_path, img)
            
            # Generate gt.txt
            gt_txt_path = os.path.join(output_sequence_dir, "gt", "gt.txt")
            with open(gt_txt_path, "w") as gt_file:
                weak_label_dir = os.path.join(dataset_root, "train", "weak_labels", scene_id)
                for i, img_file in enumerate(img_files):
                    frame_num = i + 1
                    pkl_file = os.path.join(weak_label_dir, img_file.replace(".tiff", ".pkl"))
                    if os.path.exists(pkl_file):
                        with open(pkl_file, "rb") as f:
                            annotations = pickle.load(f)
                            for j, annotation in enumerate(annotations.items()):
                                bbox = annotation[1].get("bbox", [])
                                if len(bbox) != 4:
                                    continue  # 跳过空的或无效的 bbox
                                instance_id = j + 1
                                x_min, y_min, width, height = annotation[1]["bbox"]
                                # sem_label = annotation["semantic_label"]
                                sem_label = 1.0
                                visibility = 1.0  # Assuming full visibility since no occlusion data is provided
                                gt_line = f"{frame_num},{instance_id},{x_min},{y_min},{width},{height},1,{sem_label},{visibility}\n"
                                gt_file.write(gt_line)
            
            # Generate det.txt (using mask2former output)
            det_txt_path = os.path.join(output_sequence_dir, "det", "det.txt")
            with open(det_txt_path, "w") as det_file:
                mask2former_dir = os.path.join(dataset_root, "train", "mask2former_output", scene_id)
                for i, img_file in enumerate(img_files):
                    frame_num = i + 1
                    pkl_file = os.path.join(mask2former_dir, img_file.replace(".tiff", ".pkl"))
                    if os.path.exists(pkl_file):
                        with open(pkl_file, "rb") as f:
                            detections = pickle.load(f)
                            for j, detection in enumerate(detections.items()):
                                bbox = detection[1].get("bbox", [])
                                if len(bbox) != 4:
                                    continue  # 跳过空的或无效的 bbox
                                instance_id = j + 1
                                x_min, y_min, width, height = detection[1]["bbox"]
                                det_line = f"{frame_num},-1,{x_min},{y_min},{width},{height},1,-1,-1,-1\n"
                                det_file.write(det_line)
            
            # Create seqinfo.ini
            seq_length = len(img_files)
            create_seqinfo(output_dir, sequence_name, frame_rate, seq_length, im_width, im_height)

    # Example usage
    dataset_root = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT_CVPPA24_DATA"
    output_dir = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT20_CVPPA24_DATA"
    convert_to_mot20_format(dataset_root, output_dir)
elif args.subset == 'val':

    """
    valid:
    """
    def create_seqinfo(dataset_dir, sequence_name, frame_rate, seq_length, im_width, im_height, im_ext=".jpg"):
        seqinfo_content = f"""[Sequence]
    name={sequence_name}
    imDir=img1
    frameRate={frame_rate}
    seqLength={seq_length}
    imWidth={im_width}
    imHeight={im_height}
    imExt={im_ext}
    """
        seqinfo_path = os.path.join(dataset_dir, "train", sequence_name, "seqinfo.ini")
        with open(seqinfo_path, "w") as f:
            f.write(seqinfo_content)

    def convert_to_mot20_format(dataset_root, output_dir, frame_rate=25, im_width=720, im_height=1280):
        os.makedirs(output_dir, exist_ok=True)
        
        train_dir = os.path.join(dataset_root, "valid", "rgb")
        for scene_id in tqdm.tqdm(os.listdir(train_dir)):
            if not scene_id.isdigit():  # 跳过非数字文件夹
                continue
            
            scene_dir = os.path.join(train_dir, scene_id)
            
            # Create output directories
            sequence_name = f"MOT20-{int(scene_id):02d}"
            output_sequence_dir = os.path.join(output_dir, "train", sequence_name)
            os.makedirs(os.path.join(output_sequence_dir, "img1"), exist_ok=True)
            os.makedirs(os.path.join(output_sequence_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(output_sequence_dir, "det"), exist_ok=True)
            
            # Copy images to the img1 directory and convert them to .jpg
            # rgb_dir = os.path.join(scene_dir, "rgb", scene_id)
            rgb_dir = scene_dir
            # img_files = sorted(os.listdir(rgb_dir))
            img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.tiff', '.jpg', '.png')) and f != 'gt.txt'])
            for i, img_file in enumerate(img_files):
                img_path = os.path.join(rgb_dir, img_file)
                img = cv2.imread(img_path)
                output_img_path = os.path.join(output_sequence_dir, "img1", f"{i+1:06d}.jpg")
                cv2.imwrite(output_img_path, img)
            
            # Generate gt.txt
            gt_txt_path = os.path.join(output_sequence_dir, "gt", "gt.txt")
            with open(gt_txt_path, "w") as gt_file:
                weak_label_dir = os.path.join(dataset_root, "valid", "weak_labels", scene_id)
                for i, img_file in enumerate(img_files):
                    frame_num = i + 1
                    pkl_file = os.path.join(weak_label_dir, img_file.replace(".tiff", ".pkl"))
                    if os.path.exists(pkl_file):
                        with open(pkl_file, "rb") as f:
                            annotations = pickle.load(f)
                            for j, annotation in enumerate(annotations.items()):
                                bbox = annotation[1].get("bbox", [])
                                if len(bbox) != 4:
                                    continue  # 跳过空的或无效的 bbox
                                instance_id = j + 1
                                x_min, y_min, width, height = annotation[1]["bbox"]
                                # sem_label = annotation["semantic_label"]
                                sem_label = 1.0
                                visibility = 1.0  # Assuming full visibility since no occlusion data is provided
                                gt_line = f"{frame_num},{instance_id},{x_min},{y_min},{width},{height},1,{sem_label},{visibility}\n"
                                gt_file.write(gt_line)
            
            # Generate det.txt (using mask2former output)
            det_txt_path = os.path.join(output_sequence_dir, "det", "det.txt")
            with open(det_txt_path, "w") as det_file:
                mask2former_dir = os.path.join(dataset_root, "valid", "mask2former_output", scene_id)
                for i, img_file in enumerate(img_files):
                    frame_num = i + 1
                    pkl_file = os.path.join(mask2former_dir, img_file.replace(".tiff", ".pkl"))
                    if os.path.exists(pkl_file):
                        with open(pkl_file, "rb") as f:
                            detections = pickle.load(f)
                            for j, detection in enumerate(detections.items()):
                                bbox = detection[1].get("bbox", [])
                                if len(bbox) != 4:
                                    continue  # 跳过空的或无效的 bbox
                                instance_id = j + 1
                                x_min, y_min, width, height = detection[1]["bbox"]
                                det_line = f"{frame_num},-1,{x_min},{y_min},{width},{height},1,-1,-1,-1\n"
                                det_file.write(det_line)
            
            # Create seqinfo.ini
            seq_length = len(img_files)
            create_seqinfo(output_dir, sequence_name, frame_rate, seq_length, im_width, im_height)

    # Example usage
    dataset_root = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT_CVPPA24_DATA"
    output_dir = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT20_CVPPA24_DATA"
    convert_to_mot20_format(dataset_root, output_dir)
elif args.subset == 'test':
    """
    test
    """
    def create_seqinfo(dataset_dir, sequence_name, frame_rate, seq_length, im_width, im_height, im_ext=".jpg"):
        seqinfo_content = f"""[Sequence]
    name={sequence_name}
    imDir=img1
    frameRate={frame_rate}
    seqLength={seq_length}
    imWidth={im_width}
    imHeight={im_height}
    imExt={im_ext}
    """
        seqinfo_path = os.path.join(dataset_dir, "test", sequence_name, "seqinfo.ini")
        with open(seqinfo_path, "w") as f:
            f.write(seqinfo_content)

    def convert_to_mot20_format(dataset_root, output_dir, frame_rate=25, im_width=720, im_height=1280):
        os.makedirs(output_dir, exist_ok=True)
        
        train_dir = os.path.join(dataset_root, "test", "rgb")
        for scene_id in tqdm.tqdm(os.listdir(train_dir)):
            if not scene_id.isdigit():  # 跳过非数字文件夹
                continue
            
            scene_dir = os.path.join(train_dir, scene_id)
            
            # Create output directories
            sequence_name = f"MOT20-{int(scene_id):02d}"
            output_sequence_dir = os.path.join(output_dir, "test", sequence_name)
            os.makedirs(os.path.join(output_sequence_dir, "img1"), exist_ok=True)
            # os.makedirs(os.path.join(output_sequence_dir, "gt"), exist_ok=True)
            os.makedirs(os.path.join(output_sequence_dir, "det"), exist_ok=True)
            
            # Copy images to the img1 directory and convert them to .jpg
            # rgb_dir = os.path.join(scene_dir, "rgb", scene_id)
            rgb_dir = scene_dir
            # img_files = sorted(os.listdir(rgb_dir))
            img_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.tiff', '.jpg', '.png')) and f != 'gt.txt'])
            for i, img_file in enumerate(img_files):
                img_path = os.path.join(rgb_dir, img_file)
                img = cv2.imread(img_path)
                output_img_path = os.path.join(output_sequence_dir, "img1", f"{i+1:06d}.jpg")
                cv2.imwrite(output_img_path, img)
            
            # Generate gt.txt
            # gt_txt_path = os.path.join(output_sequence_dir, "gt", "gt.txt")
            # with open(gt_txt_path, "w") as gt_file:
            #     weak_label_dir = os.path.join(dataset_root, "valid", "weak_labels", scene_id)
            #     for i, img_file in enumerate(img_files):
            #         frame_num = i + 1
            #         pkl_file = os.path.join(weak_label_dir, img_file.replace(".tiff", ".pkl"))
            #         if os.path.exists(pkl_file):
            #             with open(pkl_file, "rb") as f:
            #                 annotations = pickle.load(f)
            #                 for j, annotation in enumerate(annotations.items()):
            #                     bbox = annotation[1].get("bbox", [])
            #                     if len(bbox) != 4:
            #                         continue  # 跳过空的或无效的 bbox
            #                     instance_id = j + 1
            #                     x_min, y_min, width, height = annotation[1]["bbox"]
            #                     # sem_label = annotation["semantic_label"]
            #                     sem_label = 1.0
            #                     visibility = 1.0  # Assuming full visibility since no occlusion data is provided
            #                     gt_line = f"{frame_num},{instance_id},{x_min},{y_min},{width},{height},1,{sem_label},{visibility}\n"
            #                     gt_file.write(gt_line)
            
            # Generate det.txt (using mask2former output)
            det_txt_path = os.path.join(output_sequence_dir, "det", "det.txt")
            with open(det_txt_path, "w") as det_file:
                mask2former_dir = os.path.join(dataset_root, "test", "mask2former_output", scene_id)
                for i, img_file in enumerate(img_files):
                    frame_num = i + 1
                    pkl_file = os.path.join(mask2former_dir, img_file.replace(".tiff", ".pkl"))
                    if os.path.exists(pkl_file):
                        with open(pkl_file, "rb") as f:
                            detections = pickle.load(f)
                            for j, detection in enumerate(detections.items()):
                                bbox = detection[1].get("bbox", [])
                                if len(bbox) != 4:
                                    continue  # 跳过空的或无效的 bbox
                                instance_id = j + 1
                                x_min, y_min, width, height = detection[1]["bbox"]
                                det_line = f"{frame_num},-1,{x_min},{y_min},{width},{height},1,-1,-1,-1\n"
                                det_file.write(det_line)
            
            # Create seqinfo.ini
            seq_length = len(img_files)
            create_seqinfo(output_dir, sequence_name, frame_rate, seq_length, im_width, im_height)

    # Example usage
    dataset_root = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT_CVPPA24_DATA"
    output_dir = "/data/ChaiJM/Competition/CVPPA-DMOT/Dataset/MOT20_CVPPA24_DATA"
    convert_to_mot20_format(dataset_root, output_dir)

else:
    raise NotImplementedError