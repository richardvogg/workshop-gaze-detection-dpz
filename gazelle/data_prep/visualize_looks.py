import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
root_dir = 'data/lemurattentiontarget'

dataset_type = 'test'  # or 'test'
video_id = '7'
camera_id = 'c6'
#subject_file = 's02.txt'
frame_name = 'frame10039.jpg'
#output_dir = os.path.join('output_images', 'images', video_id, camera_id)  # Directory to save output images

for video_id in ['4', '6', '7', '8', '9', '10']:
    for camera_id in ['c6', 'c7']:
        if video_id == '7':
            dataset_type = 'test'
        else:
            dataset_type = 'train'
        output_dir = os.path.join('/usr/users/vogg/sfb1528s3/B06/gaze_estimation/output_images', 'images', video_id, camera_id)
        os.makedirs(output_dir, exist_ok=True)

        frame_names = [f for f in os.listdir(os.path.join(root_dir, 'images', video_id, camera_id)) if f.endswith('.jpg')]


        anno_dir = os.path.join(root_dir, 'annotations', dataset_type, video_id, camera_id)
        anno_lines = []
        for fname in os.listdir(anno_dir):
            if fname.endswith('.txt'):
                with open(os.path.join(anno_dir, fname), 'r') as f:
                    anno_lines.extend(f.readlines())

        # Split lines into DataFrame
        anno_data = []
        for line in anno_lines:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                frame, x1, y1, x2, y2, gx, gy = parts[:7]
                anno_data.append([frame, int(x1), int(y1), int(x2), int(y2), int(gx), int(gy)])
        anno_df = pd.DataFrame(anno_data, columns=['frame_name', 'x1', 'y1', 'x2', 'y2', 'gx', 'gy'])



    
        for frame_name in frame_names:
            # === Paths ===
            img_path = os.path.join(root_dir, 'images', video_id, camera_id, frame_name)
            
            anno_df_filtered = anno_df[anno_df['frame_name'] == frame_name]


            # === Load Image ===
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # === Load Annotation ===
            bboxes = [
                (row['x1'], row['y1'], row['x2'], row['y2'], row['gx'], row['gy'])
                for _, row in anno_df_filtered.iterrows()
            ]

            if len(bboxes) == 0:
                raise ValueError(f"No annotation found for frame {frame_name}")

            # === Draw Bounding Box and Gaze ===

            for bbox in bboxes:
                x1, y1, x2, y2, gx, gy = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Draw box
                # Draw yellow rectangle with alpha=0.5
                overlay = img_rgb.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
                img_rgb = cv2.addWeighted(overlay, 0.5, img_rgb, 0.5, 0)

                # Draw arrow
                if gx != -1 and gy != -1:
                    cv2.arrowedLine(img_rgb, (cx, cy), (gx, gy), (255, 255, 0), 2, tipLength=0.05)

            # === Show Result ===
            output_path = os.path.join(output_dir, frame_name)
            plt.figure(figsize=(12, 8))  # Set higher resolution
            plt.imshow(img_rgb)
            plt.title(f"{frame_name}")
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)  # Save at higher DPI
            plt.close()
            #print(f"Saved visualization to {output_path}")

