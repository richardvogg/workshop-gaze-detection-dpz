import os
import json
import random
import cv2
from collections import defaultdict


BASE_PATH = "/user/vogg/u24033/.project/dir.project/Richard_Vogg/data/Rocamadour"
VIDEO_DIR = os.path.join(BASE_PATH, "videos")
IMG_DIR = os.path.join(BASE_PATH, "interaction_images")
LABEL_DIR = os.path.join(BASE_PATH, "interaction_labels")   # where tracking files live

TRAIN_JSON = os.path.join(LABEL_DIR, "train_preprocessed.json")
VAL_JSON   = os.path.join(LABEL_DIR, "val_preprocessed.json")

OUT_TRAIN_JSON = os.path.join(LABEL_DIR, "train_preprocessed_with_neg.json")
OUT_VAL_JSON   = os.path.join(LABEL_DIR, "val_preprocessed_with_neg.json")

MONKEY_CLASS = 0   # class_dict: monkey = 0


def extract_frame(video_path, frame_number, out_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(out_path, frame)
    return ret


def load_tracking(video_name):
    """Load <video>_tracking.txt and return tracking[frame][class] = bbox"""
    track_file = os.path.join(
        LABEL_DIR, video_name, f"{video_name}_tracking.txt"
    )
    tracking = defaultdict(dict)

    with open(track_file, "r") as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            frame = int(parts[0])
            cls = int(parts[7])
            x, y, w, h = map(float, parts[2:6])
            bbox = [x, y, x + w, y + h]
            tracking[frame][cls] = bbox

    return tracking


def norm(v, maxv):
    return round(v / maxv, 6)


def add_negatives(input_json, output_json):
    print(f"\nProcessing {input_json} → {output_json}")

    with open(input_json, "r") as f:
        data = json.load(f)

    for video_entry in data:

        video_name = video_entry["path"].split("/")[-1]
        video_path = os.path.join(VIDEO_DIR, f"{video_name}.MP4")
        width = video_entry["width"]
        height = video_entry["height"]
        out_dir = os.path.join(IMG_DIR, video_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n--- Video: {video_name} ---")

        # Load tracking for monkey bbox retrieval
        tracking = load_tracking(video_name)

        # ------------------------
        # 1. Collect used positive frames
        # ------------------------
        positive_frames = []
        for fr in video_entry["frames"]:
            fname = os.path.basename(fr["path"])
            num = int(fname.split("frame")[-1].split(".")[0])
            positive_frames.append(num)

        if not positive_frames:
            print("No positive frames, skipping negative sampling.")
            continue

        min_used = min(positive_frames)
        max_used = max(positive_frames)

        # ------------------------
        # 2. Define negative frame range
        # ------------------------
        neg_start = max(0, min_used)
        neg_end = max_used + 1000

        closed = set(positive_frames)
        candidates = [
            f for f in range(neg_start, neg_end + 1)
            if f not in closed
        ]
        if not candidates:
            print("No valid candidate frames for negatives.")
            continue

        # ------------------------
        # 3. Sample negatives
        # ------------------------
        num_neg = max(1, len(positive_frames) // 4)
        random.shuffle(candidates)

        neg_frames = []
        for f in candidates:
            if f in tracking and MONKEY_CLASS in tracking[f]:
                neg_frames.append(f)
            if len(neg_frames) >= num_neg:
                break

        print(f"Positives: {len(positive_frames)}  →  negatives: {len(neg_frames)}")

        # ------------------------
        # 4. Process negative samples
        # ------------------------
        for frame_no in neg_frames:
            bbox = tracking[frame_no][MONKEY_CLASS]
            x1, y1, x2, y2 = bbox
            bbox_norm = [
                norm(x1, width),
                norm(y1, height),
                norm(x2, width),
                norm(y2, height),
            ]

            img_path = f"{out_dir}/{video_name}_frame{frame_no}.jpg"

            extract_frame(video_path, frame_no, img_path)

            video_entry["frames"].append({
                "path": img_path,
                "heads": [
                    {
                        "bbox": bbox,
                        "bbox_norm": bbox_norm,
                        "gazex": [-1.0],
                        "gazex_norm": [-1.0],
                        "gazey": [-1.0],
                        "gazey_norm": [-1.0],
                        "inout": 0
                    }
                ]
            })

        # Sort all frames by frame number
        video_entry["frames"].sort(
            key=lambda fr: int(os.path.basename(fr["path"]).split("frame")[1].split(".")[0])
        )

    # ------------------------
    # 5. Save updated JSON
    # ------------------------
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {output_json}")


def main():
    add_negatives(TRAIN_JSON, OUT_TRAIN_JSON)
    add_negatives(VAL_JSON, OUT_VAL_JSON)


if __name__ == "__main__":
    main()
