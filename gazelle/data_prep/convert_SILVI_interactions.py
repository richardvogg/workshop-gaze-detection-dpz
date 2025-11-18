import os
import json
import csv
import cv2
from collections import defaultdict

CLASS_DICT = {
    0: "monkey",
    1: "snake",
    2: "hand",
    3: "snakecube",
    4: "popcorn",
    5: "peanut"
}

INV_CLASS_DICT = {v: k for k, v in CLASS_DICT.items()}

BASE_PATH = "/user/vogg/u24033/.project/dir.project/Richard_Vogg/data/Rocamadour"
INPUT_DIR = os.path.join(BASE_PATH, "interaction_labels")
VIDEO_DIR = os.path.join(BASE_PATH, "videos")
OUTPUT_JSON = os.path.join(INPUT_DIR, "all_preprocessed.json")
OUTPUT_FRAMES = os.path.join(BASE_PATH, "interaction_images")
INTERACTION = "looking_at"

os.makedirs(OUTPUT_FRAMES, exist_ok=True)


def load_tracking_file(path):
    """Load tracking.txt → {frame → {class → bbox}}"""
    tracking = defaultdict(dict)

    with open(path, "r") as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            frame = int(parts[0])
            cls = int(parts[7])
            x, y, w, h = map(float, parts[2:6])
            bbox = [x, y, x + w, y + h]
            tracking[frame][cls] = bbox

    return tracking


def load_behaviors(path):
    """Load behaviors.csv → list of behaviors"""
    behaviors = []

    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            behaviors.append({
                "subject": row["Subject"],
                "action": row["Action"],
                "target": row["Target"],
                "start": int(row["StartFrame"]),
                "end": int(row["EndFrame"])
            })
    return behaviors


def bbox_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def norm(v, maxv):
    return round(v / maxv, 6)


def extract_frame(video_path, frame_number, out_path):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(out_path, frame)
    return ret


def main():
    result = []

    for video_name in sorted(os.listdir(INPUT_DIR)):
        folder = os.path.join(INPUT_DIR, video_name)
        if not os.path.isdir(folder):
            continue

        print(f"Processing {video_name}...")

        behavior_file = os.path.join(folder, f"{video_name}_behaviors.csv")
        tracking_file = os.path.join(folder, f"{video_name}_tracking.txt")
        video_path = os.path.join(VIDEO_DIR, f"{video_name}.MP4")

        if not os.path.exists(behavior_file) or not os.path.exists(tracking_file):
            print(f"Missing files for {video_name}, skipping.")
            continue

        # Load files
        behaviors = load_behaviors(behavior_file)
        tracking = load_tracking_file(tracking_file)

        # Load video metadata
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Prepare output folder
        out_dir = os.path.join(OUTPUT_FRAMES, video_name)
        os.makedirs(out_dir, exist_ok=True)

        
        frames_out = []
        used_frames = set()     # ensure each frame is processed once
        pending_heads = {}      # frame_no → list of heads for that frame

        for b in behaviors:
            subject = b["subject"]
            target = b["target"]
            action = b["action"]
            subj_class = INV_CLASS_DICT[subject]
            targ_class = INV_CLASS_DICT.get(target, None)

            for frame_no in range(b["start"], b["end"] + 1):

                # Skip if we don’t have tracking data for this frame
                if frame_no not in tracking:
                    continue

                frame_data = tracking[frame_no]

                # Skip if the subject is not tracked in this frame
                if subj_class not in frame_data:
                    continue

                bbox = frame_data[subj_class]
                x1, y1, x2, y2 = bbox
                bbox_norm = [
                    norm(x1, width),
                    norm(y1, height),
                    norm(x2, width),
                    norm(y2, height),
                ]

                # Looking-at behavior → target determines gaze
                if action == INTERACTION and targ_class in frame_data:
                    target_bbox = frame_data[targ_class]
                    cx, cy = bbox_center(target_bbox)

                    head = {
                        "bbox": bbox,
                        "bbox_norm": bbox_norm,
                        "gazex": [cx],
                        "gazex_norm": [norm(cx, width)],
                        "gazey": [cy],
                        "gazey_norm": [norm(cy, height)],
                        "inout": 1
                    }
                else:
                    continue

                # Add head to pending heads for this frame
                if frame_no not in pending_heads:
                    pending_heads[frame_no] = []
                pending_heads[frame_no].append(head)


        # ------------------------------------------------------
        # Now convert pending_heads into frame entries
        # ------------------------------------------------------
        for frame_no in sorted(pending_heads.keys()):

            img_path = f"{out_dir}/{video_name}_frame{frame_no}.jpg"

            # Save frame once
            if frame_no not in used_frames:
                extract_frame(video_path, frame_no, img_path)
                used_frames.add(frame_no)

            frames_out.append({
                "path": img_path,
                "heads": pending_heads[frame_no]
            })

        # Add to result list
        result.append({
            "path": f"{OUTPUT_FRAMES}/{video_name}",
            "width": width,
            "height": height,
            "frames": frames_out
        })

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("Done! JSON saved.")


if __name__ == "__main__":
    main()
