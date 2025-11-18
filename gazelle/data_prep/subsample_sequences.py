import os
import re
import random
import shutil
from pathlib import Path
from itertools import groupby

def extract_frame_number(line):
    """Extract numeric part from something like frame2370.jpg"""
    match = re.search(r'frame(\d+)\.jpg', line)
    return int(match.group(1)) if match else None

def extract_frame_name(line):
    """Extract the frame filename itself (e.g. frame2370.jpg)"""
    return line.split(",")[0]

def group_consecutive_frames(frames):
    """Group frame numbers into consecutive sequences"""
    frames = sorted(set(frames))
    groups = []
    for _, group in groupby(enumerate(frames), lambda x: x[0] - x[1]):
        groups.append([g[1] for g in group])
    return groups

def subsample_txt_and_copy_images(input_path, output_path, images_root, output_images_root, max_frames=3):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Map frame -> list of lines
    frame_to_lines = {}
    frame_to_names = {}
    for line in lines:
        frame_num = extract_frame_number(line)
        frame_name = extract_frame_name(line)
        if frame_num is not None and frame_name is not None:
            frame_to_lines.setdefault(frame_num, []).append(line)
            frame_to_names[frame_num] = frame_name

    frames = sorted(frame_to_lines.keys())
    groups = group_consecutive_frames(frames)

    sampled_lines = []
    sampled_frame_names = []

    for group in groups:
        sampled_frames = random.sample(group, min(max_frames, len(group)))
        sampled_frames.sort()  # keep frames in order
        for fnum in sampled_frames:
            sampled_lines.extend(frame_to_lines[fnum])
            sampled_frame_names.append(frame_to_names[fnum])

    # Write the subsampled .txt
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out:
        out.writelines(sampled_lines)

    # Copy the corresponding images
    for frame_name in set(sampled_frame_names):
        src_img = Path(images_root) / frame_name
        dst_img = Path(output_images_root) / frame_name
        if src_img.exists():
            os.makedirs(dst_img.parent, exist_ok=True)
            shutil.copy2(src_img, dst_img)
        else:
            print(f"⚠️ Warning: missing image {src_img}")

def process_all_roots(roots, output_root, max_frames=3):
    for root in roots:
        ann_dir = Path(root) / "annotations"
        img_dir = Path(root) / "images"
        for split in ["train", "test"]:
            for txt_path in ann_dir.glob(f"{split}/**/*.txt"):
                relative_path = txt_path.relative_to(ann_dir)
                output_txt_path = Path(output_root) / "annotations" / relative_path

                # Find corresponding image subfolder
                # e.g. annotations/train/A_e1_c1/file.txt → images/A_e1_c1/
                # image_subdir = everything after the split (may be nested)
                rel_parent = relative_path.parent  # e.g. train/A_e1_c1 or train/A/e1
                parts_after_split = rel_parent.parts[1:] if len(rel_parent.parts) > 1 else []
                image_subdir = Path(*parts_after_split) if parts_after_split else Path()
                images_root = img_dir / image_subdir
                output_images_root = Path(output_root) / "images" / image_subdir

                subsample_txt_and_copy_images(
                    txt_path,
                    output_txt_path,
                    images_root,
                    output_images_root,
                    max_frames=max_frames
                )


if __name__ == "__main__":
    # Example usage
    #roots = [
    #    "data/lemurattentiontarget_joana1_clean",
    #    "data/lemurattentiontarget_new_clean",
    #    "data/lemurattentiontarget_clean"
    #]
    roots = ["data/lemurattentiontarget_test"]
    output_root = "data/o_lemurattentiontarget_mini"
    process_all_roots(roots, output_root)
