from pathlib import Path

# ===== Global Parameters =====
DATA_ROOT = Path("../.project/dir.project/workshop-data/")       # main dataset folder
IMAGE_FOLDER_NAME = "tracking_labels"               # subfolder containing the images
OUTPUT_FILE_PATH = "PriMAT-tracking/src/data/rocamadour.train"         # name of the output .train file
# =============================

def generate_train_file():
    image_folder = DATA_ROOT / IMAGE_FOLDER_NAME
    output_file = Path(OUTPUT_FILE_PATH)
    # Clear file if it exists
    output_file.write_text("")

    # List files
    for file_path in sorted(image_folder.iterdir()):
        if file_path.suffix.lower() in [".txt"]:
            line = f"{IMAGE_FOLDER_NAME.replace('labels', 'images')}/{file_path.name.replace('.txt', '.png')}\n"
            with open(output_file, "a") as f:
                f.write(line)

    print(f"Finished writing {output_file}")

if __name__ == "__main__":
    generate_train_file()
