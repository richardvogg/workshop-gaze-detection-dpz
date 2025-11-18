import json
import os
import random

BASE_PATH = "/user/vogg/u24033/.project/dir.project/Richard_Vogg/data/Rocamadour/interaction_labels/"
INPUT_JSON = "all_preprocessed.json"
TRAIN_JSON = "train_preprocessed.json"
VAL_JSON = "val_preprocessed.json"

VAL_RATIO = 0.20 

def main():
    # Load all videos
    with open(os.path.join(BASE_PATH, INPUT_JSON), "r") as f:
        data = json.load(f)

    # Shuffle videos randomly
    random.shuffle(data)

    # Compute split index
    n_total = len(data)
    n_val = max(1, int(n_total * VAL_RATIO))

    val_data = data[:n_val]
    train_data = data[n_val:]

    # Save files
    with open(os.path.join(BASE_PATH, VAL_JSON), "w") as f:
        json.dump(val_data, f, indent=2)

    with open(os.path.join(BASE_PATH, TRAIN_JSON), "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"Total videos: {n_total}")
    print(f"Validation videos: {len(val_data)} → saved to {VAL_JSON}")
    print(f"Training videos:   {len(train_data)} → saved to {TRAIN_JSON}")


if __name__ == "__main__":
    main()