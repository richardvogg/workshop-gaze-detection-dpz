import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2

from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/lemurattentiontarget_small")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument("--ckpt_path", type=str, default="./experiments/lemurs_smallnew_ViTB14_largeinput/2025-08-05_17-23-20/epoch_0.pt") #"./experiments/train_gazelle_vitb_lemurs/2025-06-17_18-09-57/epoch_7.pt"
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()



class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.sequences = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        path = frame['path']
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, path, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)
    
def collate(batch):
    images, path, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(path), list(bboxes), list(gazex), list(gazey), list(inout)


@torch.no_grad()
def main(vis = False):
    if vis:
        os.makedirs("output_images", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    dataset = VideoAttentionTarget(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    running_param = 0

    for k, (images, paths, bboxes, gazex, gazey, inout) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})



        # eval each instance (head)
        for i in range(images.shape[0]): # per image

            for j in range(len(bboxes[i])): # per individual

                if vis & (k%100==0):
                    # Parameters
                    hm_alpha = 0.5  # Change this for stronger or weaker heatmap visibility

                    # Convert image tensor [3, 448, 448] to RGB numpy array
                    img_tensor = images[i].cpu()

                    img_np = img_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
                    img_np = (img_np - np.min(img_np)) / (np.ptp(img_np) + 1e-8)  # Normalize to [0,1]
                    img_np = cv2.resize(img_np, (448, 252))  # Resize to width=448, height=300
                    # Resize heatmap [1, 64, 64] -> [1, 448, 448]
                    heatmap_tensor = preds['heatmap'][i][j].unsqueeze(0).unsqueeze(0)  # shape: [1,1,64,64]
                    heatmap_resized = F.interpolate(heatmap_tensor, size=(448, 448), mode='bilinear', align_corners=False)
                    heatmap_resized = heatmap_resized.squeeze().cpu().numpy()  # shape: [448, 448]

                    # Normalize heatmap to [0, 1]
                    #heatmap_norm = (heatmap_resized - np.min(heatmap_resized)) / (np.ptp(heatmap_resized) + 1e-8)
                    heatmap_norm = cv2.resize(heatmap_resized, (448, 252))  # Resize to match image dimensions
                    # Create alpha mask
                    alpha_mask = heatmap_norm.copy()
                    alpha_mask[alpha_mask < 0.05] = 0  # Fully transparent where heat is low
                    # Scale the remaining values to [0, hm_alpha]
                    alpha_mask = np.clip(alpha_mask, 0, 1) * hm_alpha
                    alpha_mask = cv2.resize(alpha_mask, (448, 252))  # Resize to match image dimensions


                    # Plot the image with the bounding boxes drawn on top
                    fig, ax = plt.subplots(figsize=(20, 13))
                    ax.imshow(img_np)
                    
                    
                    ax.imshow(heatmap_norm, cmap='jet', alpha=alpha_mask)  # Overlay heatmap


                    

                    if preds['inout'][i][j].item() > 0.5:
                        # Find the maximum of heatmap_norm and plot a yellow dot
                        max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
                        ax.plot(max_x, max_y, 'yo', markersize=10)  
                    elif preds['inout'][i][j].item() > 0.2:
                        # Find the maximum of heatmap_norm and plot a yellow dot
                        max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
                        ax.plot(max_x, max_y, 'bo', markersize=10)  

                    x1, y1, x2, y2 = bboxes[i][j]
                    rect = plt.Rectangle((x1 * 448, y1 * 252), (x2 - x1) * 448, (y2 - y1) * 252, linewidth=2, edgecolor='yellow', facecolor='none')
                    ax.add_patch(rect)
                    #ax.axis('off')
                    #plt.tight_layout()
                    #plt.savefig(f"output_images/bboxes_{running_param}.png", bbox_inches='tight', pad_inches=0)
                    #plt.close()

                    # Plot and overlay
                    #fig, ax = plt.subplots(figsize=(20, 13))
                    
                    #ax.imshow(img_np)  # Show base image

                    
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"output_images/overlay_{running_param}.png", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    running_param += 1

                if inout[i][j] == 1: # in frame
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    
    print("AUC: {}".format(np.array(aucs).mean()))
    print("Avg L2: {}".format(np.array(l2s).mean()))
    print("Inout AP: {}".format(average_precision_score(inout_gts, inout_preds)))

        
if __name__ == "__main__":
    main(vis = True)