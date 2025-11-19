import argparse
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
#import wandb
import torchvision.transforms as transforms

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument('--init_ckpt', type=str, default='../.project/dir.project/workshop-models/gazelle_dinov2_vitb14_inout.pt', help='checkpoint for initialization (trained on GazeFollow)')
parser.add_argument('--data_path', type=str, default='../.project/dir.project/workshop-data')
parser.add_argument('--dataset_list', type=list, default=['interaction_labels'])
parser.add_argument('--frame_sample_every', type=int, default=3)
parser.add_argument('--ckpt_save_dir', type=str, default='gazelle/experiments')
parser.add_argument('--wandb_project', type=str, default='gazelle')
parser.add_argument('--exp_name', type=str, default='macaque_gaze_with_neg')
parser.add_argument('--log_iter', type=int, default=20, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--inout_loss_lambda', type=float, default=10)
parser.add_argument('--lr_non_inout', type=float, default=1e-4)
parser.add_argument('--lr_inout', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=6)
args = parser.parse_args()


def main():
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir)


    model, transform = get_gazelle_model(args.model)
    resize_transform = next(t for t in transform.transforms if isinstance(t, transforms.Resize))
    new_size = tuple(s // 14 for s in resize_transform.size)
    print("Initializing from {}".format(args.init_ckpt))
    a = torch.load(args.init_ckpt, weights_only=True)
    # Extrapolate 'pos_embed' from 32x32 to new_size
    pos_embed = a['pos_embed']  # shape: (256, 32, 32)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False
    ).squeeze(0)
    a['pos_embed'] = pos_embed
    #{k: v for k,v in a.items() if k != 'pos_embed'}
    model.load_gazelle_state_dict(a) # initializing from ckpt without inout head

    model.cuda()

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")



    # Load and concatenate datasets from args.dataset_list
    train_datasets = [
        GazeDataset('videoattentiontarget', os.path.join(args.data_path, dataset_name), 'train', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
        for dataset_name in args.dataset_list
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    eval_datasets = [
        GazeDataset('videoattentiontarget', os.path.join(args.data_path, dataset_name), 'val', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
        for dataset_name in args.dataset_list
    ]
    eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
    #eval_dataset = GazeDataset('videoattentiontarget', args.data_path, 'test', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    heatmap_loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()
    param_groups = [
        {'params': [param for name, param in model.named_parameters() if "inout" in name], 'lr': args.lr_inout},
        {'params': [param for name, param in model.named_parameters() if "inout" not in name], 'lr': args.lr_non_inout}
    ]
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            heatmap_loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].cuda())
            inout_loss = inout_loss_fn(inout_preds, inout.float().cuda())
            loss = heatmap_loss + args.inout_loss_lambda * inout_loss
            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                #wandb.log({
                #    "train/loss": loss.item(),
                #    "train/heatmap_loss": heatmap_loss.item(),
                #    "train/inout_loss": inout_loss.item()
                #})
                print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH
        print("Running evaluation")
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        #wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "eval/inout_ap": epoch_inout_ap, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}".format(epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)))
        # Write results to a txt file
        with open(os.path.join(exp_dir, "eval_results.txt"), "a") as f:
            f.write("EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}\n".format(
            epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)
            ))



if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()