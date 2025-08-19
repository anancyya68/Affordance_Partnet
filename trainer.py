from __future__ import print_function, division
from pathlib import Path
import json
import h5py
import os
import numpy as np
import plotly.graph_objects as go
import scipy.spatial.distance
import math
import random
from torchvision import transforms, utils
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *
from models import *
from tqdm import tqdm
import provider
import datetime
from torch.utils.data import DataLoader, random_split
from pathlib import Path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, pts,gt_label,gt_mask,gt_valid, gt_other_mask_total,transform=None):
        
        self.pts = pts
        self.gt_label = gt_label
        self.gt_mask = gt_mask
        self.gt_valid=gt_valid
        self.gt_other_mask_total = gt_other_mask_total

    def __len__(self):
        return self.pts.shape[0]

    def __getitem__(self, idx):
        #if not self.valid:
          #  theta = random.random()*360
         #   image2 = utils.RandRotation_z()(utils.RandomNoise()(image2))
        return {'image': np.array(self.pts[idx], dtype="float32"), 'category': self.gt_label[idx].astype(int) , 'masks': self.gt_mask[idx], 'valid':np.array(self.gt_valid[idx])}


def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        gt_label = fin['gt_label'][:]
        gt_mask = fin['gt_mask'][:]
        gt_valid = fin['gt_valid'][:]
        gt_other_mask = fin['gt_other_mask'][:]
        return pts, gt_label, gt_mask, gt_valid, gt_other_mask

def load_data(fn):
    cur_json_fn = fn.replace('.h5', '.json')
    record = load_json(cur_json_fn)
    pts, gt_label, gt_mask, gt_valid, gt_other_mask = load_h5(fn)
    return pts, gt_label, gt_mask, gt_valid, gt_other_mask, record

def load_label_map(txt_fn):
    """
    读取 level-1.txt 文件，将原始类别ID映射为连续标签，
    返回两个字典：
      original_id -> new_id
      new_id -> part_name
    """
    label_map = {}
    part_names = []
    with open(txt_fn, 'r') as f:
        lines = f.readlines()
    # txt 文件格式类似：
    # 2 bag/bag_body leaf
    # 3 bag/handle leaf
    # 4 bag/shoulder_strap leaf
    # 其中数字是原始类别标签
    for new_id, line in enumerate(lines):
        original_id = int(line.strip().split()[0])
        part_name = line.strip().split()[1]
        label_map[original_id] = new_id
        part_names.append(part_name)
    return label_map, part_names


def map_labels(gt_label, label_map):
    """
    gt_label: numpy array, 原始标签，形状(样本数, 点数)
    label_map: dict，原始标签->连续标签映射
    """
    new_label = np.zeros_like(gt_label)
    for orig_id, new_id in label_map.items():
        new_label[gt_label == orig_id] = new_id
    return new_label


def make_dir():
    current = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    Folder = os.path.join('saved_model', current)
    if not os.path.exists(Folder):
        os.makedirs(Folder)
    return Folder



'''
def build_dataloaders(categories,batch_size=4, root_dir='data/sem_seg_h5', label_id_dir='./data/after_merging_label_ids'):
    all_pts = []
    all_labels = []
    all_masks = []
    all_valids = []
    all_other_masks = []
    all_part_names = []
    all_label_maps = {}
    root_dir = Path(root_dir)
    label_id_dir = Path(label_id_dir)

    for cat in categories:
        print(f"Processing category {cat}...")
        data_dir = root_dir / cat
        label_file = label_id_dir / f"{cat.split('-')[0]}-level-1.txt"

        label_map, part_names = load_label_map(label_file)
        all_label_maps[cat] = {}
        all_label_maps[cat]['label'] = label_map
        all_label_maps[cat]['names'] = part_names
        all_part_names.extend(part_names)


        train_files = [f for f in os.listdir(data_dir) if f.startswith('train-') and f.endswith('.h5')]
        for h5_fn in train_files:
            full_path = data_dir / h5_fn
            pts, gt_label, gt_mask, gt_valid, gt_other_mask, _ = load_data(str(full_path))
            gt_label = map_labels(gt_label, label_map)

            all_pts.append(pts)
            all_labels.append(gt_label)
            all_masks.append(gt_mask)
            all_valids.append(gt_valid)
            all_other_masks.append(gt_other_mask)

    pts_total = np.concatenate(all_pts, axis=0)
    gt_label_total = np.concatenate(all_labels, axis=0)
    gt_mask_total = np.concatenate(all_masks, axis=0)
    gt_valid_total = np.concatenate(all_valids, axis=0)
    gt_other_mask_total = np.concatenate(all_other_masks, axis=0)

    print(f"Total pts shape: {pts_total.shape}")
    print(f"Total label shape: {gt_label_total.shape}")

    # 数据增强
    pts_total = provider.jitter_point_cloud(pts_total)
    pts_total = provider.shift_point_cloud(pts_total)
    pts_total = provider.random_scale_point_cloud(pts_total)
    pts_total = provider.rotate_perturbation_point_cloud(pts_total)

    dset = Data(pts_total, gt_label_total, gt_mask_total, gt_valid_total,gt_other_mask_total)
    train_num = int(len(dset) * 0.95)
    val_num = len(dset) - train_num

    train_dataset, val_dataset = random_split(dset, [train_num, val_num])
    val_dataset.dataset.valid = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    num_classes = len(all_part_names)
    return num_classes, train_loader, val_loader,all_label_maps
'''


def build_dataloaders(categories, batch_size=4, root_dir='data/sem_seg_h5', label_id_dir='./data/after_merging_label_ids'):
    all_pts, all_labels = [], []
    all_masks, all_valids, all_other_masks = [], [], []
    all_part_names = []
    all_label_maps = {}

    global_part_id = 1  # Start from 1, 0 reserved for background
    part_name_to_global_id = {}  # For consistent ID mapping

    for cat in categories:
        print(f"Processing category {cat}...")
        data_dir = Path(root_dir) / cat
        label_file = Path(label_id_dir) / f"{cat.split('-')[0]}-level-1.txt"

        label_map = {}
        part_names = []

        # 读取对应的 label 文件并构建局部->全局映射
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                raw_label_id = int(parts[0])
                part_name = parts[1]

                # 跳过背景类（但记录局部ID）
                if raw_label_id == 0:
                    label_map[raw_label_id] = 0
                    continue

                # 建立全局唯一 ID 映射
                if part_name not in part_name_to_global_id:
                    part_name_to_global_id[part_name] = global_part_id
                    global_part_id += 1

                label_map[raw_label_id] = part_name_to_global_id[part_name]
                part_names.append(part_name)

        all_label_maps[cat] = {
            "label": label_map,
            "names": part_names
        }
        all_part_names.extend(part_names)

        # 遍历该类所有h5文件
        train_files = [f for f in os.listdir(data_dir) if f.startswith('train-') and f.endswith('.h5')]
        for h5_fn in train_files:
            full_path = data_dir / h5_fn
            pts, gt_label, gt_mask, gt_valid, gt_other_mask, _ = load_data(str(full_path))

            # 标签映射：每个原始标签 -> 全局唯一 ID（或背景0）
            new_label = np.vectorize(lambda x: label_map.get(x, 0))(gt_label)

            all_pts.append(pts)
            all_labels.append(new_label)
            all_masks.append(gt_mask)
            all_valids.append(gt_valid)
            all_other_masks.append(gt_other_mask)

    # 拼接数据
    pts_total = np.concatenate(all_pts, axis=0)
    gt_label_total = np.concatenate(all_labels, axis=0)
    gt_mask_total = np.concatenate(all_masks, axis=0)
    gt_valid_total = np.concatenate(all_valids, axis=0)
    gt_other_mask_total = np.concatenate(all_other_masks, axis=0)

    print(f"Total pts shape: {pts_total.shape}")
    print(f"Total label shape: {gt_label_total.shape}")
    print(f"Total classes (excluding background): {global_part_id - 1}")

    # 数据增强（可选）
    pts_total = provider.jitter_point_cloud(pts_total)
    pts_total = provider.shift_point_cloud(pts_total)
    pts_total = provider.random_scale_point_cloud(pts_total)
    pts_total = provider.rotate_perturbation_point_cloud(pts_total)

    dset = Data(pts_total, gt_label_total, gt_mask_total, gt_valid_total, gt_other_mask_total)
    train_num = int(len(dset) * 0.95)
    val_num = len(dset) - train_num

    train_dataset, val_dataset = random_split(dset, [train_num, val_num])
    val_dataset.dataset.valid = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_classes = global_part_id  
    return num_classes, train_loader, val_loader, all_label_maps



def train(model,train_loader, val_loader, Folder, epochs=45, save=True):
    pointnet = model
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
    for epoch in tqdm(range(epochs)): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data['image'].to(device)
            labels = data['category'].to(device)
            gt_mask_pl = data['masks'].to(device)
            gt_valid_pl = data['valid'].to(device)
            #gt_other_mask = data['other_masks'].to(device)
            optimizer.zero_grad()
            #outputs ,mask_pred,end_points, other_mask_pred,pred_conf, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            outputs ,mask_pred,end_points, m3x3, m64x64,point_fea = pointnet(inputs.transpose(1,2))
            seg_loss ,end_points = get_seg_loss(outputs, labels, m3x3, m64x64,end_points)
            ins_loss, end_points = get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points)
            #l2_loss_norm , end_points = get_l21_norm(mask_pred, other_mask_pred)
            #conf_loss, end_points= get_conf_loss(pred_conf, gt_valid_pl, end_points, 200):
            #other_ins_loss,end_points= get_other_ins_loss(other_mask_pred, gt_other_mask):
            loss=ins_loss+seg_loss
            loss.backward()
            optimizer.step()
            
            seg_pred_id = torch.argmax(outputs, dim=1)
            acc=seg_pred_id == labels
            acc=acc.float()
            seg_acc = torch.mean(acc)
            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 10 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #     (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
                    print('[Train Epoch %d, Batch %d] Loss: %f = %f seg_loss (Seg Acc: %f) + %f (ins_loss)' \
                    % (epoch+1, i+1, loss, seg_loss, seg_acc, ins_loss))
        pointnet.eval()
        correct = total = 0
        mcorrect = mtotal = 0
        # validation
        if val_loader:
            with torch.no_grad():
                for i,data in enumerate(val_loader,0):
                    inputs,labels,gt_mask_pl,gt_valid_pl = data['image'].to(device),data['category'].to(device),data['masks'].to(device),data['valid'].to(device)
                    outputs ,mask_pred,end_points, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0) * labels.size(1) 
                    correct += (predicted == labels).sum().item()

                    seg_pred_id = torch.argmax(outputs, dim=1)
                    acc=seg_pred_id == labels
                    seg_acc = torch.mean(acc.float())
            val_acc = 100 * correct / total
            print('Valid accuracy: label %d %%' % val_acc)
            print('[Validation Epoch %03d, Batch %03d] Loss: %f = %f seg_loss (Seg Acc: %f) + %f (ins_loss)' \
                   % (epoch+1, i+1, loss, seg_loss, seg_acc, ins_loss))
        # save the model
        if save:
            save_path = os.path.join(Folder, f"total_loss{epoch}_{val_acc:.4f}.pth")
            torch.save(pointnet.state_dict(), save_path)
            
def compute_loss(logits, labels):
    """
    logits: (B, N, num_classes)
    labels: (B, N) 
    """
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                           labels.view(-1))
    return loss
   
def test(model,val_loader,date_file,para_name,batch_id,idx):
    pointnet  = model.to(device)
    model_path = os.path.join('./saved_model',date_file,para_name)
    pointnet.load_state_dict(torch.load(model_path, map_location=device))
    pointnet.eval()
    for i, batch in enumerate(val_loader):
        if i == batch_id:
            batch = {k: v.to(device) for k, v in batch.items()}
            break
    else:
        raise IndexError(f"batch_id {batch_id} out of range for val_loader with length {len(val_loader)}")
    pred = pointnet(batch['image'].transpose(1,2))
    pred_res = torch.argmax(pred[0], dim=1)  # shape: [B, N]
    pred_np = pred_res[idx].cpu().numpy()
    true_np = batch['category'][idx].cpu().numpy()
    acc = (pred_np == true_np)
    resulting_acc = np.sum(acc) / len(acc)
    print(f'Acc of {idx} sample:',resulting_acc)

    # plot true label
    x, y, z = batch['image'][idx].cpu().numpy().T
    c = batch['category'][idx].cpu().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=30,color=c, colorscale='Viridis', opacity=1.0 ))])
    fig.update_traces(marker=dict(size=2, line=dict(width=2,color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.write_html(os.path.join('./results',"true_{}_{}.html".format(batch_id,idx)))
    
    # plot pred label
    x, y, z = batch['image'][idx].cpu().numpy().T
    pred_res = torch.argmax(pred[0], dim=1) 
    c = pred_res[idx].cpu().numpy()
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,  mode='markers', marker=dict( size=30, color=c, opacity=1.0 ))])
    fig.update_traces(marker=dict(size=2,line=dict(width=2, color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.write_html(os.path.join('./results',"pred_{}_{}.html".format(batch_id,idx)))


## text-point 
def compute_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.shape[-1]),
                           labels.view(-1))

def train_TextPoint(model, train_loader, val_loader, all_names, tokenizer, 
                num_epochs=50, lr=1e-4, device="cuda", save_dir="./checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        # --------- Training ---------
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            points = batch['image'].to(device)
            labels = batch['category'].to(device)

            logits, _, _ = model(points, all_names, tokenizer)
            loss = compute_loss(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # --------- Validation ---------
        if val_loader is not None:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    points = batch['image'].to(device)
                    labels = batch['category'].to(device)

                    logits, _, _ = model(points, all_names, tokenizer)
                    loss = compute_loss(logits, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # --------- Save Best ---------
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(save_dir, f"best_model_epoch{epoch+1}_acc{val_acc:.4f}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at {best_model_path}")
    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}, saved at {best_model_path}")

def predict_mask(model, points, part_name, tokenizer, all_names, device="cuda"):
    """
    Given point and part name --> mask
    Args:
        model: PointTextSegModel
        points: (1, N, 3) 
        part_name: str, part name
        tokenizer: CLIP tokenizer
    Returns:
        mask: (N,)  (0/1)
        probs: (N,) p
    """
    
    model.eval()
    points = points.to(device)
    with torch.no_grad():
        # logits: (1, N, num_classes)
        logits, _, _ = model(points, all_names, tokenizer)

        if part_name not in all_names:
            raise ValueError(f"{part_name} not in list")
        class_idx = all_names.index(part_name)
        probs = torch.softmax(logits, dim=-1)[0, :, class_idx]  # (N,)

        mask = (probs > 0.5).cpu().numpy()

    return mask, probs.cpu().numpy()

def plot_pointcloud(points,labels,mask, save_path, prefix="sample", part_name=""):
    os.makedirs(save_path, exist_ok=True)
    if not isinstance(points, (list, tuple)):
        points = points.cpu().numpy() if hasattr(points, "cpu") else points
    x, y, z = points.T
    # 1. true labels
    if labels is not None:
        labels = labels.cpu().numpy() if hasattr(labels, "cpu") else labels
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=labels, colorscale='Viridis', opacity=1.0,
                        line=dict(width=2, color='DarkSlateGrey')) )])
        fig.update_layout(title="Ground Truth Labels")
        fig.write_html(os.path.join(save_path, f"{prefix}_true.html"))
    # 2.  predicted mask
    if mask is not None:
        mask = mask.cpu().numpy() if hasattr(mask, "cpu") else mask
        colors = np.where(mask > 0.5, 1, 0)   
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=colors, colorscale='Jet', opacity=1.0)
        )])
        fig.update_layout(title=f"Predicted Mask: {part_name}")
        fig.write_html(os.path.join(save_path, f"{prefix}_mask_{part_name.replace('/', '_')}.html"))


if __name__ == "__main__":
    print('start:')

    ### Original code for point segmentation
    '''
    categories = ['Bag-1', 'Chair-1','Bed-1','Clock-1','StorageFurniture-1']
    #categories = ['StorageFurniture-1']
    Folder = make_dir()
    num_classes, train_loader, val_loader,all_label_maps = build_dataloaders(categories)
    print(num_classes)
    print(all_label_maps)
    for batch in train_loader:
        data, label = batch  
        print("Data shape:", data.shape)
        print("Label:", label)
        print("Single sample data:", data[0])
        print("Single sample label:", label[0])
        break  

    file_path = os.path.join(Folder,"label_map.json")
    with open(file_path, 'w') as f:
        json.dump(all_label_maps, f, indent=2)

    pointnet = PointNetSeg(classes=num_classes, num_instances=200)
    pointnet.to(device)

    train(pointnet, train_loader, val_loader,Folder=Folder, epochs=50, save=True)
    
    date_file = '2025_07_05-10_05_50'
    para_name = 'total_loss37_80.5221.pth'
    batch_ids = [5,6,2,8,12]
    ids = [0,1,2,3]
    for batch_id in batch_ids:
         for idx in ids:
            test(pointnet,val_loader,date_file,para_name, batch_id,idx)
   
    for i, batch in enumerate(val_loader):
        if i == 0:
            batch = {k: v.to(device) for k, v in batch.items()}
            break
    else:
        raise IndexError(f"batch_id {0} out of range for val_loader with length {len(val_loader)}")
    print(batch)
    print(batch.keys())
    print( np.unique(batch['category'].cpu().numpy()))
    '''
   
   
   ### text and point align
    #categories = ['Bag-1', 'Chair-1','Bed-1','Clock-1','StorageFurniture-1']
    categories = ['Chair-1']
    Folder = make_dir()
    num_classes, train_loader, val_loader,all_label_maps = build_dataloaders(categories,batch_size=4)
    print(num_classes)
    print(all_label_maps)
    
    file_path = os.path.join(Folder,"label_map.json")
    with open(file_path, 'w') as f:
        json.dump(all_label_maps, f, indent=2)


    all_names = []
    for cat in all_label_maps.values():
        all_names.extend(cat['names'])

    print("Total classes:", len(all_names))  
    print(all_names[:5])

    # text
    text_encoder = TextEncoderCLIP(proj_dim=128)
    text_proj = text_encoder(all_names)  
    print("Text embedding shape:", text_proj.shape)

    # point
    pointnet = PointNetSeg(classes=num_classes)  
    pointnet.to(device)
    text_encoder = TextEncoderCLIP().text_encoder 
    tokenizer = TextEncoderCLIP().tokenizer

    # text to point
    model = PointTextSegModel(pointnet, text_encoder, proj_dim=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 一个 batch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_TextPoint(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_names=all_names,
        tokenizer=tokenizer,
        num_epochs=1000,
        lr=1e-4,
        device=device,
        save_dir=Folder )

   
    ## test
    '''
    model_path =  "./saved_model/2025_08_19-14_56_01"
    best_model_path = os.path.join(model_path ,"best_model_epoch45_acc0.2727.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    batch = next(iter(val_loader))
    points = batch['image'][0].unsqueeze(0).to(device)  # (1, N, 3)
    labels = batch['category'][0]  # (N,)  ground-truth 
    part_name = "chair/chair_head"
    mask, probs = predict_mask(model, points, part_name, tokenizer, all_names, device)
    print("mask shape:", mask.shape)   # (N,)
    print("prob:", probs[:10])
    print("mask min:", mask.min(), "max:", mask.max(), "unique:", np.unique(mask))
    save_path = "./results"
    plot_pointcloud(
        points=batch['image'][0], 
        labels=labels,             
        mask=mask,                 
        save_path= model_path,
        prefix="sample0",        
        part_name=part_name )
    '''










