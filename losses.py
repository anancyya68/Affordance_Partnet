# @File : losses.py
# @Time : 2025/7/3 15:20
# @Author : wyp
# @Purpose :
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.dataset import random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_seg_loss (outputs, labels, m3x3, m128x128, end_points, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    #id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    #id128x128 = torch.eye(128, requires_grad=True).repeat(bs,1,1)
    #if outputs.is_cuda:
     #   id3x3=id3x3.cuda()
     #   id128x128=id128x128.cuda()
    #diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    #diff128x128 = id128x128-torch.bmm(m128x128,m128x128.transpose(1,2))
    logsoftmax = nn.LogSoftmax(dim=1)
    per_point_loss= criterion(logsoftmax(outputs), labels) #+ alpha * (torch.norm(diff3x3)+torch.norm(diff128x128)) / float(bs)
    end_points['per_point_seg_loss'] = per_point_loss
    per_shape_loss = torch.mean(per_point_loss, -1)
    end_points['per_shape_seg_loss'] = per_shape_loss
   # loss = torch.mean(per_shape_loss)
    return per_shape_loss, end_points

def get_other_ins_loss(other_mask_pred, gt_other_mask):
    """ Input:  other_mask_pred B x N
                gt_other_mask   B x N
    """
    #batch_size = other_mask_pred.get_shape()[0].value
    #num_point = other_mask_pred.get_shape()[1].value
    matching_score = torch.sum(torch.multiply(other_mask_pred, gt_other_mask), dim=-1)
    iou = torch.divide(matching_score, torch.sum(other_mask_pred, dim=-1) + torch.sum(gt_other_mask, dim=-1) - matching_score + 1e-8)
    loss = - torch.mean(iou)
    return loss

def get_l21_norm(mask_pred, other_mask_pred):
    eps = 1e-6
    num_points = other_mask_pred.shape[1]
    full_mask = torch.cat((mask_pred, torch.unsqueeze(other_mask_pred, dim=1)), dim=1) + eps
    per_shape_l21_norm = torch.norm(torch.norm(full_mask, p=2, dim=-1), p=1, dim=-1) / num_points
    loss = torch.mean(per_shape_l21_norm)
    return loss

def get_conf_loss(pred_conf, gt_conf, end_points, nmask=200):
    """ Input:  conf_pred       B x K
                gt_valid        B x K
    """
    # calculate conf loss as well
    batch_size = pred_conf.shape[0]
    iou_all = end_points['per_shape_all_iou']
    matching_idx = end_points['matching_idx']
    matching_idx_column = matching_idx[:, :, 1]
    idx=(matching_idx_column >= 0).nonzero(as_tuple=False)
    matching_idx_column=torch.cat((torch.unsqueeze(idx[:, 0].int(),dim=-1 ) , torch.reshape( matching_idx_column, (-1, 1)) ),1)

    all_indices = torch.reshape(matching_idx_column, [batch_size, nmask, 2])

    valid_idx = torch.where(torch.greater(gt_conf, 0.5))
    predicted_indices = all_indices[valid_idx[0].long(), valid_idx[1].long()]
    valid_iou = iou_all[valid_idx[0].long(), valid_idx[1].long()]

    target_conf = torch.zeros([batch_size, nmask], dtype = torch.float32).to(pred_conf.device)
    target_conf[predicted_indices[:, 0].long(), predicted_indices[:, 1].long()] = valid_iou
    #target_conf = torch.scatter(predicted_indices, valid_iou, torch.tensor([batch_size, nmask]))

    per_part_loss = (pred_conf - target_conf)**2

    target_pos_mask = torch.greater(target_conf, 0.1).type(torch.float32)
    target_neg_mask = 1 - target_pos_mask

    pos_per_shape_loss = torch.divide(torch.sum(target_pos_mask*per_part_loss, dim=-1),
                                      torch.clamp(torch.sum(target_pos_mask, dim=-1), min = 1e-6))
    neg_per_shape_loss = torch.divide(torch.sum(target_neg_mask * per_part_loss, dim=-1),
                                      torch.clamp(torch.sum(target_neg_mask, dim=-1), min=1e-6))

    per_shape_loss = pos_per_shape_loss + neg_per_shape_loss
    conf_loss = torch.mean(per_shape_loss)

    return conf_loss, end_points


def hungarian_matching(pred_x, gt_x, curnmasks):
    """ pred_x, gt_x: B x nmask x n_point
        curnmasks:
        return matcBhing_idx: B x nmask x 2 """

    curnmasks = torch.sum(curnmasks, dim=-1)
    batch_size = gt_x.shape[0]
    nmask = gt_x.shape[1]
    matching_score = torch.matmul(gt_x, torch.transpose(pred_x, 2, 1))
    sum = torch.unsqueeze(torch.sum(pred_x, dim=2), dim=1) + torch.sum(gt_x, 2, keepdim=True) - matching_score
    dim0, dim1, dim2 = sum.shape
    matching_score = 1 - torch.divide(matching_score,
                                      torch.maximum(sum, torch.full((dim0, dim1, dim2), 1e-8).to(device)))
    matching_idx = torch.zeros((batch_size, nmask, 2), dtype=int)
    curnmasks = curnmasks.type(torch.int32)
    for i, curnmask in enumerate(curnmasks):
        matching_score_np = matching_score.clone().detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(matching_score_np[i, :curnmask, :])
        matching_idx[i, :curnmask, 0] = torch.tensor(row_ind).to(device)
        matching_idx[i, :curnmask, 1] = torch.tensor(col_ind).to(device)
    return matching_idx


def iou(mask_pred, gt_x, gt_valid_pl, n_point, nmask, end_points):
    matching_idx = hungarian_matching(mask_pred, gt_x, gt_valid_pl)
    matching_idx.requires_grad = False
    end_points['matching_idx'] = matching_idx

    matching_idx_row = matching_idx[:, :, 0]
    idx = (matching_idx_row >= 0).nonzero(as_tuple=False)
    matching_idx_row = torch.cat((torch.unsqueeze(idx[:, 0].int(), dim=-1), torch.reshape(matching_idx_row, (-1, 1))),
                                 1)
    gt_x_matched = torch.reshape(gt_x[list(matching_idx_row.T.long())], (-1, nmask, n_point))

    matching_idx_column = matching_idx[:, :, 1]
    idx = (matching_idx_column >= 0).nonzero(as_tuple=False)
    matching_idx_column = torch.cat(
        (torch.unsqueeze(idx[:, 0].int(), dim=-1), torch.reshape(matching_idx_column, (-1, 1))), 1)
    matching_idx_column_torch2 = matching_idx_column.detach().cpu().numpy()
    pred_x_matched = torch.reshape(mask_pred[list(matching_idx_column.T)], (-1, nmask, n_point))

    matching_score = torch.sum(torch.multiply(gt_x_matched, pred_x_matched), 2)
    iou_all = torch.div(matching_score,
                        torch.sum(gt_x_matched, 2) + torch.sum(pred_x_matched, 2) - matching_score + 1e-8)
    end_points['per_shape_all_iou'] = iou_all
    meaniou = torch.div(torch.sum(torch.multiply(iou_all, gt_valid_pl), 1), torch.sum(gt_valid_pl, -1) + 1e-8)  # B
    return meaniou, end_points


def get_ins_loss(mask_pred, gt_mask_pl, gt_valid_pl, end_points):
    """ Input:      mask_pred   B x K x N
                    mask_gt     B x K x N
                    gt_valid    B x K
    """
    gt_x = gt_mask_pl.float()
    gt_valid_pl = gt_valid_pl.float()
    _, num_ins, num_point = mask_pred.shape
    meaniou, end_points = iou(mask_pred, gt_x, gt_valid_pl, num_point, num_ins, end_points)
    #print(meaniou)
    end_points['per_shape_mean_iou'] = meaniou
    loss = - torch.mean(meaniou)
    return loss, end_points