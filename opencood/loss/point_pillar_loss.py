# -*- coding: utf-8 -*-
# Author: Lantao Li
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.utils.common_utils import limit_period
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from icecream import ic
import datetime
import matplotlib.pyplot as plt

class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.pos_cls_weight = args['pos_cls_weight']

        self.cls = args['cls']
        self.reg = args['reg']

        if 'dir' in args:
            self.dir = args['dir']
        else:
            self.dir = None

        if 'iou' in args:
            from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import aligned_boxes_iou3d_gpu
            self.iou_loss_func = aligned_boxes_iou3d_gpu
            self.iou = args['iou']
        else:
            self.iou = None

        if 'diffusion' in args:
            self.dif = args['diffusion']
        else:
            self.dif = None

        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if 'record_len' in output_dict:
            batch_size = int(output_dict['record_len'].sum())
        elif 'batch_size' in output_dict:
            batch_size = output_dict['batch_size']
        else:
            batch_size = target_dict['pos_equal_one'].shape[0]

        cls_labls = target_dict['pos_equal_one'].view(batch_size, -1,  1)
        positives = cls_labls > 0
        negatives = target_dict['neg_equal_one'].view(batch_size, -1,  1) > 0
        # cared = torch.logical_or(positives, negatives)
        # cls_labls = cls_labls * cared.type_as(cls_labls)
        # num_normalizer = cared.sum(1, keepdim=True)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        #print(output_dict)
        # rename variable
        if f'psm{suffix}' in output_dict:
            output_dict[f'cls_preds{suffix}'] = output_dict[f'psm{suffix}']
        if f'rm{suffix}' in output_dict:
            output_dict[f'reg_preds{suffix}'] = output_dict[f'rm{suffix}']
        if f'dm{suffix}' in output_dict:
            output_dict[f'dir_preds{suffix}'] = output_dict[f'dm{suffix}']

        total_loss = 0

        # cls loss
        cls_preds = output_dict[f'cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = sigmoid_focal_loss(cls_preds, cls_labls, weights=cls_weights, **self.cls)
        cls_loss = cls_loss.sum() * self.cls['weight'] / batch_size

        # reg loss
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = target_dict['targets'].view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg['sigma'])
        reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size


        ######## direction ##########
        if self.dir:
            dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
            dir_logits = output_dict[f"dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]

            dir_loss = softmax_cross_entropy_with_logits(dir_logits.view(-1, self.anchor_num), dir_targets.view(-1, self.anchor_num))
            dir_loss = dir_loss.flatten() * reg_weights.flatten()
            dir_loss = dir_loss.sum() * self.dir['weight'] / batch_size
            total_loss += dir_loss
            self.loss_dict.update({'dir_loss': dir_loss.item()})


        ######## IoU ###########
        if self.iou:
            iou_preds = output_dict[f'iou_preds{suffix}'].permute(0, 2, 3, 1).contiguous()
            pos_pred_mask = reg_weights.squeeze(dim=-1) > 0 # (4, 70400)
            iou_pos_preds = iou_preds.view(batch_size, -1)[pos_pred_mask]
            boxes3d_pred = VoxelPostprocessor.delta_to_boxes3d(output_dict[f'reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().detach(),
                                                            output_dict['anchor_box'])[pos_pred_mask]
            boxes3d_tgt = VoxelPostprocessor.delta_to_boxes3d(target_dict['targets'],
                                                            output_dict['anchor_box'])[pos_pred_mask]
            iou_weights = reg_weights[pos_pred_mask].view(-1)
            iou_pos_targets = self.iou_loss_func(boxes3d_pred.float()[:, [0, 1, 2, 5, 4, 3, 6]], # hwl -> dx dy dz
                                                    boxes3d_tgt.float()[:, [0, 1, 2, 5, 4, 3, 6]]).detach().squeeze()
            iou_pos_targets = 2 * iou_pos_targets.view(-1) - 1
            iou_loss = weighted_smooth_l1_loss(iou_pos_preds, iou_pos_targets, weights=iou_weights, sigma=self.iou['sigma'])

            iou_loss = iou_loss.sum() * self.iou['weight'] / batch_size
            total_loss += iou_loss
            self.loss_dict.update({'iou_loss': iou_loss.item()})

        ######## Diffusion middle result ###########
        if self.dif != None and self.dif['dif']:
            # cls loss
            dif_cls_preds = output_dict[f'dif_cls_preds{suffix}'].permute(0, 2, 3, 1).contiguous() \
                        .view(batch_size, -1,  1)
            dif_cls_weights = positives * self.pos_cls_weight + negatives * 1.0
            dif_cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            dif_cls_loss = sigmoid_focal_loss(dif_cls_preds, cls_labls, weights=dif_cls_weights, **self.cls)
            dif_cls_loss = dif_cls_loss.sum() * self.cls['weight'] * self.dif['weight']/ batch_size
            # reg loss
            dif_reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
            dif_reg_preds = output_dict[f'dif_reg_preds{suffix}'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
            dif_reg_targets = target_dict['targets'].view(batch_size, -1, 7)
            dif_reg_preds, dif_reg_targets = self.add_sin_difference(dif_reg_preds, dif_reg_targets)
            dif_reg_loss = weighted_smooth_l1_loss(dif_reg_preds, dif_reg_targets, weights=dif_reg_weights, sigma=self.reg['sigma'])
            dif_reg_loss = dif_reg_loss.sum() * self.reg['weight'] * self.dif['weight']/ batch_size

            if self.dir:
                dif_dir_targets = self.get_direction_target(target_dict['targets'].view(batch_size, -1, 7))
                dif_dir_logits = output_dict[f"dif_dir_preds{suffix}"].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # [N, H*W*#anchor, 2]
                dif_dir_loss = softmax_cross_entropy_with_logits(dif_dir_logits.view(-1, self.anchor_num), dif_dir_targets.view(-1, self.anchor_num))
                dif_dir_loss = dif_dir_loss.flatten() * dif_reg_weights.flatten()
                dif_dir_loss = dif_dir_loss.sum() * self.dir['weight'] * self.dif['weight']/ batch_size
                dir_loss += dif_dir_loss
                total_loss += dif_dir_loss
                self.loss_dict.update({'dir_loss': dir_loss.item()})

            reg_loss += dif_reg_loss
            cls_loss += dif_cls_loss
            print("intermediate loss calculated for diffusion!")
        ######## Final Loss Calculation###########
        total_loss += reg_loss + cls_loss

        self.loss_dict.update({'total_loss': total_loss.item(),
                               'reg_loss': reg_loss.item(),
                               'cls_loss': cls_loss.item()})

        ######## Flow Matching Progressive ###########
        if 'fm_cls_list' in output_dict:
            #total_loss = 0.0 #reset total loss
            cls_0_preds = output_dict['cls_0_preds'].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,  1)
            flow_cls_loss, flow_cls_loss_dict = self.progressive_focal_flow_loss(batch_size, cls_0_preds, output_dict['fm_cls_list'], cls_labls, weights=cls_weights, **self.cls)
            flow_cls_loss = flow_cls_loss.sum() * self.cls['weight'] / batch_size
            print(flow_cls_loss)
            total_loss += flow_cls_loss
            for k, v in flow_cls_loss_dict.items():
                print(f"{k:>15s}: {v:.6f}")

        if 'fm_reg_list' in output_dict:
            #reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
            reg_0_preds = output_dict['reg_0_preds']
            reg_targets = target_dict['targets']
            flow_reg_loss, flow_reg_loss_dict = self.progressive_reg_flow_loss(batch_size, reg_0_preds, output_dict['fm_reg_list'], reg_targets,
                weights=reg_weights, sigma=self.reg['sigma'])
            flow_reg_loss = reg_loss.sum() * self.reg['weight'] / batch_size
            print(flow_reg_loss)
            total_loss += flow_reg_loss
            for k, v in flow_reg_loss_dict.items():
                print(f"{k:>15s}: {v:.6f}")

        return total_loss


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
            reg_targets:  [N, H * W * #anchor_num, 7]
                The last term is (theta_gt - theta_a)

        Returns:
            dir_targets:
                theta_gt: [N, H * W * #anchor_num, NUM_BIN]
                NUM_BIN = 2
        """
        num_bins = self.dir['args']['num_bins']
        dir_offset = self.dir['args']['dir_offset']
        anchor_yaw = np.deg2rad(np.array(self.dir['args']['anchor_yaw']))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1,-1,1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num//self.anchor_num, 1).to(reg_targets.device) # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1] # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets

    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss'+suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss'+suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss'+suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss'+suffix, iou_loss,
                            epoch*batch_len + batch_id)

    def progressive_focal_flow_loss(self, batch_size, preds, flowed_preds, targets, weights=None, **kwargs):
        N = len(flowed_preds)

        with torch.no_grad():
            p0 = torch.sigmoid(preds)
            progressive_targets = [
                (1 - (i+1)/(N+1))*p0 + ((i+1)/(N+1))*targets
                for i in range(N)
            ]

        total_loss = 0.0
        loss_dict = {}
        for i, (p_i, y_i) in enumerate(zip(flowed_preds, progressive_targets)):
            p_i = p_i.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,  1)
            loss_i = sigmoid_focal_loss(p_i, y_i, weights=weights, **kwargs)
            # for visualization
            """
            p_i_v = p_i.view(batch_size, 128, 256, 6).permute(0, 3, 1, 2).contiguous()
            y_i_v = y_i.view(batch_size, 128, 256, 6).permute(0, 3, 1, 2).contiguous()
            heatmap_score_function(y_i_v[0][1], str(i))
            """
            total_loss += loss_i
            loss_dict[f"cls_flow_step_{i+1}"] = loss_i.sum().item()
        total_loss /= N
        return total_loss, loss_dict

    def progressive_reg_flow_loss(self, batch_size, preds, flowed_preds, targets, sigma=3.0, weights=None):
        #preds: 原始 frozen 模型输出 (B, N_obj, 7)
        #flowed_preds: list of N 个中间预测，每个形状同 preds
        #targets: 最终目标 (B, N_obj, 7)
        #weights: (B, N_obj) 或 (B, N_obj, 1)
        N = len(flowed_preds)
        with torch.no_grad():
        # progressive 目标从原始 pred 平滑过渡到 target
            preds_0 = preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
            targets_0 = targets.view(batch_size, -1, 7)
            #preds_0, targets_0 = self.add_sin_difference(preds_0, targets)
            progressive_targets = [
                (1 - (i + 1) / (N+1)) * preds_0 + ((i + 1) / (N+1)) * targets_0
                for i in range(N)
            ]
        total_loss = 0.0
        loss_dict = {}
        for i, (p_i, y_i) in enumerate(zip(flowed_preds, progressive_targets)):
            p_i = p_i.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
            #p_i, _ = self.add_sin_difference(p_i, targets)
            #p_i, y_i = self.add_sin_difference(p_i, y_i)
            loss_i = weighted_smooth_l1_loss(p_i, y_i, sigma=sigma, weights=weights)
            total_loss += loss_i
            loss_dict[f"reg_flow_step_{i+1}"] = loss_i.sum().item()
        total_loss /= N
        return total_loss, loss_dict

def one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot

def softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert 'gamma' in kwargs and 'alpha' in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs['gamma'])
    alpha_weight_factor = targets * kwargs['alpha'] + (1 - targets) * (1 - kwargs['alpha'])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss

def heatmap_score_function(input_tensor, image_info, quantile=0.0):
    input_tensor = input_tensor.squeeze(0).squeeze(0)  # 形状变为 (128, 256)
    # 归一化到 0-1 范围
    tensor_min = input_tensor.min()
    tensor_max = input_tensor.max()
    tensor_normalized = (input_tensor - tensor_min) / (tensor_max - tensor_min + 1e-6)
    if quantile != 0.0:
        threshold = torch.quantile(tensor_normalized, quantile)
        modified = torch.where(tensor_normalized <= threshold, threshold, tensor_normalized)
        tensor_min = modified.min()
        tensor_max = modified.max()
        tensor_normalized = (modified - tensor_min) / (tensor_max - tensor_min + 1e-6)

    now = datetime.datetime.now()
    #tensor_filename = f"{image_info}_tensor"
    #torch.save(input_tensor.cpu(), f"{tensor_filename}.pt")
    heatmap_data = tensor_normalized.cpu().numpy() #.detach().numpy()
    symmetric_data = np.flipud(heatmap_data)  # 关键修改：上下翻转数组
    # 创建热力图
    plt.figure(figsize=(6, 3))
    im = plt.imshow(symmetric_data, cmap='inferno', aspect='auto')
    plt.axis('off')  # 关闭坐标轴
    # 添加颜色条（Colorbar）
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity", fontsize=10)  # 颜色条标题
    cbar.ax.tick_params(labelsize=8)  # 颜色条刻度字号
    # 生成文件名并保存
    #now = datetime.datetime.now()
    #filename = f"{image_info}_heatmap_{now.strftime('%m%d%H%M%S')}.png"
    #filename = f"{image_info}_heatmap_{now.strftime('%m%d%H%M%S')}.pdf"
    filename = f"{image_info}_heatmap.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
