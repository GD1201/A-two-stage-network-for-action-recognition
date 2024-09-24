# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import numpy as np
import math
import sys
from typing import Iterable, Optional
import matplotlib.pyplot as plt 
import os
import csv
import torch
from datetime import datetime
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

name_list=[ "Wave",
    "Nod",
    "Bow",
    "Random move",
    "Push",
    "Shrink",
    "Jump",
    "Tremble",
    "Nod head",
    "Shake head",
    "Stretch",
    "Lower head",
    "Fall down",
    "Grab",
    "Push and pull",
    "Lift up",
    "Bend",
    "Bend and squat",
    "Raise hand",
    "Push and shove",
    "Stretch",
    "Throw",
    "Swing",
    "Raise over head",
    "Crawl",
    "Twist",
    "Walk",
    "Run",
    "Bend over",
    "Kick",
    "Punch",
    "Rotate",
    "Kneel",
    "Climb",
    "Lie flat",
    "Slide",
    "Flop",
    "Stand",
    "Sleep",
    "Lie on side",
    "Drift",
    "Go",
    "Stand",
    "Sit",
    "Stand",
    "Lie",
    "Sit down",
    "Stand",
    "Walk",
    "Fall down",
    "Lie down",
    "Lean",
    "Speak",
    "Push",
    "Lean against",
    "Stand up",
    "Sit down",
    "Sleep",
    "Nap",
    "Tremor"]
def plot_confusion_matrix(confusion_matrix, name_list, save_dir='images', filename='confusion_matrix.png'):
    # 绘制混淆矩阵
    plt.figure(figsize=(60,60))
    
    # 计算每行的总数，将计数转换为概率
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    
    # 绘制归一化的混淆矩阵
    plt.imshow(normalized_confusion_matrix, interpolation='nearest', cmap='Blues')
    
    # 添加颜色栏
    cbar = plt.colorbar(shrink=0.8)
    cbar.ax.tick_params(labelsize=16)
    
    # 设置坐标轴标签和刻度
    plt.xticks(ticks=np.arange(len(name_list)), labels=name_list, rotation=90, fontsize=12)
    plt.yticks(ticks=np.arange(len(name_list)), labels=name_list, fontsize=12)
    
    # 在每个单元格上添加概率值标签
    for i in range(len(name_list)):
        for j in range(len(name_list)):
            plt.text(j, i, "{:.2f}".format(normalized_confusion_matrix[i, j]),
                     ha='center', va='center', color='black', fontsize=16)
    
    # 设置标题和坐标轴标签
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    
    # 保存混淆矩阵图像
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_time = f"{filename}_{current_time}.png"
    save_path = os.path.join(save_dir, filename_with_time)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved at {save_path}")
    
    # 保存混淆矩阵数据到CSV文件
    csv_filename = os.path.join(save_dir, f"{filename}_{current_time}.csv")
    save_confusion_matrix_to_csv(confusion_matrix, name_list, csv_filename)
    print(f"Confusion matrix data saved to CSV file at {csv_filename}")
    
    return save_path

def save_confusion_matrix_to_csv(confusion_matrix, name_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入列名（类别名称）
        writer.writerow([''] + name_list)
        
        # 写入混淆矩阵数据
        for i, row in enumerate(confusion_matrix):
            writer.writerow([name_list[i]] + list(row))

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(11)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        prev_confusion_matrix = None
        # compute output
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # p,r,confusion_matrix=calculate_recall_precision(target,output,prev_confusion_matrix=prev_confusion_matrix)
        # precision, recall,confusion_matrix = calculate_recall_precision(target, output, prev_confusion_matrix)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    #     prev_confusion_matrix = confusion_matrix
    # plot_confusion_matrix(prev_confusion_matrix ,name_list,save_dir='images', filename='confusion_matrix.png')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list

def calculate_recall_precision(label, score,prev_confusion_matrix=None):
    instance_num, class_num = score.shape
    
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
    
        confusion_matrix[true_l][pred_l] += 1
    if prev_confusion_matrix is not None:
        # Add previous confusion matrix
        confusion_matrix += prev_confusion_matrix
    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall,confusion_matrix