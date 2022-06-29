from logging import error
import math
import sys
import time
import os
import torch
from pic_predict_utils.getPredictedPic import instance_segmentation_api

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric


def train_one_epoch(model, model_dis, optimizer, optimizer_dis, data_loader, tar_train_data_loader, device, epoch, epochs,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    if model_dis != None:
        model_dis.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    source_label = 1
    target_label = 0
    model_dis_alpah = 1
    one_epoch_dis_loss = 0
    one_epoch_adv_loss = 0

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # 获取目标数据集枚举类
    tar_trainloader_iter = enumerate(tar_train_data_loader)



    mloss = torch.zeros(1).to(device)  # mean losses
    tar_mloss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # 训练mask_rcnn
            # 在源域数据上监督训练（目标检测 + 语义分割）
            loss_dict, mask_logits, outputs = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 在目标域数据上进行目标检测监督训练 + 语义分割对抗训练
            try:
                _ ,[tar_images,tar_targets] = next(tar_trainloader_iter)
            except StopIteration as e:
                tar_trainloader_iter = enumerate(tar_train_data_loader)
                _ ,[tar_images,tar_targets] = next(tar_trainloader_iter)
                
            tar_images = list(image.to(device) for image in tar_images)
            tar_targets = [{k: v.to(device) for k, v in t.items()} for t in tar_targets]
            tar_loss_dict, tar_mask_logits, tar_outputs = model(tar_images, tar_targets)
            del tar_loss_dict['loss_mask']
            tar_losses = sum(loss for loss in tar_loss_dict.values())
            tar_losses *= model_dis_alpah

            # losses = losses + model_dis_alpah*tar_losses

            # 判断是否进行对抗
            if  model_dis != None and epoch >0:
                # 辨别器不计算梯度
                for param in model_dis.parameters():
                    param.requires_grad = False
                # 目标数据集对抗训练，使得特征提取器在目标域提取与源域相似的特征
                tar_dis_out = model_dis(tar_mask_logits)
                src_adv_loss = bce_loss(tar_dis_out, torch.FloatTensor(tar_dis_out.data.size()).fill_(source_label).cuda())
                src_adv_loss *= model_dis_alpah

                # 训练鉴别器，使得鉴别器具有鉴别能力
                for param in model_dis.parameters():
                    param.requires_grad = True
                # 鉴别器在源域训练
                mask_logits = mask_logits.detach()
                src_dis_out = model_dis(mask_logits)
                src_dis_loss = bce_loss(src_dis_out, torch.FloatTensor(src_dis_out.data.size()).fill_(source_label).to(device))
                src_dis_loss *= model_dis_alpah

                # 鉴别器在目标域训练
                tar_mask_logits = tar_mask_logits.detach()
                tar_dis_out = model_dis(tar_mask_logits)
                tar_dis_loss = bce_loss(tar_dis_out, torch.FloatTensor(tar_dis_out.data.size()).fill_(target_label).to(device))
                tar_dis_loss *= model_dis_alpah

                # 鉴别总损失
                dis_loss = src_dis_loss + tar_dis_loss

                one_epoch_adv_loss += src_adv_loss.item()
                one_epoch_dis_loss += dis_loss.item()


        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        # reduce tar_loss over all GPUs for logging purpose
        tar_loss_dict_reduced = utils.reduce_dict(tar_loss_dict)
        tar_losses_reduced = sum(loss for loss in tar_loss_dict_reduced.values())
        tar_loss_value = tar_losses_reduced.item()

        # 记录源域训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses
        # 记录目标域训练损失
        tar_mloss = (tar_mloss * i + tar_loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # 清除优化器梯度
        optimizer.zero_grad()
        if model_dis != None and epoch >0:
            optimizer_dis.zero_grad()
        # 是否裁剪梯度
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward(retain_graph=True)
            if model_dis != None and epoch >0:
                tar_losses.backward(retain_graph=True)
                src_adv_loss.backward()
                dis_loss.backward()
                optimizer_dis.step()
            optimizer.step()


        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
        
        # 日志记录
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
    
    f = open("results_file/model_dis_loss.txt" , 'a', encoding='gbk')
    print("epoch: " + str(epoch), file=f)
    print("adv_loss: ",one_epoch_adv_loss/(i+1), file=f)
    print("dis loss: ",one_epoch_dis_loss/(i+1), file=f)
    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, results_file):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name=os.path.join(results_file, "det_results.json"))
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name=os.path.join(results_file, "seg_results.json"))
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        _, _ , outputs= model(image)
        # 显示预测图
        save_path = "results_file/pre_pic"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        folder = str(len(os.listdir(save_path)))
        save_path = os.path.join(save_path,folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        instance_segmentation_api(image,outputs,save_path)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = det_metric.evaluate()
        seg_info = seg_metric.evaluate()
    else:
        coco_info = None
        seg_info = None

    return coco_info, seg_info
