import numpy as np
import torch
from tqdm import tqdm

from utils.utils import log_datasetloss, write_to_tensorboard


def train_epoch(data_loader, model, loss_module, optimizer, epoch, args):
    model.train()
    train_meter = args.train_meter
    metric_name_list = args.train_metric_name_list
    zero_loss = torch.tensor(0.0)
    mixed_dataset_name = 'mixed_dataset'
    pbar = tqdm(data_loader)
    pbar.set_description(f'Epoch: {epoch} train')
    for batch in pbar:
        data = batch['data'].cuda(non_blocking=True)
        template = batch['template'].cuda(non_blocking=True)
        audio = batch['audio'].cuda(non_blocking=True)
        identity = batch['id'].cuda(non_blocking=True)
        if args.random_id_prob > 0.0:
            if np.random.random() < args.random_id_prob:
                identity = identity.fill_(args.identity_num - 1)
        annot_type = batch['annot_type'][0]
        dataset_name = batch['dataset_name'][0]

        out_motion, out_pca, gt_pca = model(
            audio, template, data, style_idx=identity, annot_type=annot_type)

        rec_loss = loss_module(out_motion, data, annot_type)
        if out_pca is not None:
            pca_loss = loss_module.pca_loss(out_pca, gt_pca)
        else:
            pca_loss = zero_loss
        loss = rec_loss + args.pca_weight * pca_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in zip(metric_name_list, (rec_loss, pca_loss)):
            train_meter[dataset_name][k].update(v.item())
            train_meter[mixed_dataset_name][k].update(v.item())

    log_datasetloss(args.logger, epoch, 'train', train_meter, only_mix=True)
    write_to_tensorboard(
        args.writer, epoch, 'train', train_meter, only_mix=False)

    for dataset_name, sub_metric_dict in train_meter.items():
        for metric_name in sub_metric_dict.keys():
            sub_metric_dict[metric_name].reset()
    return


def validate_epoch(data_loader, model, loss_module, epoch, phase, args):
    assert phase in ('val', 'test')
    model.eval()
    val_meter = args.val_meter
    metric_name_list = args.val_metric_name_list
    mixed_dataset_name = 'mixed_dataset'
    with torch.no_grad():
        pbar = tqdm(data_loader)
        pbar.set_description(f'Epoch: {epoch} {phase:<5}')
        for batch in pbar:
            data = batch['data'].cuda(non_blocking=True)
            template = batch['template'].cuda(non_blocking=True)
            audio = batch['audio'].cuda(non_blocking=True)
            identity = batch['id'].cuda(non_blocking=True)
            annot_type = batch['annot_type'][0]
            dataset_name = batch['dataset_name'][0]
            if args.random_id_prob >= 1.0:
                identity = identity.fill_(args.identity_num - 1)
            out_motion, _, _ = model(
                audio,
                template,
                data,
                style_idx=identity,
                annot_type=annot_type)
            rec_loss = loss_module(out_motion, data, annot_type)
            mouth_metric_L2, mouth_metric_L2_norm = loss_module.mouth_metric(
                out_motion, data, annot_type)

            for k, v in zip(metric_name_list,
                            (rec_loss, mouth_metric_L2, mouth_metric_L2_norm)):
                val_meter[dataset_name][k].update(v.item())
                val_meter[mixed_dataset_name][k].update(v.item())
        if getattr(args, 'logger', None) is not None:
            log_datasetloss(
                args.logger, epoch, phase, val_meter, only_mix=True)
        if getattr(args, 'writer', None) is not None:
            write_to_tensorboard(
                args.writer, epoch, phase, val_meter, only_mix=False)
        for dataset_name, sub_metric_dict in val_meter.items():
            for metric_name in sub_metric_dict.keys():
                sub_metric_dict[metric_name].reset()
    return
