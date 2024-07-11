#!/usr/bin/env python
# yapf: disable
import os
import torch
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from dataset.dataset import get_dataloaders
from loss.loss import UniTalkerLoss
from models.unitalker import UniTalker
from utils.utils import (
    get_average_meter_dict, get_logger, get_parser, get_audio_encoder_dim,
    load_ckpt, seed_everything,
)
from .train_eval_loop import train_epoch, validate_epoch

# yapf: enable


def main():
    seed_everything(42)
    args = get_parser()
    args.audio_encoder_feature_dim = get_audio_encoder_dim(args.audio_encoder_repo)
    train_loader, val_loader, test_loader = get_dataloaders(args)
    args.identity_num = train_loader.dataset.get_identity_num()
    args.annot_type_list = train_loader.dataset.annot_type_list
    if args.random_id_prob > 0.0:
        args.identity_num = args.identity_num + 1

    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, 'train.log')
    logger = get_logger(log_file)
    writer = SummaryWriter(args.save_path)
    logger.info('=> creating model ...')
    model = UniTalker(args)
    if args.weight_path is not None:
        logger.info(f'=> loading model {args.weight_path} ...')
        load_ckpt(model, args.weight_path, args.re_init_decoder_and_head)
    logger.info(args)
    model = model.cuda()
    model.summary(logger)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    loss_module = UniTalkerLoss(args).cuda()

    args.train_meter, args.train_metric_name_list = \
        get_average_meter_dict(args, 'train')
    args.val_meter, args.val_metric_name_list = \
        get_average_meter_dict(args, 'val')
    args.logger = logger
    args.writer = writer

    for epoch in range(args.epochs):
        if args.fix_encoder_first:
            if epoch == 0:
                model.audio_encoder._freeze_wav2vec2_parameters(True)
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr)
            elif epoch == 1:
                model.audio_encoder._freeze_wav2vec2_parameters(
                    args.freeze_wav2vec)
                model.audio_encoder.feature_extractor._freeze_parameters()
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=args.lr)
        epoch_plus = epoch + 1
        train_epoch(train_loader, model, loss_module, optimizer, epoch_plus,
                    args)

        if epoch_plus % args.eval_every == 0:
            validate_epoch(val_loader, model, loss_module, epoch_plus, 'val',
                           args)

        if epoch_plus % args.test_every == 0:
            validate_epoch(test_loader, model, loss_module, epoch_plus, 'test',
                           args)

        if epoch_plus % args.save_every == 0:
            model_path = os.path.join(args.save_path, f'{epoch_plus:03d}.pt')
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
