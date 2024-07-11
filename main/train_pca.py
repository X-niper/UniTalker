#!/usr/bin/env python
import numpy as np

from models.pca import PCA
from utils.utils import get_parser, get_audio_encoder_dim


def main():
    args = get_parser()
    args.audio_encoder_feature_dim = get_audio_encoder_dim(
        args.wav2vec2_type)
    main_worker(args)


def main_worker(cfg):
    pca_builder = PCA()
    from dataset.dataset import get_dataset_list
    train_dataset_list = get_dataset_list(cfg)
    for dataset in train_dataset_list:
        data = pca_builder.load_dataset(dataset)
        pca_info = pca_builder.build_incremental_PCA(
            data, batch_size=1095, trunc_dim=647)
        np.savez('mesh_talk_pca.npz', **pca_info)


if __name__ == '__main__':
    main()
