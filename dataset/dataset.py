import json
import numpy as np
import os.path as osp
from torch.utils import data
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from typing import Any, List

from .data_item import DataItem
from .dataset_config import dataset_config


class AudioFaceDataset(data.Dataset):

    def __init__(self, data_label_path: str, args: dict = None) -> None:
        super().__init__()
        data_root = osp.dirname(data_label_path, )
        id_template_path = osp.join(data_root, 'id_template.npy')
        id_template_list = np.load(id_template_path)
        with open(data_label_path) as f:
            labels = json.load(f)
        info = labels['info']
        self.id_list = info['id_list']
        self.sr = 16000
        data_info_list = labels['data']
        data_list = []
        for data_info in tqdm(data_info_list):
            data = DataItem(
                annot_path=osp.join(data_root, data_info['annot_path']),
                audio_path=osp.join(data_root, data_info['audio_path']),
                identity_idx=data_info['id'],
                annot_type=data_info['annot_type'],
                dataset_name=data_info['dataset'],
                id_template=id_template_list[data_info['id']],
                fps=data_info['fps'],
                processor=args.processor,
            )
            if data.duration < 0.5:
                continue
            data_list.append(data)
        self.data_list = data_list
        return

    def __len__(self, ):
        return len(self.data_list)

    def __getitem__(self, index: Any) -> Any:
        data = self.data_list[index]
        return data.get_dict()

    def get_identity_num(self, ):
        return len(self.id_list)


class MixAudioFaceDataset(AudioFaceDataset):

    def __init__(self,
                 dataset_list: List[AudioFaceDataset],
                 duplicate_list: list = None) -> None:
        super(AudioFaceDataset).__init__()
        self.id_list = []
        self.sr = 16000
        self.data_list = []
        self.annot_type_list = []
        if duplicate_list is None:
            duplicate_list = [1] * len(dataset_list)
        else:
            assert len(duplicate_list) == len(dataset_list)
        for dup, dataset in zip(duplicate_list, dataset_list):
            id_index_offset = len(self.id_list)
            for d in dataset.data_list:
                d.offset_id(id_index_offset)
                if d.annot_type not in self.annot_type_list:
                    self.annot_type_list.append(d.annot_type)
            self.id_list = self.id_list + dataset.id_list
            self.data_list = self.data_list + dataset.data_list * dup
        return


def get_dataset_list(args):
    dataset_name_list = args.dataset
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    train_json_list = [
        osp.join(args.data_root, dataset_name, args.train_json_key)
        for dataset_name in dataset_dir_list
    ]
    args.read_audio = False
    args.processor = None
    train_dataset_list = []
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor
    for train_json in train_json_list:
        dataset = AudioFaceDataset(
            train_json,
            args,
        )
        train_dataset_list.append(dataset)
    return train_dataset_list


def get_single_dataset(args, json_path: str = ''):
    args.read_audio = True
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor

    dataset = AudioFaceDataset(
        json_path,
        args,
    )
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers)
    return test_loader


def get_dataloaders(args):
    dataset_name_list = args.dataset
    print(dataset_name_list)
    dataset_dir_list = [
        dataset_config[name]['dirname'] for name in dataset_name_list
    ]
    train_json_list = [
        osp.join(args.data_root, dataset_dir, args.train_json_key)
        for dataset_dir in dataset_dir_list
    ]
    val_json_list = [
        osp.join(args.data_root, dataset_dir, 'val.json')
        for dataset_dir in dataset_dir_list
    ]
    test_json_list = [
        osp.join(args.data_root, dataset_dir, 'test.json')
        for dataset_dir in dataset_dir_list
    ]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo)
    args.processor = processor
    if getattr(args, 'do_train', True) is True:
        train_dataset_list = []
        for train_json in train_json_list:
            dataset = AudioFaceDataset(train_json, args)
            train_dataset_list.append(dataset)
        mix_train_dataset = MixAudioFaceDataset(train_dataset_list,
                                                args.duplicate_list)
        train_loader = data.DataLoader(
            dataset=mix_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.workers)
    else:
        train_loader = None
    if getattr(args, 'do_validate', True) is True:
        val_dataset_list = []
        for val_json in val_json_list:
            dataset = AudioFaceDataset(val_json, args)
            val_dataset_list.append(dataset)
        mix_val_dataset = MixAudioFaceDataset(val_dataset_list)
        val_loader = data.DataLoader(
            dataset=mix_val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers)
    else:
        val_loader = None
    if getattr(args, 'do_test', True) is True:
        test_dataset_list = []
        for test_json in test_json_list:
            dataset = AudioFaceDataset(test_json, args)
            test_dataset_list.append(dataset)
        mix_test_dataset = MixAudioFaceDataset(test_dataset_list)
        test_loader = data.DataLoader(
            dataset=mix_test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader
