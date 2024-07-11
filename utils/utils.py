import argparse
import logging
import numpy as np
import os
import random
import torch
from copy import deepcopy

from . import config


def get_audio_encoder_dim(audio_encoder: str):
    if audio_encoder == 'microsoft/wavlm-base-plus':
        return 768
    elif audio_encoder == 'facebook/wav2vec2-large-xlsr-53':
        return  1024
    else:
        raise ValueError("wrong audio_encoder")

def filer_list(in_list: list, key: str):
    ret_list = []
    for line in in_list:
        if line.startswith(key):
            ret_list.append(line)
    return ret_list


def count_checkpoint_params(ckpt: dict):
    params = 0
    pca_params = 0
    for k, v in ckpt.items():
        if 'pca' in k:
            print(k)
            pca_params += np.prod(v.size())
        params += np.prod(v.size())
    return params, pca_params


def read_obj(in_path):
    with open(in_path, 'r') as obj_file:
        # Read the lines of the OBJ file
        lines = obj_file.readlines()

    # Initialize empty lists for vertices and faces
    verts = []
    faces = []
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        elements = line.split()  # Split the line into elements

        if len(elements) == 0:
            continue  # Skip empty lines

        # Check the type of line (vertex or face)
        if elements[0] == 'v':
            # Vertex line
            x, y, z = map(float,
                          elements[1:4])  # Extract the vertex coordinates
            verts.append((x, y, z))  # Add the vertex to the list
        elif elements[0] == 'f':
            # Face line
            face_indices = [
                int(index.split('/')[0]) for index in elements[1:]
            ]  # Extract the vertex indices
            faces.append(face_indices)  # Add the face to the list
    verts = np.array(verts)
    faces = np.array(faces)
    if faces.min() == 1:
        faces = faces - 1
    return verts, faces


def get_logger(log_file: str):
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]=>%(message)s'  # noqa: E501
    handler.setFormatter(logging.Formatter(fmt))
    for hdl in logger.handlers:
        logger.removeHandler(hdl)
    logger.addHandler(handler)
    return logger


def get_template_dict(data_root, annot_type: str):
    import pickle
    if annot_type == 'BIWI_23370_vertices':
        template_pkl_path = os.path.join(data_root, 'BIWI', 'templates.pkl')
    elif annot_type == 'FLAME_5023_vertices':
        template_pkl_path = os.path.join(data_root, 'vocaset', 'templates.pkl')
    with open(template_pkl_path, 'rb') as f:
        template_dict = pickle.load(f, encoding='latin')
    return template_dict


# def get_template_verts(data_root, annot_type: str, subject: str):
#     if isinstance(subject, str):
#         if subject.endswith('.obj'):
#             verts = read_obj(subject)[0]
#             return verts
#         if subject.endswith('.npy'):
#             return np.load(subject)[0]
#         template_dict = get_template_dict(data_root, annot_type)
#         return template_dict[subject]
#     else:
#         return subject

def get_template_verts(data_root, dataset_name: str, subject: int):
    from dataset.dataset_config import dataset_config
    template_path = os.path.join(data_root, dataset_config[dataset_name]['dirname'], 'id_template.npy')
    return np.load(template_path)[subject]


def load_ckpt(model, ckpt_path, re_init_decoder_and_head: bool = False):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    id_embedding_key = 'decoder.learnable_style_emb.weight'
    if len(checkpoint[id_embedding_key]) != len(
            model.state_dict()[id_embedding_key]):
        target_id_num = len(model.state_dict()[id_embedding_key])
        checkpoint[id_embedding_key] = checkpoint[id_embedding_key][-1][
            None].repeat(target_id_num, 1)
    if re_init_decoder_and_head is True:
        ckpt_keys = list(checkpoint.keys())
        encoder_keys = filer_list(ckpt_keys, 'audio_encoder')
        reinit_keys = list(set(ckpt_keys) - set(encoder_keys))
        for k in reinit_keys:
            if k in model.state_dict():
                checkpoint[k] = model.state_dict()[k]
    model.load_state_dict(checkpoint, strict=False)
    return


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self, ):
        return self.avg


def get_average_meter_dict(args, phase):
    if phase == 'train':
        metric_name_list = ['rec_L2_loss', 'pca_loss']
    elif phase == 'val' or phase == 'test':
        metric_name_list = ['rec_L2_loss', 'LVE', 'LVD']
    else:
        raise ValueError('wrong phase for average meter')
    average_meter_dict_template = {}
    for k in metric_name_list:
        average_meter_dict_template[k] = AverageMeter()
    mixed_dataset_name = 'mixed_dataset'
    dataset_name_list = args.dataset + [mixed_dataset_name]
    average_meter_dict = {}
    for dataset_name in dataset_name_list:
        average_meter_dict[dataset_name] = deepcopy(
            average_meter_dict_template)
    return average_meter_dict, metric_name_list


def log_datasetloss(
    logger,
    epoch: int,
    phase: str,
    dataset_loss_dict: dict,
    only_mix: bool = True,
):
    base_str = f'{phase.upper():<5} Epoch: {epoch:03d} '
    for dataset_name, loss_info in dataset_loss_dict.items():
        if only_mix and dataset_name != 'mixed_dataset':
            continue
        for loss_type, avg_meter in loss_info.items():
            base_str += f'{loss_type}_{dataset_name}:{avg_meter.avg: .8f}, '
    logger.info(base_str)
    return base_str


def write_to_tensorboard(writer,
                         epoch: int,
                         phase: str,
                         dataset_loss_dict: dict,
                         only_mix: bool = False):
    for dataset_name, loss_info in dataset_loss_dict.items():
        if only_mix and dataset_name != 'mixed_dataset':
            continue
        for loss_type, avg_meter in loss_info.items():
            key = f'{phase}/{loss_type}_{dataset_name}'
            writer.add_scalar(key, avg_meter.avg, epoch)
    return 0


def get_parser():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument(
        '--config',
        type=str,
        default='config/unitalker.yaml',
        help='config file')
    parser.add_argument(
        '--condition_id', type=str, default='common', help='condition id')
    parser.add_argument(
        'opts', help=' ', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.condition_id = args.condition_id
    cfg.dataset = cfg.dataset.split(',')
    if isinstance(cfg.duplicate_list, str):
        cfg.duplicate_list = cfg.duplicate_list.split(',')
        cfg.duplicate_list = [int(i) for i in cfg.duplicate_list]
    return cfg


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return
