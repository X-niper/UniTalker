import librosa
import numpy as np
import os
import torch
import torch.optim
import torch.utils.data
from transformers import Wav2Vec2FeatureExtractor

from dataset.dataset_config import dataset_config
from loss.loss import UniTalkerLoss
from models.unitalker import UniTalker
from utils.utils import get_parser, get_template_verts, get_audio_encoder_dim

def get_all_audios(audio_root:str):
    wav_f_names = []
    for r, _, f in os.walk(audio_root):
        for wav in f:
            if wav.endswith('.wav') or wav.endswith('.mp3'):
                relative_path = os.path.join(r, wav)
                relative_path = os.path.relpath(relative_path,
                                                audio_root)
                wav_f_names.append(relative_path)
    wav_f_names = sorted(wav_f_names)
    return wav_f_names

def split_long_audio(
    audio: np.ndarray,
    processor:Wav2Vec2FeatureExtractor
):
    # audio = audio.squeeze(0)
    a, b = 25, 5
    sr = 16000 
    total_length = len(audio) /sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    in_audio_split_list = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for i in range(reps):
        audio_split = audio[start:end]
        audio_split = np.squeeze(
            processor(audio_split, sampling_rate=sr).input_values)
        in_audio_split_list.append(audio_split)
        start += step
        end += step
    return in_audio_split_list

def merge_out_list(out_list: list, fps:int):
    if len(out_list) == 1:
        return out_list[0]
    a, b = 25, 5
    left_weight = np.linspace(1, 0, b * fps)[:, np.newaxis]
    right_weight = 1 - left_weight
    a = a * fps 
    b = b * fps 
    offset = a - b

    out_length = len(out_list[-1]) + offset * (len(out_list) - 1)
    merged_out = np.empty((out_length, out_list[-1].shape[-1]),
                            dtype=out_list[-1].dtype)
    merged_out[:a] = out_list[0]
    for out_piece in out_list[1:]:
        merged_out[a - b:a] = left_weight * merged_out[
            a - b:a] + right_weight * out_piece[:b]
        merged_out[a:a + offset] = out_piece[b:]
        a += offset
    return merged_out


def main():
    args = get_parser()
    training_dataset_name_list = args.dataset
    
    condition_id_config = {
        'D0': 3,
        'D1': 3,
        'D2': 0,
        'D3': 0,
        'D4': 0,
        'D5': 0,
        'D6': 4,
        'D7': 0,
    }
    template_id_config = {
        'D0': 3,
        'D1': 3,
        'D2': 0,
        'D3': 0,
        'D4': 0,
        'D5': 0,
        'D6': 4,
        'D7': 0,
    }

    checkpoint = torch.load(args.weight_path, map_location='cpu')
    args.identity_num = len(checkpoint['decoder.learnable_style_emb.weight'])


    start_idx = 0
    for dataset_name in training_dataset_name_list:
        annot_type = dataset_config[dataset_name]['annot_type']
        id_num = dataset_config[dataset_name]['subjects']
        end_idx = start_idx + id_num
        local_condition_idx = condition_id_config[dataset_name]
        template_idx = template_id_config[dataset_name]
        template = get_template_verts(args.data_root, dataset_name, template_idx)
        template_id_config[dataset_name] = torch.Tensor(template.reshape(1, -1))
        if args.condition_id == 'each':
            condition_id_config[dataset_name] = torch.tensor(
                start_idx + local_condition_idx).reshape(1)
        elif args.condition_id == 'common':
            condition_id_config[dataset_name] = torch.tensor(args.identity_num -
                                                      1).reshape(1)
        else:
            try:
                condition_id = int(args.condition_id)
                condition_id_config[dataset_name] = torch.tensor(
                    condition_id).reshape(1)
            except ValueError:
                assert args.condition_id in dataset_config.keys()
                condition_id_config[dataset_name] = torch.tensor(
                    start_idx + local_condition_idx).reshape(1)
        start_idx = end_idx
    if args.random_id_prob > 0:
        assert end_idx == (args.identity_num - 1)
    else:
        assert end_idx == args.identity_num

    args.audio_encoder_feature_dim = get_audio_encoder_dim(
        args.audio_encoder_repo)

    model = UniTalker(args)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.cuda()

    wav_f_names = get_all_audios(args.test_wav_dir)
    out_npz_path = args.test_out_path


    out_dict = {}
    loss_module = UniTalkerLoss(args).cuda()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.audio_encoder_repo,)
    
    with torch.no_grad():
        for wav_f in wav_f_names:
            out_dict[wav_f] = {}
            wav_path = os.path.join(args.test_wav_dir, wav_f)
            audio_data, sr = librosa.load(wav_path, sr=16000)
            audio_data = np.squeeze(
                processor(audio_data, sampling_rate=sr).input_values)

            audio_data_splits = split_long_audio(audio_data, processor)
            for dataset_name in training_dataset_name_list:
                template = template_id_config[dataset_name].cuda()
                scale = dataset_config[dataset_name]['scale']
                template = scale * template
                condition_id = condition_id_config[dataset_name].cuda()
                annot_type = dataset_config[dataset_name]['annot_type']
                if annot_type == 'BIWI_23370_vertices':
                    fps = 25
                else:
                    fps = 30

                out_list = []
                for audio_data in audio_data_splits:
                    audio_data = torch.Tensor(audio_data[None]).cuda()
                    out, _, _ = model(
                        audio_data,
                        template,
                        face_motion=None,
                        style_idx=condition_id,
                        annot_type=annot_type,
                        fps=fps,
                    )
                    out_list.append(out.cpu().numpy().squeeze(0))
                out = merge_out_list(out_list, fps)
                out = loss_module.get_vertices(torch.from_numpy(out).cuda(), annot_type)
                out_dict[wav_f][dataset_name] = out 
        print(f"save results to {out_npz_path}")
        np.savez(out_npz_path, **out_dict)


if __name__ == '__main__':
    main()
