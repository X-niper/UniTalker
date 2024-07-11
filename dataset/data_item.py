import librosa
import numpy as np
from typing import Any

from .dataset_config import dataset_config


class DataItem:

    def __init__(self,
                 annot_path: str,
                 audio_path: str,
                 identity_idx: int,
                 annot_type: str = '',
                 dataset_name: str = '',
                 id_template: np.ndarray = None,
                 fps: float = 30,
                 processor=None) -> None:
        self.annot_path = annot_path
        self.audio_path = audio_path
        self.identity_idx = identity_idx
        self.fps = fps
        self.annot_data = np.load(annot_path)
        if self.fps == 60:
            self.annot_data = self.annot_data[::2]
            self.fps = 30
        elif self.fps == 100:
            self.annot_data = self.annot_data[::4]
            self.fps = 25
        elif self.fps != 30 and self.fps != 25:
            raise ValueError('wrong fps')
        scale = dataset_config[dataset_name]['scale']
        self.annot_data = self.scale_and_offset(self.annot_data, scale, 0.0)
        self.audio_data, sr = self.load_audio(audio_path)
        self.original_audio_data, self.annot_data = self.truncate(
            self.audio_data, sr, self.annot_data, self.fps)
        self.audio_data = np.squeeze(
            processor(self.original_audio_data, sampling_rate=sr).input_values)
        self.audio_data = self.audio_data.astype(np.float32)
        self.annot_type = annot_type
        self.id_template = self.scale_and_offset(id_template, scale, 0.0)
        self.annot_data = self.annot_data.reshape(len(self.annot_data),
                                                  -1).astype(np.float32)
        self.id_template = self.id_template.reshape(1, -1).astype(np.float32)
        self.dataset_name = dataset_name
        self.data_dict = self.to_dict()
        return

    def load_audio(self, audio_path: str):
        if audio_path.endswith('.npy'):
            return np.load(audio_path), 16000
        else:
            return librosa.load(audio_path, sr=16000)

    def truncate(self, audio_array: np.ndarray, sr: int,
                 annot_data: np.ndarray, fps: int):
        audio_duration = len(audio_array) / sr
        annot_duration = len(annot_data) / fps
        duration = min(audio_duration, annot_duration, 20)
        audio_length = round(duration * sr)
        annot_length = round(duration * fps)
        self.duration = duration
        return audio_array[:audio_length], annot_data[:annot_length]

    def offset_id(self, offset: int):
        self.identity_idx = self.identity_idx + offset
        self.data_dict['id'] = self.identity_idx
        return

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        return data * scale + offset

    def to_dict(self, ) -> Any:
        item = {
            'data': self.annot_data,
            'audio': self.audio_data,
            'original_audio': self.original_audio_data,
            'fps': self.fps,
            'id': self.identity_idx,
            'annot_path': self.annot_path,
            'audio_path': self.audio_path,
            'annot_type': self.annot_type,
            'template': self.id_template,
            'dataset_name': self.dataset_name
        }
        return item

    def get_dict(self, ):
        return self.data_dict
