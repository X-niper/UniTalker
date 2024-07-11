# yapf: disable
import torch
import torch.nn as nn

from .base_model import BaseModel
from .model_utils import (
    TCN, PeriodicPositionalEncoding, enc_dec_mask, init_biased_mask,
    linear_interpolation,
)

# yapf: enable


class UniTalkerDecoderTCN(BaseModel):

    def __init__(self, args) -> None:
        super().__init__()
        self.learnable_style_emb = nn.Embedding(args.identity_num, 64)
        in_dim = args.decoder_dimension
        out_dim = args.decoder_dimension
        self.audio_feature_map = nn.Linear(args.audio_encoder_feature_dim, in_dim)
        self.tcn = TCN(in_dim + 64, out_dim)
        self.interpolate_pos = args.interpolate_pos

    def forward(self,
                hidden_states: torch.Tensor,
                style_idx: torch.Tensor,
                frame_num: int = None):
        batch_size = len(hidden_states)
        if style_idx is None:
            style_embedding = self.learnable_style_emb.weight[-1].unsqueeze(
                0).repeat(batch_size, 1)
        else:
            style_embedding = self.learnable_style_emb(style_idx)
        feature = self.audio_feature_map(hidden_states).transpose(1, 2)
        feature = self.tcn(feature, style_embedding)
        feature = feature.transpose(1, 2)
        if self.interpolate_pos == 2:
            feature = linear_interpolation(feature, output_len=frame_num)
        return feature


class UniTalkerDecoderTransformer(BaseModel):

    def __init__(self, args) -> None:
        super().__init__()
        out_dim = args.decoder_dimension
        self.learnable_style_emb = nn.Embedding(args.identity_num, out_dim)
        self.audio_feature_map = nn.Linear(args.audio_encoder_feature_dim, out_dim)
        self.PPE = PeriodicPositionalEncoding(
            out_dim, period=args.period, max_seq_len=3000)
        self.biased_mask = init_biased_mask(
            n_head=4, max_seq_len=3000, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=4,
            dim_feedforward=2 * out_dim,
            batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=1)
        self.interpolate_pos = args.interpolate_pos

    def forward(self, hidden_states: torch.Tensor, style_idx: torch.Tensor,
                frame_num: int):
        obj_embedding = self.learnable_style_emb(style_idx)
        obj_embedding = obj_embedding.unsqueeze(1).repeat(1, frame_num, 1)
        hidden_states = self.audio_feature_map(hidden_states)
        style_input = self.PPE(obj_embedding)
        tgt_mask = self.biased_mask[:, :style_input.shape[1], :style_input.
                                    shape[1]].clone().detach().to(
                                        device=style_input.device)
        memory_mask = enc_dec_mask(hidden_states.device, style_input.shape[1],
                                   frame_num)
        feat_out = self.transformer_decoder(
            style_input,
            hidden_states,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask)
        if self.interpolate_pos == 2:
            feat_out = linear_interpolation(feat_out, output_len=frame_num)
        return feat_out
