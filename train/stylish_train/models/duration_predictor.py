import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .common import LinearNorm
from .text_encoder import FFN, MultiHeadAttention, sequence_mask
from .common import InstanceNorm1d
from .conv_next import ConvNeXtBlock
from .text_encoder import TextEncoder
from .fine_style_encoder import FineStyleEncoder
from .prosody_encoder import ProsodyEncoder
from utils import length_to_mask
from .plbert import PLBERT

from xlstm import (
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    xLSTMBlockStack,
)


# class DurationPredictor(nn.Module):
#     def __init__(self, style_dim, in_channels, filter_channels, kernel_size, dropout):
#         super().__init__()
#         self.in_channels = in_channels
#         self.filter_channels = filter_channels
#         self.dropout = dropout
#
#         self.drop = torch.nn.Dropout(dropout)
#         self.conv_1 = torch.nn.Conv1d(
#             in_channels, filter_channels, kernel_size, padding=kernel_size // 2
#         )
#         self.norm_1 = AdaLayerNorm(style_dim, filter_channels)
#         self.conv_2 = torch.nn.Conv1d(
#             filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
#         )
#         self.norm_2 = AdaLayerNorm(style_dim, filter_channels)
#         self.proj = torch.nn.Conv1d(filter_channels, 1, 1)
#
#     def forward(self, x, style, x_mask):
#         x = self.conv_1(x * x_mask)
#         x = torch.relu(x)
#         x = rearrange(x, "b c t -> b t c")
#         x = self.norm_1(x, style)
#         x = rearrange(x, "b t c -> b c t")
#         x = self.drop(x)
#         x = self.conv_2(x * x_mask)
#         x = torch.relu(x)
#         x = rearrange(x, "b c t -> b t c")
#         x = self.norm_2(x, style)
#         x = rearrange(x, "b t c -> b c t")
#         x = self.drop(x)
#         x = self.proj(x * x_mask)
#         return x * x_mask


class DurationPredictor(nn.Module):
    def __init__(
        self, style_dim, inter_dim, text_config, style_config, duration_config
    ):
        super().__init__()
        self.text_encoder = TextEncoder(inter_dim=inter_dim, config=text_config)
        self.bert = PLBERT(
            vocab_size=178,
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=2048,
            max_position_embeddings=512,
            num_hidden_layers=12,
            dropout=0.1,
        )
        self.bert_encoder = nn.Linear(768, 512)
        self.style_encoder = FineStyleEncoder(
            inter_dim,
            style_dim,
            config=style_config,
            # model_config.style_encoder.layers,
        )
        # self.prosody_encoder = ProsodyEncoder(
        #     sty_dim=style_dim,
        #     d_model=768,
        #     nlayers=duration_config.n_layer,
        #     dropout=duration_config.dropout,
        # )
        self.prosody_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=512, nlayers=3, dropout=0.2
        )
        # self.cfg = xLSTMBlockStackConfig(
        #     mlstm_block=mLSTMBlockConfig(
        #         mlstm=mLSTMLayerConfig(
        #             conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        #         )
        #     ),
        #     context_length=512,
        #     num_blocks=8,
        #     embedding_dim=512 + style_dim,
        # )
        # self.lstm = xLSTMBlockStack(self.cfg)
        self.lstm = nn.LSTM(
            512 + style_dim, 512 // 2, 1, batch_first=True, bidirectional=True
        )
        self.duration_proj = LinearNorm(512, duration_config.max_dur)
        # self.blocks = nn.ModuleList()
        # for _ in range(nlayers):
        #     # self.blocks.append(AdaWinBlock1d(dim_in=d_hid, dim_out=d_hid, style_dim=style_dim, window_length=37, dropout_p=dropout))
        #     self.blocks.append(
        #         ConvNeXtBlock(
        #             dim_in=d_hid,
        #             dim_out=d_hid,
        #             intermediate_dim=d_hid * 4,
        #             style_dim=style_dim,
        #             dilation=[1, 3, 5],
        #             dropout=dropout,
        #             window_length=37,
        #         )
        #     )
        # self.duration_proj = LinearNorm(d_hid, 1)  # max_dur)

    def forward(self, texts, text_lengths):
        mask = length_to_mask(text_lengths).to(texts.device)
        bertmask = (~mask).int()
        with torch.no_grad():
            bert_embed = self.bert(texts, bertmask)
            encoding = self.bert_encoder(bert_embed)
            # encoding = bert_embed
            encoding = rearrange(encoding, "b t c -> b c t")
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        style = self.style_encoder(text_encoding, text_lengths)
        prosody = self.prosody_encoder(encoding, style, text_lengths, mask)

        nlengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            prosody, nlengths, batch_first=True, enforce_sorted=False
        )

        m = mask.unsqueeze(1)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, : x.shape[1], :] = x
        x = x_pad.to(x.device)

        # prosody = self.lstm(prosody)
        prosody = nn.functional.dropout(x, 0.5, training=self.training)
        duration = self.duration_proj(prosody)
        # x = texts
        # for block in self.blocks:
        #     x = block(x, style, text_lengths)
        # x = x.transpose(-2, -1)
        # duration = self.duration_proj(x)
        return duration  # .squeeze(-1)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        stylemean = (style * (~masks).int().unsqueeze(1)).sum(
            dim=2
        ) / text_lengths.unsqueeze(1)
        x = torch.cat([x, style], dim=1)
        x = x.permute(2, 0, 1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu()
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), stylemean).transpose(-1, -2)
                x = torch.cat([x, style], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad.to(x.device)
        return x.transpose(-1, -2)


#     def infer(self, x, style):
#         """
#         x: (batch, channels, tokens)
#         style: (batch, embedding)
#         """
#         # s = rearrange(style, "b e -> b e 1")
#         # s = s.expand(-1, -1, x.shape[2])  # batch embedding tokens
#         x = torch.cat([x, style], dim=1)

#         for block in self.lstms:
#             if isinstance(block, AdaLayerNorm):
#                 x = rearrange(x, "b c t -> b t c")
#                 x = block(x, style)
#                 x = rearrange(x, "b t c -> b c t")
#                 x = torch.cat([x, style], dim=1)
#             else:
#                 x = rearrange(x, "1 c t -> t c")
#                 block.flatten_parameters()
#                 x, _ = block(x)
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#                 x = rearrange(x, "t c -> 1 c t")

#         return rearrange(x, "b c t -> b t c")


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
