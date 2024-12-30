# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


# === TimeSeriesEmbedding ===
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, config):
        super(TimeSeriesEmbedding, self).__init__()
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_features = config['num_features']

        layers = []
        input_size = 1 * self.patch_size

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(input_size, self.hidden_size))
            layers.append(nn.GELU())
            input_size = self.hidden_size
        layers.append(nn.Linear(input_size, self.hidden_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)

        mask = x[:, :, -1]
        valid_lengths = mask.sum(dim=1).long()  # Shape: (batch_size)

        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size  # 向上取整

        patches_list = []
        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0:
                continue
            xi = x[i, :vl, :1]
            total_padded_length = pc * self.patch_size
            padding_length = total_padded_length - vl
            if padding_length > 0:
                padding = torch.zeros(padding_length, 1, device=x.device, dtype=x.dtype)
                xi = torch.cat([xi, padding], dim=0)
            xi = xi.reshape(pc, self.patch_size * 1)
            patches_list.append(xi)

        if patches_list:
            x_patches = torch.cat(patches_list, dim=0)  # Shape: (total_patch_cnt, patch_size * num_features)
            x = self.mlp(x_patches)
        else:
            x = torch.empty(0, self.hidden_size, device=x.device)

        return x, patch_cnt


# === TS Encoder === #
# get_patch_cnt: From Time Series Embedding
def get_patch_cnt(x: torch.Tensor, ts_config):
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1, ts_config['num_features'])

    mask = x[:, :, -1]
    valid_lengths = mask.sum(1).long()  # Shape: (batch_size)

    patch_cnt = (valid_lengths + ts_config['patch_size'] - 1) // ts_config['patch_size']
    return patch_cnt
