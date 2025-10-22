import math

import torch

from ros2_person_detection_2d_lidar.models.cnn import BlockConv1d


class MemorySpatialAttention(torch.nn.Module):
    def __init__(self, shape_input, num_channels_hidden, num_neighbors_attention, rate_update_autoregression, use_full_fov=False):
        """A memory module that updates both with similarity-based spatial attention and auto-regressively"""
        super().__init__()

        self.inds_neighbor = None
        self.mask_neighbor = None
        self.num_neighbors_attention = num_neighbors_attention
        self.num_channels_hidden = num_channels_hidden
        self.rate_update_autoregression = rate_update_autoregression
        self.shape_input = shape_input
        self.state_memory = None
        self.use_full_fov = use_full_fov

        self._init()

    def _init(self):
        self.conv = BlockConv1d(
            self.shape_input[2],
            self.num_channels_hidden,
            shape_kernel_conv=(self.shape_input[3]),
            kwargs_conv=dict(padding=0),
            name_layer_norm="BatchNorm1d",
            name_layer_act="LeakyReLU",
            kwargs_act=dict(negative_slope=0.1, inplace=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)

    def reset(self):
        self.state_memory = None

    def update_attention_mask(self, num_cutouts, device):
        inds_col = torch.arange(num_cutouts, device=device)[..., None].long()
        size_window_half = self.num_neighbors_attention // 2
        inds_window = torch.arange(-size_window_half, size_window_half + 1, device=device).long()
        inds_col = inds_col + inds_window[None, ...]

        # NOTE: On JRDB, DR-SPAAM takes part of the panoramic scan and at test time takes the whole panoramic scan
        if self.use_full_fov and not self.training:
            inds_col = inds_col % num_cutouts
        else:
            inds_col = inds_col.clamp(min=0, max=num_cutouts - 1)

        inds_row = torch.arange(num_cutouts, device=device)[..., None].expand_as(inds_col).long()
        inds_full = torch.stack((inds_row, inds_col), dim=2).view(-1, 2)

        masks = torch.zeros(num_cutouts, num_cutouts, device=device).float()
        masks[inds_full[:, 0], inds_full[:, 1]] = 1.0

        self.mask_neighbor = masks
        self.inds_neighbor = inds_full

    def forward(self, input):
        if self.state_memory is None:
            self.state_memory = input
            return self.state_memory

        if self.mask_neighbor is None or self.mask_neighbor.shape[0] != input.shape[1]:
            self.update_attention_mask(input.shape[1], input.device)

        output = input.view(-1, input.shape[2], input.shape[3])
        output = self.conv(output)
        output = output.view(input.shape[0], input.shape[1], -1)

        output_memory = self.state_memory.view(-1, input.shape[2], input.shape[3])
        output_memory = self.conv(output_memory)
        output_memory = output_memory.view(input.shape[0], input.shape[1], -1)

        similarity = torch.matmul(output, output_memory.permute(0, 2, 1))
        # Make sure the out-of-window elements have small values
        similarity = similarity - 1e10 * (1.0 - self.mask_neighbor)
        # Masked softmax
        max_similarity = similarity.max(dim=-1, keepdim=True)[0]
        exps = torch.exp(similarity - max_similarity) * self.mask_neighbor
        similarity = exps / exps.sum(dim=-1, keepdim=True)

        # NOTE: This gather scatter version is only marginally more efficient on memory
        # similarity_w = torch.gather(similarity, 2, self.inds_neighbor[None, ...])
        # similarity_w = similarity_w.softmax(dim=2)
        # similarity = torch.zeros_like(similarity)
        # similarity.scatter_(2, self.inds_neighbor[None, ...], similarity_w)

        attention_memory = self.state_memory.view(input.shape[0], input.shape[1], -1)
        attention_memory = torch.matmul(similarity, attention_memory)
        attention_memory = attention_memory.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])

        output = self.rate_update_autoregression * input + (1.0 - self.rate_update_autoregression) * attention_memory
        self.state_memory = output

        return output


class Drspaam(torch.nn.Module):
    def __init__(
        self,
        shape_input,
        num_channels_hidden_memory,
        num_neighbors_attention,
        rate_update_autoregression,
        prob_dropout,
        temperature_confidence=1.0,
        use_full_fov=False,
    ):
        super().__init__()

        self.num_channels_hidden_memory = num_channels_hidden_memory
        self.num_neighbors_attention = num_neighbors_attention
        self.prob_dropout = prob_dropout
        self.rate_update_autoregression = rate_update_autoregression
        self.shape_input = shape_input
        self.temperature_confidence = temperature_confidence
        self.use_full_fov = use_full_fov

        self._init()

    def _init(self):
        self.conv_block_1 = torch.nn.Sequential(
            BlockConv1d(
                1,
                64,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                64,
                64,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                64,
                128,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
                name_layer_pool="MaxPool1d",
                kwargs_pool=dict(kernel_size=2),
                prob_dropout=self.prob_dropout,
            ),
        )

        self.conv_block_2 = torch.nn.Sequential(
            BlockConv1d(
                128,
                128,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                128,
                128,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                128,
                256,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
                name_layer_pool="MaxPool1d",
                kwargs_pool=dict(kernel_size=2),
                prob_dropout=self.prob_dropout,
            ),
        )

        self.conv_block_3 = torch.nn.Sequential(
            BlockConv1d(
                256,
                256,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                256,
                256,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                256,
                512,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
                name_layer_pool="MaxPool1d",
                kwargs_pool=dict(kernel_size=2),
                prob_dropout=self.prob_dropout,
            ),
        )

        self.conv_block_4 = torch.nn.Sequential(
            BlockConv1d(
                512,
                256,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
            BlockConv1d(
                256,
                128,
                shape_kernel_conv=(3),
                kwargs_conv=dict(padding=1),
                name_layer_norm="BatchNorm1d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.1, inplace=True),
            ),
        )

        self.conv_cls = torch.nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = torch.nn.Conv1d(128, 2, kernel_size=1)

        self.gate = MemorySpatialAttention(
            shape_input=(self.shape_input[0], self.shape_input[1], 256, int(math.ceil(self.shape_input[3] / 4))),
            num_channels_hidden=self.num_channels_hidden_memory,
            num_neighbors_attention=self.num_neighbors_attention,
            rate_update_autoregression=self.rate_update_autoregression,
            use_full_fov=self.use_full_fov,
        )

        self.sigmoid = torch.nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)

    def forward_body(self, input):
        if self.training:
            self.gate.reset()

        for i in range(input.shape[2]):
            input_scan = input[:, :, i, :]

            output = input_scan.view(-1, 1, input.shape[3])
            output = self.conv_block_1(output)
            output = self.conv_block_2(output)
            output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2])

            output = self.gate(output)

        output = output.view(-1, output.shape[2], output.shape[3])
        output = self.conv_block_3(output)
        output = self.conv_block_4(output)
        output = torch.nn.functional.avg_pool1d(output, kernel_size=output.shape[-1])

        return output

    def forward_head_confidence(self, input, shape):
        output = self.conv_cls(input)
        output = output.view(shape[0], shape[1], -1)
        output /= self.temperature_confidence
        output = self.sigmoid(output)

        return output

    def forward_head_position(self, input, shape):
        output = self.conv_reg(input)
        output = output.view(shape[0], shape[1], -1)

        return output

    def forward(self, input):
        output = self.forward_body(input)

        output_confidence = self.forward_head_confidence(output, input.shape)
        output_position = self.forward_head_position(output, input.shape)

        output = dict(position=output_position, confidence=output_confidence)
        return output


class DrspaamBbox(Drspaam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init(self):
        super()._init()
        self.conv_box = torch.nn.Conv1d(128, 4, kernel_size=1)

    def forward_head_bbox(self, input, shape):
        output = self.conv_box(input)
        output = output.view(shape[0], shape[1], -1)

        return output

    def forward(self, input):
        output = self.forward_body(input)

        output_confidence = self.forward_head_confidence(output, input.shape)
        output_position = self.forward_head_position(output, input.shape)
        output_bbox = self.forward_head_bbox(output, input.shape)

        output = dict(position=output_position, confidence=output_confidence, bbox=output_bbox)
        return output
