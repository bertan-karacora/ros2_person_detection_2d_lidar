import torch

import ros2_person_detection_2d_lidar.libs.utils_import as utils_import


class BlockConv1d(torch.nn.Sequential):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv,
        kwargs_conv=None,
        name_layer_norm=None,
        kwargs_norm=None,
        name_layer_act=None,
        kwargs_act=None,
        name_layer_pool=None,
        kwargs_pool=None,
        prob_dropout=None,
        kwargs_dropout=None,
    ):
        self.kwargs_act = kwargs_act or {}
        self.kwargs_dropout = kwargs_dropout or {}
        self.kwargs_conv = kwargs_conv or {}
        self.kwargs_norm = kwargs_norm or {}
        self.kwargs_pool = kwargs_pool or {}
        self.name_layer_act = name_layer_act
        self.name_layer_norm = name_layer_norm
        self.name_layer_pool = name_layer_pool
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.shape_kernel_conv = shape_kernel_conv
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        modules = [torch.nn.Conv1d(self.num_channels_in, self.num_channels_out, self.shape_kernel_conv, **self.kwargs_conv)]

        if self.name_layer_norm is not None:
            class_module = utils_import.import_module(self.name_layer_norm)
            if self.name_layer_norm in ["BatchNorm1d", "InstanceNorm1d"]:
                self.kwargs_norm["num_features"] = self.num_channels_out
            modules.append(class_module(**self.kwargs_norm))

        if self.name_layer_act is not None:
            class_module = utils_import.import_module(self.name_layer_act)
            modules.append(class_module(**self.kwargs_act))

        if self.name_layer_pool is not None:
            class_module = utils_import.import_module(self.name_layer_pool)
            modules.append(class_module(**self.kwargs_pool))

        if self.prob_dropout is not None:
            modules.append(torch.nn.Dropout(self.prob_dropout, **self.kwargs_dropout))

        super().__init__(*modules)
