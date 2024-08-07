# Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================
import numpy as np
import torch
from torch import nn

from . import measure


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


@measure("zen", bn=True)
def compute_zen_score(
    net,
    inputs,
    targets,  # pylint: disable=unused-argument
    loss_fn=None,  # pylint: disable=unused-argument
    split_data=1,  # pylint: disable=unused-argument
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    with torch.no_grad():
        for _repeat_count in range(repeat):  # pylint: disable=unused-variable
            network_weight_gaussian_init(net)
            input = torch.randn(  # pylint: disable=redefined-builtin
                size=list(inputs.shape),
                device=device,
                dtype=dtype,
            )
            input2 = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2

            # output = net.forward_before_global_avg_pool(input)
            # mixup_output = net.forward_before_global_avg_pool(mixup_input)
            output = net(input)
            mixup_output = net(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    return float(np.mean(nas_score_list))
