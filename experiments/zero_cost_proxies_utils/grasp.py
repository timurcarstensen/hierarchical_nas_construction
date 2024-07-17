# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn.functional as F
from torch import autograd, nn

from . import measure
from .p_utils import get_layer_metric_array


# code from https://github.com/ultralytics/yolov5/blob/6ab589583c86f265a8a226620cc1e87c1c2266f1/utils/activations.py#L15
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for torchscript, CoreML and ONNX


def replace_hardswish(base_module: nn.Module):
    for name, module in base_module.named_children():
        if isinstance(module, nn.Hardswish):
            setattr(base_module, name, Hardswish())
        elif isinstance(module, nn.Module):
            new_module = replace_hardswish(module)
            setattr(base_module, name, new_module)
    return base_module


@measure("grasp", bn=True, mode="param")
def compute_grasp_per_weight(
    net,
    inputs,
    targets,
    mode,
    loss_fn,
    T=1,
    num_iters=1,
    split_data=1,
):
    net = replace_hardswish(net)

    # get all applicable weights
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d | nn.Linear):
            weights.append(layer.weight)
            layer.weight.requires_grad_(True)  # TODO isn't this already true?

    # NOTE original code had some input/target splitting into 2
    # I am guessing this was because of GPU mem limit
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass #1
        grad_w = None
        for _ in range(num_iters):
            # TODO get new data, otherwise num_iters is useless!
            outputs = net.forward(inputs[st:en]) / T
            loss = loss_fn(outputs, targets[st:en])
            grad_w_p = autograd.grad(loss, weights, allow_unused=True)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx, _ in enumerate(grad_w):
                    grad_w[idx] += grad_w_p[idx]

    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        # forward/grad pass #2
        outputs = net.forward(inputs[st:en]) / T
        loss = loss_fn(outputs, targets[st:en])
        grad_f = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d | nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    # compute final sensitivity metric and put in grads
    def grasp(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad  # -theta_q Hg
            # NOTE in the grasp code they take the *bottom* (1-p)% of values
            # but we take the *top* (1-p)%, therefore we remove the -ve sign
            # EDIT accuracy seems to be negatively correlated with this metric, so we add -ve sign here!
        else:
            return torch.zeros_like(layer.weight)

    return get_layer_metric_array(net, grasp, mode)
