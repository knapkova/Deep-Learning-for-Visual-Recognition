"""ResNet50 的四种微调深度；冻结参数时同时冻结 BatchNorm 统计。"""
from __future__ import annotations

import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

DEPTHS = ('head', 'layer4', 'layer3_layer4', 'full')


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # CPU algorithms are deterministic where supported; MPS reproducibility is best effort.
    torch.use_deterministic_algorithms(True, warn_only=True)


def trainable_prefixes(depth: str) -> tuple[str, ...]:
    if depth not in DEPTHS:
        raise ValueError(f'Unknown depth {depth!r}; expected one of {DEPTHS}')
    return {'head': ('fc.',), 'layer4': ('layer4.', 'fc.'),
            'layer3_layer4': ('layer3.', 'layer4.', 'fc.'), 'full': ('',)}[depth]


def build_model(depth: str, seed: int = 42, pretrained: bool = True) -> nn.Module:
    """所有深度先用同一随机种子创建同一模型，然后才选择可训练层。"""
    prefixes = trainable_prefixes(depth)
    seed_everything(seed)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    model.finetune_depth = depth
    set_train_mode(model, depth)
    return model


def set_train_mode(model: nn.Module, depth: str) -> None:
    """可训练块正常训练；冻结块 BN 不更新 running_mean/var 或 batch counter。"""
    trainable_prefixes(depth)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()
