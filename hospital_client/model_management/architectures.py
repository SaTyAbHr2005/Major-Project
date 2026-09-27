"""
Real architecture factory implementations for Module 6's model catalog.

Seven of the eight catalog entries wrap torchvision's reference
implementations (EfficientNet-B0/B4, ResNet18/50, MobileNetV2,
MobileNetV3-Small, ViT-B/16). torchvision has no MobileViT implementation in
any released version, and adding a new third-party dependency (e.g. timm)
for a single architecture was not part of the approved dependency set
(torch, torchvision, safetensors). Rather than fake a substitute or skip the
entry, MobileViT-XXS is implemented here from scratch in pure PyTorch,
following the published MobileViT block design (local conv representation ->
unfold into patches -> transformer over patches -> fold back -> fuse with a
residual). It is NOT loaded from any reference/pretrained checkpoint - see
`supports_pretrained=False` in its catalog entry.

Every factory takes a ModelConfig and returns a real, freshly constructed
torch.nn.Module with a classifier head sized to config.num_classes.
"""
import os
from dataclasses import dataclass
from typing import Callable, Dict
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision.models as tv_models

from hospital_client.model_management.config import ModelConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _set_module_by_path(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def _adapt_first_conv_for_color_mode(model: nn.Module, color_mode: str) -> None:
    """
    Replaces the first Conv2d encountered (the network's stem/patch-embedding
    conv) with an equivalent one accepting 1 input channel, for grayscale
    ("L") input. RGB is a no-op. Never guesses/zero-fills weights for the
    replaced layer beyond standard PyTorch initialization - a stem swap is a
    structural change, not a weight-preserving one.
    """
    if color_mode != "L":
        return
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=1,
                bias=module.bias is not None,
            )
            _set_module_by_path(model, name, new_conv)
            return
    raise ValueError("Could not locate a Conv2d stem layer to adapt for grayscale input.")


def _is_weights_cached_locally(url: str) -> bool:
    """
    Checks torch hub's local checkpoint cache WITHOUT ever attempting a
    network request. Used to guarantee pretrained=True never silently
    downloads anything: if the file isn't already present locally, the
    caller raises instead of letting torchvision's loader attempt a fetch.
    """
    filename = os.path.basename(urlparse(url).path)
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    return os.path.exists(os.path.join(cache_dir, filename))


def _require_local_weights(weights_enum, architecture_name: str):
    if not _is_weights_cached_locally(weights_enum.url):
        raise ValueError(
            f"Pretrained weights for '{architecture_name}' are not available locally "
            f"(expected under {os.path.join(torch.hub.get_dir(), 'checkpoints')}). "
            "Module 6 does not download model weights automatically. Provision the "
            "weights locally (e.g. pre-populate the torch hub cache) or set pretrained=False."
        )
    return weights_enum


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# torchvision-backed factories
# ---------------------------------------------------------------------------

def build_efficientnet_b0(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.EfficientNet_B0_Weights.DEFAULT, "efficientnet_b0")
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    dropout_p = config.dropout if config.dropout is not None else model.classifier[0].p
    model.classifier = nn.Sequential(nn.Dropout(p=dropout_p, inplace=True), nn.Linear(in_features, config.num_classes))
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_efficientnet_b4(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.EfficientNet_B4_Weights.DEFAULT, "efficientnet_b4")
    model = tv_models.efficientnet_b4(weights=weights)
    in_features = model.classifier[-1].in_features
    dropout_p = config.dropout if config.dropout is not None else model.classifier[0].p
    model.classifier = nn.Sequential(nn.Dropout(p=dropout_p, inplace=True), nn.Linear(in_features, config.num_classes))
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_resnet18(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.ResNet18_Weights.DEFAULT, "resnet18")
    model = tv_models.resnet18(weights=weights)
    in_features = model.fc.in_features
    if config.dropout:
        model.fc = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(in_features, config.num_classes))
    else:
        model.fc = nn.Linear(in_features, config.num_classes)
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_resnet50(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.ResNet50_Weights.DEFAULT, "resnet50")
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    if config.dropout:
        model.fc = nn.Sequential(nn.Dropout(p=config.dropout), nn.Linear(in_features, config.num_classes))
    else:
        model.fc = nn.Linear(in_features, config.num_classes)
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_mobilenet_v2(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.MobileNet_V2_Weights.DEFAULT, "mobilenet_v2")
    model = tv_models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    dropout_p = config.dropout if config.dropout is not None else model.classifier[0].p
    model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, config.num_classes))
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_mobilenet_v3_small(config: ModelConfig) -> nn.Module:
    weights = None
    if config.pretrained:
        weights = _require_local_weights(tv_models.MobileNet_V3_Small_Weights.DEFAULT, "mobilenet_v3_small")
    model = tv_models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, config.num_classes)
    if config.dropout is not None:
        # classifier: [Linear, Hardswish, Dropout, Linear] - index 2 is Dropout
        model.classifier[2] = nn.Dropout(p=config.dropout, inplace=True)
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


def build_vit_b16(config: ModelConfig) -> nn.Module:
    h, w = config.input_size
    if h != w:
        raise ValueError(f"vit_b16 requires a square input_size, got {config.input_size!r}")
    dropout_p = config.dropout if config.dropout is not None else 0.0

    if config.pretrained:
        if h != 224:
            raise ValueError(
                f"Pretrained vit_b16 weights are only available for image_size=224, got {h}. "
                "Set pretrained=False to use a different input_size."
            )
        weights = _require_local_weights(tv_models.ViT_B_16_Weights.DEFAULT, "vit_b16")
        model = tv_models.vit_b_16(weights=weights, dropout=dropout_p)
    else:
        model = tv_models.vit_b_16(weights=None, image_size=h, dropout=dropout_p)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, config.num_classes)
    _adapt_first_conv_for_color_mode(model, config.color_mode)
    return model


# ---------------------------------------------------------------------------
# MobileViT-XXS - implemented from scratch (see module docstring)
# ---------------------------------------------------------------------------

class _MV2Block(nn.Module):
    """MobileNetV2-style inverted residual block."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU()]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class _TinyTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout)),
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            y = norm1(x)
            attn_out, _ = attn(y, y, y, need_weights=False)
            x = x + attn_out
            x = x + mlp(norm2(x))
        return x


class _MobileViTBlock(nn.Module):
    def __init__(self, dim: int, depth: int, channel: int, heads: int = 4, patch_size: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ph = self.pw = patch_size
        self.local_rep = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
            nn.Conv2d(channel, dim, 1, bias=False),
        )
        self.transformer = _TinyTransformer(dim, depth, heads, dim * 2, dropout)
        self.proj = nn.Conv2d(dim, channel, 1, bias=False)
        self.fusion = nn.Conv2d(2 * channel, channel, 3, 1, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.local_rep(x)
        b, d, h, w = x.shape
        ph, pw = self.ph, self.pw
        pad_h, pad_w = (ph - h % ph) % ph, (pw - w % pw) % pw
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        nh, nw = hp // ph, wp // pw

        x = x.reshape(b, d, nh, ph, nw, pw).permute(0, 3, 5, 2, 4, 1).reshape(b * ph * pw, nh * nw, d)
        x = self.transformer(x)
        x = x.reshape(b, ph, pw, nh, nw, d).permute(0, 5, 3, 1, 4, 2).reshape(b, d, hp, wp)

        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        x = self.proj(x)
        x = torch.cat([x, residual], dim=1)
        return self.fusion(x)


class MobileViTXXS(nn.Module):
    """
    A from-scratch, faithful-in-structure MobileViT-XXS: MobileNetV2 stages
    for local features, with three MobileViT transformer blocks fusing
    global context at progressively lower resolutions. Not derived from or
    weight-compatible with any published checkpoint.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU())
        self.stage1 = _MV2Block(16, 16, 1, expand_ratio=2)
        self.stage2 = nn.Sequential(_MV2Block(16, 24, 2, expand_ratio=2), _MV2Block(24, 24, 1, expand_ratio=2))
        self.stage3_mv2 = _MV2Block(24, 48, 2, expand_ratio=2)
        self.stage3_vit = _MobileViTBlock(dim=64, depth=2, channel=48)
        self.stage4_mv2 = _MV2Block(48, 64, 2, expand_ratio=2)
        self.stage4_vit = _MobileViTBlock(dim=80, depth=4, channel=64)
        self.stage5_mv2 = _MV2Block(64, 80, 2, expand_ratio=2)
        self.stage5_vit = _MobileViTBlock(dim=96, depth=3, channel=80)
        self.conv_last = nn.Sequential(nn.Conv2d(80, 320, 1, bias=False), nn.BatchNorm2d(320), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3_vit(self.stage3_mv2(x))
        x = self.stage4_vit(self.stage4_mv2(x))
        x = self.stage5_vit(self.stage5_mv2(x))
        x = self.conv_last(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def build_mobilevit_xxs(config: ModelConfig) -> nn.Module:
    if config.pretrained:
        raise ValueError(
            "mobilevit_xxs has no pretrained weights implementation in Module 6 "
            "(it is a from-scratch architecture, not loaded from a reference checkpoint). "
            "Set pretrained=False."
        )
    in_channels = 1 if config.color_mode == "L" else 3
    dropout_p = config.dropout if config.dropout is not None else 0.0
    return MobileViTXXS(num_classes=config.num_classes, in_channels=in_channels, dropout=dropout_p)
