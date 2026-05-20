from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_heads(channels: int, preferred_heads: int) -> int:
    for heads in range(min(preferred_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class LocalDetailExpert(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContMixDynamicConvExpert(nn.Module):
    """Dependency-free ContMix-style dynamic local/global kernel mixer.

    This keeps the OverLoCK idea that pooled global context predicts token-wise
    small/large kernels, but uses unfold instead of natten/custom large-kernel ops.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        small_kernel_size: int = 5,
        context_size: int = 7,
        num_heads: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        kernel_size = max(3, kernel_size | 1)
        small_kernel_size = min(max(3, small_kernel_size | 1), kernel_size)
        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size
        self.context_size = context_size
        self.num_heads = _safe_heads(dim, num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.pre_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.weight_proj = nn.Linear(
            context_size * context_size,
            small_kernel_size * small_kernel_size + kernel_size * kernel_size,
            bias=True,
        )
        self.large_lepe = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=bias)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias), nn.SiLU())
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _neighborhood_mix(self, value: torch.Tensor, attn: torch.Tensor, kernel_size: int) -> torch.Tensor:
        b, c, h, w = value.shape
        patches = F.unfold(value, kernel_size=kernel_size, padding=kernel_size // 2)
        patches = patches.view(b, self.num_heads, self.head_dim, kernel_size * kernel_size, h * w)
        mixed = torch.einsum("bgnk,bgckn->bgcn", attn, patches)
        return mixed.reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x + self.pre_dw(x)
        b, c, h, w = x.shape
        context = F.adaptive_avg_pool2d(x, self.context_size)

        query = self.q(x).view(b, self.num_heads, self.head_dim, h * w) * self.scale
        key = self.k(context).view(b, self.num_heads, self.head_dim, self.context_size * self.context_size)
        affinity = torch.einsum("bgcn,bgcm->bgnm", query, key)
        weights = self.weight_proj(affinity)
        small_weights, large_weights = torch.split(
            weights,
            [self.small_kernel_size * self.small_kernel_size, self.kernel_size * self.kernel_size],
            dim=-1,
        )
        small_weights = small_weights.softmax(dim=-1)
        large_weights = large_weights.softmax(dim=-1)

        value = self.v(x)
        small = self._neighborhood_mix(value, small_weights, self.small_kernel_size)
        large = self._neighborhood_mix(value, large_weights, self.kernel_size)
        out = 0.5 * (small + large) + self.large_lepe(x)
        out = self.proj(out * self.gate(x))
        return out + residual


class FrequencyDynamicConvExpert(nn.Module):
    """FDConv-inspired depthwise dynamic convolution.

    The learned kernels live in disjoint Fourier bands and are transformed back
    with iFFT. A lightweight attention head mixes the frequency-diverse kernels
    per sample, with a low/high band feature gate for spatial adaptation.
    """

    def __init__(self, dim: int, kernel_size: int = 3, kernel_num: int = 4, bias: bool = False):
        super().__init__()
        kernel_size = max(3, kernel_size | 1)
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        self.freq_real = nn.Parameter(torch.randn(kernel_num, dim, kernel_size, kernel_size) * 0.02)
        self.freq_imag = nn.Parameter(torch.randn(kernel_num, dim, kernel_size, kernel_size) * 0.02)
        self.register_buffer("band_masks", self._build_radial_masks(kernel_num, kernel_size), persistent=False)

        hidden = max(dim // 4, 8)
        self.kernel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, kernel_num, kernel_size=1, bias=True),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.band_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _build_radial_masks(kernel_num: int, kernel_size: int) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy)
        radius = radius / radius.max().clamp_min(1.0)
        edges = torch.linspace(0, 1, steps=kernel_num + 1)
        masks = []
        for idx in range(kernel_num):
            if idx == kernel_num - 1:
                mask = (radius >= edges[idx]) & (radius <= edges[idx + 1])
            else:
                mask = (radius >= edges[idx]) & (radius < edges[idx + 1])
            masks.append(mask.float())
        return torch.stack(masks, dim=0).unsqueeze(1)

    def _spatial_kernels(self) -> torch.Tensor:
        coeff = torch.complex(self.freq_real, self.freq_imag) * self.band_masks.to(self.freq_real.dtype)
        kernels = torch.fft.ifft2(torch.fft.ifftshift(coeff, dim=(-2, -1)), dim=(-2, -1)).real
        return kernels - kernels.mean(dim=(-2, -1), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        kernels = self._spatial_kernels()
        branch_outputs = []
        for idx in range(self.kernel_num):
            weight = kernels[idx].unsqueeze(1)
            branch_outputs.append(F.conv2d(x, weight, padding=self.kernel_size // 2, groups=c))
        branches = torch.stack(branch_outputs, dim=1)

        kernel_att = self.kernel_att(x).flatten(1).softmax(dim=1).view(b, self.kernel_num, 1, 1, 1)
        out = (branches * kernel_att).sum(dim=1)

        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        out = out * self.channel_att(x) * self.band_gate(torch.cat([low, high], dim=1))
        return self.proj(out) + residual


class PeripheralLargeKernelExpert(nn.Module):
    """PeLK-inspired large receptive field expert with logarithmic ring sharing."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: Sequence[int] = (7, 15, 31),
        partial_ratio: float = 0.375,
        bias: bool = False,
    ):
        super().__init__()
        conv_channels = max(1, int(dim * partial_ratio))
        self.conv_channels = conv_channels
        self.id_channels = dim - conv_channels
        self.kernel_sizes: Tuple[int, ...] = tuple(max(3, int(k) | 1) for k in kernel_sizes)
        self.center = nn.Conv2d(conv_channels, conv_channels, kernel_size=5, padding=2, groups=conv_channels, bias=bias)
        self.ring_scale = nn.Parameter(torch.ones(len(self.kernel_sizes), conv_channels, 1, 1))
        self.position_comp = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, groups=conv_channels, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_conv, x_id = torch.split(x, [self.conv_channels, self.id_channels], dim=1)
        out = self.center(x_conv)
        previous = x_conv
        for idx, kernel_size in enumerate(self.kernel_sizes):
            blurred = F.avg_pool2d(x_conv, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            ring = blurred - previous
            out = out + ring * self.ring_scale[idx]
            previous = blurred
        out = out + self.position_comp(x_conv)
        if self.id_channels > 0:
            out = torch.cat([out, x_id], dim=1)
        return self.proj(out) + residual


class HeterogeneousConvExpert(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        expert_type: str,
        depth: int = 1,
        kernel_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()
        rank = rank or dim
        self.depth = depth
        self.proj_in = nn.Conv2d(dim, rank, kernel_size=1, bias=bias)
        self.shared_gate = nn.Conv2d(dim, rank, kernel_size=1, bias=bias)
        self.blocks = nn.ModuleList([self._make_block(expert_type, rank, kernel_size, bias) for _ in range(depth)])
        self.proj_out = nn.Conv2d(rank, dim, kernel_size=1, bias=bias)

    @staticmethod
    def _make_block(expert_type: str, dim: int, kernel_size: int, bias: bool) -> nn.Module:
        if expert_type == "contmix":
            return ContMixDynamicConvExpert(dim, kernel_size=max(kernel_size, 7), num_heads=4, bias=bias)
        if expert_type == "fdconv":
            return FrequencyDynamicConvExpert(dim, kernel_size=3, kernel_num=4, bias=bias)
        if expert_type == "pelk":
            return PeripheralLargeKernelExpert(dim, kernel_sizes=(7, 15, max(kernel_size, 31)), bias=bias)
        if expert_type == "local":
            return LocalDetailExpert(dim, kernel_size=3, bias=bias)
        raise ValueError(f"Unknown convolution expert type: {expert_type}")

    def forward(self, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        shortcut = x
        x = self.proj_in(x)
        gate = F.silu(self.shared_gate(shared))
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x * gate)
        return x + shortcut
