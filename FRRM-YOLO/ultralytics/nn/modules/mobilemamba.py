# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MobileMamba modules for lightweight YOLO backbone replacement.

Implements the Wavelet-Semantic Splitter (WSS) architecture:
  - HaarWaveletTransform / InverseHaarWaveletTransform
  - MambaSSM (four-directional SS2D scanning)
  - PureVSSBlock (Mamba + Local DWConv for LL subband)
  - MultiKernelDeConv (multi-kernel depthwise conv for LH/HL/HH subbands)
  - SEModule (cross-subband channel attention)
  - WaveletSemanticSplitterBlock (frequency-domain semantic decoupling)
  - C3k2_Mamba_Block (YAML-callable wrapper)

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None


# ==================== Wavelet Transform ====================

class HaarWaveletTransform(nn.Module):
    """Haar Discrete Wavelet Transform: decomposes H×W×C → H/2×W/2×4C.

    Uses four Haar filters per paper Eq 4:
      f_LL (low-pass), f_LH, f_HL, f_HH (high-pass)
    """

    def __init__(self):
        super().__init__()
        # Haar filters: LL, LH, HL, HH
        self.register_buffer('f_LL', torch.tensor([[1., 1.], [1., 1.]]) / 2.0)
        self.register_buffer('f_LH', torch.tensor([[1., -1.], [1., -1.]]) / 2.0)
        self.register_buffer('f_HL', torch.tensor([[1., 1.], [-1., -1.]]) / 2.0)
        self.register_buffer('f_HH', torch.tensor([[1., -1.], [-1., 1.]]) / 2.0)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)

        def _conv2d_groups(t, kernel):
            # Apply a 2D convolution with kernel as a single-group filter.
            k = kernel.view(1, 1, 2, 2)
            return F.conv2d(t, k, stride=2, padding=0)

        LL = _conv2d_groups(x, self.f_LL)
        LH = _conv2d_groups(x, self.f_LH)
        HL = _conv2d_groups(x, self.f_HL)
        HH = _conv2d_groups(x, self.f_HH)

        out = torch.cat([LL, LH, HL, HH], dim=1)
        out = out.reshape(B, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4).reshape(B, C * 4, H // 2, W // 2)
        return out


class InverseHaarWaveletTransform(nn.Module):
    """Inverse Haar Discrete Wavelet Transform: H/2×W/2×4C → H×W×C."""

    def __init__(self):
        super().__init__()
        self.register_buffer('f_LL', torch.tensor([[1., 1.], [1., 1.]]) / 2.0)
        self.register_buffer('f_LH', torch.tensor([[1., -1.], [1., -1.]]) / 2.0)
        self.register_buffer('f_HL', torch.tensor([[1., 1.], [-1., -1.]]) / 2.0)
        self.register_buffer('f_HH', torch.tensor([[1., -1.], [-1., 1.]]) / 2.0)

    def forward(self, x):
        # x: (B, 4C, H_half, W_half)
        B, C4, H_half, W_half = x.shape
        C = C4 // 4
        H, W = H_half * 2, W_half * 2
        x = x.reshape(B, 4, C, H_half, W_half)
        LL = x[:, 0]
        LH = x[:, 1]
        HL = x[:, 2]
        HH = x[:, 3]

        out = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        # Manual inverse Haar: place subbands at pixel positions using 2x2 tiles
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        out[:, :, 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
        out[:, :, 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5
        return out


# ==================== Mamba SS2D (Four-directional scanning) ====================

class MambaSSM(nn.Module):
    """SS2D (Selective Scan 2D) — four-directional Mamba State Space Model.

    Upgraded from bidirectional (S=2) to SS2D four-directional scanning (S=4):
      - Scan 1: H-dim forward  (row-major, left→right → top→bottom)
      - Scan 2: H-dim reverse  (row-major flip)
      - Scan 3: W-dim forward  (col-major, top→bottom → left→right via HW transpose)
      - Scan 4: W-dim reverse  (col-major flip)

    All 4 outputs averaged. Per paper Tab 8d: d_state=1, expand=2.
    Includes output LayerNorm to stabilize SSM outputs.
    """

    def __init__(self, dim, d_state=1, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.has_mamba = HAS_MAMBA
        # Fallback convs
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1)
        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def _mamba_forward(self, x_seq):
        """Apply mamba on a sequence of shape (B, N, C)."""
        x_norm = self.norm(x_seq)
        return self.mamba(x_norm)  # no extra LN inside mamba

    def _scan_h(self, x):
        """H-dim scan: (B, C, H, W) → (B, H*W, C) row-major."""
        return x.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)

    def _scan_h_inv(self, x_seq, H, W):
        """Inverse of _scan_h: (B, H*W, C) → (B, C, H, W)."""
        return x_seq.transpose(1, 2).reshape(x_seq.shape[0], -1, H, W).contiguous()

    def _scan_w(self, x):
        """W-dim scan: (B, C, H, W) → (B, W*H, C) col-major (transpose HW)."""
        # Swap H,W so column becomes the scan dimension
        x_t = x.transpose(2, 3).contiguous()  # (B, C, W, H)
        return x_t.flatten(2).transpose(1, 2).contiguous()  # (B, W*H, C)

    def _scan_w_inv(self, x_seq, H, W):
        """Inverse of _scan_w: (B, W*H, C) → (B, C, H, W)."""
        x_t = x_seq.transpose(1, 2).reshape(
            x_seq.shape[0], -1, W, H
        ).contiguous()  # (B, C, W, H)
        return x_t.transpose(2, 3).contiguous()  # (B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.has_mamba:
            try:
                # ---- Scan 1: H-dim forward ----
                seq_h_fwd = self._scan_h(x)
                out_h_fwd = self._mamba_forward(seq_h_fwd)

                # ---- Scan 2: H-dim reverse ----
                seq_h_rev = seq_h_fwd.flip(1)
                out_h_rev = self._mamba_forward(seq_h_rev)
                out_h_rev = out_h_rev.flip(1)

                # ---- Scan 3: W-dim forward ----
                seq_w_fwd = self._scan_w(x)
                out_w_fwd = self._mamba_forward(seq_w_fwd)
                out_w_fwd = self._scan_w_inv(out_w_fwd, H, W)

                # ---- Scan 4: W-dim reverse ----
                seq_w_rev = seq_w_fwd.flip(1)
                out_w_rev = self._mamba_forward(seq_w_rev)
                out_w_rev = out_w_rev.flip(1)
                out_w_rev = self._scan_w_inv(out_w_rev, H, W)

                # Average all 4 directions (convert H-dim outputs back to 2D)
                out_h_fwd = self._scan_h_inv(out_h_fwd, H, W)
                out_h_rev = self._scan_h_inv(out_h_rev, H, W)
                x_out = (out_h_fwd + out_h_rev + out_w_fwd + out_w_rev) / 4.0

            except Exception:
                x_out = self.dwconv(x)
                x_out = self.pwconv(x_out)
        else:
            x_out = self.dwconv(x)
            x_out = self.pwconv(x_out)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_out = self.out_norm(x_out)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x_out


# ==================== Pure VSS Block (Mamba + Local, no cascaded wavelet) ====================

class PureVSSBlock(nn.Module):
    """Pure VSS Block — MambaSSM + Local DWConv, no cascaded wavelet.

    Designed specifically as the LL (low-frequency) branch processor in
    WaveletSemanticSplitterBlock.  Strips out all wavelet paths to prevent
    cascaded downsampling that would destroy small-object features.

    Architecture:
      Pre-LayerNorm → MambaSSM (four-directional global modeling)
                    → Local DWConv (k7, local structure)
      Weighted sum with learnable scalars → 1×1 proj → residual
    """

    def __init__(self, dim, d_state=1, d_conv=4, mamba_expand=2):
        super().__init__()
        self.dim = dim

        # Pre-norm
        self.pre_norm = nn.LayerNorm(dim)

        # Mamba path (global sequence modeling)
        self.mamba = MambaSSM(dim, d_state=d_state, d_conv=d_conv, expand=mamba_expand)

        # Local path (DWConv for local structure)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.local_bn = nn.BatchNorm2d(dim)
        self.local_act = nn.SiLU()

        # Learnable branch weights
        self.w_mamba = nn.Parameter(torch.ones(1))
        self.w_local = nn.Parameter(torch.ones(1))

        # Output projection
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_out_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pre-LayerNorm (channel-last → norm → channel-first)
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.pre_norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()

        # Mamba path
        x_mamba = self.mamba(x_norm)

        # Local path
        x_local = self.local_conv(x_norm)
        x_local = self.local_bn(x_local)
        x_local = self.local_act(x_local)

        # Weighted sum
        out = self.w_mamba * x_mamba + self.w_local * x_local

        out = self.proj_out(out)
        out = self.proj_out_bn(out)

        return x + out


# ==================== MK-DeConv Branch ====================

class MultiKernelDeConv(nn.Module):
    """Multi-Kernel Depthwise Convolution (MK-DeConv).

    Per paper Eq 6: split μc channels into n groups, each with k=(2j+1) DWConv, concat.
    Per paper Tab 8e: n=1 (k=3) is optimal for simplicity.
    """

    def __init__(self, dim, n_groups=1):
        super().__init__()
        self.n_groups = n_groups
        self.ch_per_group = dim // n_groups
        self.groups = nn.ModuleList()
        for j in range(1, n_groups + 1):
            k = 2 * j + 1  # 3, 5, 7, ...
            self.groups.append(nn.Sequential(
                nn.Conv2d(self.ch_per_group, self.ch_per_group, kernel_size=k, padding=k // 2,
                          groups=self.ch_per_group, bias=False),
                nn.BatchNorm2d(self.ch_per_group),
                nn.SiLU(),
            ))
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: (B, μc, H, W)
        if self.n_groups == 1:
            x_out = self.groups[0](x)
        else:
            splits = torch.chunk(x, self.n_groups, dim=1)
            processed = [group(s) for group, s in zip(self.groups, splits)]
            x_out = torch.cat(processed, dim=1)
        x_out = self.out_proj(x_out)
        x_out = self.out_bn(x_out)
        return x_out


# ==================== Utility Convs ====================

class DWConvSep(nn.Module):
    """Depthwise Separable Conv: k×k DWConv + 1×1 PWConv + BN + SiLU.

    Matches Conv arg format: (c1, c2, k=1, s=1) for direct yaml substitution.
    """

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.pwconv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        return x


# ==================== SEModule (Squeeze-and-Excitation) ====================

class SEModule(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention.

    Used between MultiKernelDeConv and 1×1 compress in the high-frequency
    branch of WaveletSemanticSplitterBlock.  Enables cross-subband
    (LH/HL/HH) interaction before fusion.

    Args:
        channels: input channels
        reduction: squeeze ratio (default 8)
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


# ==================== Wavelet Semantic Splitter Block ====================

class WaveletSemanticSplitterBlock(nn.Module):
    """Wavelet-Semantic Splitter (WSS) — frequency-domain semantic decoupling.

    Replaces blind channel-wise hard split with mathematically meaningful
    2D Haar DWT decomposition:
      - LL  (low-frequency / global structure) → PureVSSBlock
      - LH/HL/HH (high-frequency / edges & textures) → MultiKernelDeConv (group=3)

    Patch for mAP robustness:
      - High-Frequency Spatial Attention before IDWT
        Compensates for downsampling-induced fine-grained localization loss.

    Optional: cross-subband SE attention (use_se=True) for enhanced
              LH/HL/HH interaction.

    Architecture:
      DWT → [LL: PureVSSBlock] + [LH,HL,HH: MK-DeConv(g=3) → [SE?] → 1×1]
          → IDWT → +Identity(1×1) → 1×1 proj → residual

    Args:
        dim: input/output channels
        n_groups: MK-DeConv kernel groups (internally uses group=3 for subbands)
        d_state: Mamba SSM state dim (default 1)
        mamba_expand: Mamba expansion ratio (default 2)
        use_se: whether to insert SEModule between MK-DeConv and 1×1 compress
    """

    def __init__(self, dim, n_groups=1, d_state=1, mamba_expand=2, use_se=True,use_purevss=True, use_dap=True,hf_kernel_sizes=None):
        super().__init__()
        self.dim = dim
        self.use_se = use_se
        self.use_purevss = use_purevss        # ←
        self.use_dap = use_dap                # ←

        # ---- DWT / IDWT (parameter-free) ----
        self.dwt = HaarWaveletTransform()
        self.iwt = InverseHaarWaveletTransform()

        # ---- LL branch: Pure VSS Block (Mamba + Local) ----
        # LL is 1/4 channels (C out of 4C), but we treat LL as full C-dim
        # because DWT produces 4C total, so each subband = C.
        # Our DWT output is (B, 4C, H/2, W/2), LL = first C channels.
        if use_purevss:
            self.ll_processor = PureVSSBlock(dim, d_state=d_state, d_conv=4, mamba_expand=mamba_expand)

        # LL semantic enhancement: pure 1×1 convs to avoid spatial blur on H/2 scale
        # DWConv at reduced resolution would dilute small-object positioning cues
            self.ll_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        else:
            # 退化：简单 Conv 替代，参数量对齐
            self.ll_processor = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.SiLU()
            )
            self.ll_enhance = nn.Identity()

        # ---- High-Freq branch: MK-DeConv(groups=3) ----
        # 3 subbands (LH, HL, HH), each C channels → total 3C

        if use_dap:
            self.hf_conv = MultiKernelDeConv(3 * dim, n_groups=3)  # group=3 preserves direction

        # ---- Multi-Scale High-Frequency Spatial Attention ----
        # [V1] Shared spatial attention (direction-agnostic)

            attn_mid = max(min(8, dim // 4), 4)        # ← 12空格，属 if use_dap 块
            self.spatial_attn = nn.Sequential(           # ← 12空格，属 if use_dap 块
                nn.Conv2d(dim, attn_mid, 1, bias=False),
                nn.Conv2d(attn_mid, attn_mid, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(attn_mid, 1, 1, bias=False),
                nn.Sigmoid(),
        )
        else:
            # 退化：简单分组 Conv 替代
            self.hf_conv = nn.Sequential(
                nn.Conv2d(3 * dim, 3 * dim, 3, padding=1, groups=3),
                nn.BatchNorm2d(3 * dim), nn.SiLU()
            )
            self.use_se = False       # 无 CSE
            self.spatial_attn = None  # 无 HFSA

        # Cross-subband SE attention (default on for LH/HL/HH interaction)
        # [V2] reduced squeeze ratio 8→4 to preserve weak high-freq signals (e.g. crazing)
        if use_se:
            # self.hf_se = SEModule(3 * dim, reduction=8)  # [V1]
            self.hf_se = SEModule(3 * dim, reduction=4)     # [V2] weaker squeeze → stronger weak-signal retention

        # ---- Final fusion projection ----
        self.fusion_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # ---- Handle odd spatial dimensions (consistent with MCBWaveletBlock) ----
        pad_h = 1 if H % 2 == 1 else 0
        pad_w = 1 if W % 2 == 1 else 0
        x_padded = F.pad(x, (0, pad_w, 0, pad_h)) if pad_h > 0 or pad_w > 0 else x

        # ---- 2D Haar DWT: (B, C, H, W) → (B, 4C, H/2, W/2) ----
        x_dwt = self.dwt(x_padded)  # (B, 4C, H//2, W//2)

        # ---- Split subbands ----
        x_LL = x_dwt[:, :C, :, :]                         # (B, C, H//2, W//2)
        x_LH = x_dwt[:, C:2*C, :, :]                      # (B, C, H//2, W//2)
        x_HL = x_dwt[:, 2*C:3*C, :, :]                    # (B, C, H//2, W//2)
        x_HH = x_dwt[:, 3*C:, :, :]                       # (B, C, H//2, W//2)

        # ---- LL Branch: Pure VSS Block (global structure modeling) ----
        if self.use_purevss:
            y_LL = self.ll_processor(x_LL)                    # (B, C, H//2, W//2)
        # Semantic enhancement without spatial resolution loss
            y_LL = self.ll_enhance(y_LL) + y_LL
        else:
            y_LL = self.ll_processor(x_LL)   # 简单 Conv

        x_high = torch.cat([x_LH, x_HL, x_HH], dim=1)      # (B, 3C, H//2, W//2)
        y_high = self.hf_conv(x_high)                       # (B, 3C, H//2, W//2)

        if self.use_se:
            y_high = self.hf_se(y_high)

        # Split into three independent subband signals (LH, HL, HH)
        y_lh, y_hl, y_hh = torch.chunk(y_high, 3, dim=1)  # 3 × (B, C, H//2, W//2)

        # ---- [V1] Shared spatial attention (direction-agnostic) ----
        if self.spatial_attn is not None:
            y_lh = y_lh * self.spatial_attn(y_lh)
            y_hl = y_hl * self.spatial_attn(y_hl)
            y_hh = y_hh * self.spatial_attn(y_hh)

        # ---- Reassemble subbands for IDWT (direction-aware) ----
        y_dwt = torch.cat([y_LL, y_lh, y_hl, y_hh], dim=1)  # (B, 4C, H//2, W//2)

        # ---- IDWT: (B, 4C, H//2, W//2) → (B, C, H_pad, W_pad) ----
        y_recon = self.iwt(y_dwt)

        # ---- Crop back to original resolution ----
        if pad_h > 0 or pad_w > 0:
            y_recon = y_recon[:, :, :H, :W]

        # ---- Final 1×1 fusion (single residual) ----
        out = self.fusion_proj(y_recon)
        out = self.fusion_bn(out)

        return out + residual


class C3k2_Mamba_Block(nn.Module):
    """
    C3k2_Mamba_Block: MobileMamba backbone block using Wavelet-Semantic Splitting.

    Wraps WaveletSemanticSplitterBlock as a drop-in replacement for C3k2 in YAML.

    Architecture:
      - bridge: 1x1 Conv (c1 -> c2), enabled when c1 != c2.
                 In backbone where c1 == c2, bridge is Identity.
      - cv2: n × WaveletSemanticSplitterBlock (wss)
      - cv3: 1x1 Conv (c2 -> c2)
      - shortcut: residual connection when c1 == c2 and shortcut=True

    Args:
        c1: input channels
        c2: output channels
        n: number of wss block repeats
        shortcut: whether to use residual shortcut
        e_ratio: expansion ratio (not used, kept for yaml compatibility)
        tau_init: not used (kept for yaml backward compatibility)
        bottleneck_ratio: not used (kept for yaml backward compatibility)
        block_type: 'wavelet' (default), kept for yaml backward compatibility
    """

    def __init__(self, c1, c2, n=1, shortcut=False, e_ratio=1.0,use_purevss=True, use_dap=True):
        import sys
        print(f"[DEBUG] c1={c1}, c2={c2}, n={n}, shortcut={shortcut}, e_ratio={e_ratio}, use_purevss={use_purevss}, use_dap={use_dap}", file=sys.stderr)
        
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.add = shortcut and c1 == c2

        # bridge: 1x1 conv when channel changes (neck), Identity when same (backbone)
        if c1 != c2:
            self.bridge = Conv(c1, c2, 1)
        else:
            self.bridge = nn.Identity()

        # cv2: n WaveletSemanticSplitterBlock instances
        self.cv2 = nn.ModuleList([
            WaveletSemanticSplitterBlock(c2, n_groups=1, d_state=1, mamba_expand=2,
            use_purevss=use_purevss, use_dap=use_dap
            )
            for _ in range(n)
        ])

        # cv3: output projection
        self.cv3 = Conv(c2, c2, 1)

    def forward(self, x):
        x_bridge = self.bridge(x)
        x_out = x_bridge
        for block in self.cv2:
            x_out = block(x_out)
        x_out = self.cv3(x_out)
        if self.add:
            x_out = x_out + x_bridge
        return x_out
