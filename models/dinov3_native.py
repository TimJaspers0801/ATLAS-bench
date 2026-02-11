# DINOv3 native implementation for loading Meta-trained DINOv3 checkpoints
# This is a minimal implementation to support loading SurgeNet2M-trained DINOv3 models
# The key difference from HF is layer naming: attention.q_proj vs attn.qkv, etc.

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.lambda1 = nn.Parameter(init_values * torch.ones((dim,)))

    def forward(self, x):
        return x.mul(self.lambda1) if self.inplace else x * self.lambda1


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ffn_layer=None, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.up_proj = nn.Linear(in_features, hidden_features)
        self.down_proj = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attention = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.layer_scale1 = LayerScale(dim, init_values=1e-6)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        self.layer_scale2 = LayerScale(dim, init_values=1e-6)

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, 0, embed_dim))
        self.patch_embeddings_weight = None
        
    def forward(self, x):
        x = self.proj(x)
        return x


class DINOv3ViT(nn.Module):
    """Minimal DINOv3 ViT compatible with Meta training checkpoints"""
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: str = "layernorm",
        n_storage_tokens: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # CLS and storage tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer_cls = nn.LayerNorm if norm_layer == "layernorm" else nn.LayerNorm
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer_cls
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer_cls(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Set num_prefix_tokens for ViTBackbone compatibility
        # 1 for CLS token + n_storage_tokens
        self.num_prefix_tokens = 1 + n_storage_tokens
        
        self.init_weights()

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'storage_tokens'):
            torch.nn.init.trunc_normal_(self.storage_tokens, std=0.02)
        torch.nn.init.zeros_(self.mask_token)

    def _prepare_tokens_with_masks(self, x, masks=None):
        """Internal helper for token preparation"""
        x = self.patch_embed.proj(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
        
        x = torch.cat([
            cls_token.expand(B, -1, -1),
            storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)
        
        return x, (H, W)

    def prepare_tokens_with_masks(self, x, masks=None):
        """Public method compatible with ViTBackbone"""
        x_prepared, _ = self._prepare_tokens_with_masks(x, masks)
        return x_prepared

    def forward(self, x):
        x, _ = self._prepare_tokens_with_masks(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Don't remove prefix tokens here - let ViTBackbone handle it
        # Just return all tokens for the wrapper to process
        return x
