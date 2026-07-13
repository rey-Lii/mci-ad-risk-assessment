"""V6.1 candidate: gap-controlled, latest-anchored modular Transformer.

This candidate preserves the frozen V6 tensor contract and resource-adaptive
fusion design. The training launcher zeros the standardized raw gap channel
(token channel 4) consistently for training, validation, and held-out
prediction.

Within each module, the temporal CLS representation is explicitly fused with
the latest observed token representation. This prevents the historical summary
from entirely replacing the current clinical state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = 8
    context_dim: int = 6
    n_modules: int = 5
    n_intervals: int = 4
    d_model: int = 48
    nhead: int = 4
    module_layers: int = 1
    fusion_layers: int = 1
    feedforward_dim: int = 96
    dropout: float = 0.15


class LatestAnchoredSharedModuleTemporalEncoder(nn.Module):
    """Shared module encoder with explicit current-state anchoring."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_projection = nn.Sequential(
            nn.Linear(config.token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.module_embedding = nn.Embedding(config.n_modules, config.d_model)
        self.module_cls = nn.Parameter(torch.zeros(1, 1, config.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.module_layers,
        )
        self.history_norm = nn.LayerNorm(config.d_model)
        self.latest_norm = nn.LayerNorm(config.d_model)
        self.latest_history_fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        nn.init.normal_(self.module_cls, std=0.02)

    def forward(
        self,
        token_features: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_modules, seq_len, _ = token_features.shape
        if n_modules != self.config.n_modules:
            raise ValueError(
                f"Expected {self.config.n_modules} modules, got {n_modules}."
            )

        projected = self.token_projection(token_features)
        module_ids = torch.arange(n_modules, device=token_features.device)
        module_emb = self.module_embedding(module_ids).view(
            1, n_modules, 1, -1
        )
        projected = projected + module_emb

        projected = projected.reshape(
            batch_size * n_modules,
            seq_len,
            self.config.d_model,
        )
        flat_mask = observation_mask.reshape(
            batch_size * n_modules,
            seq_len,
        ).bool()

        cls = self.module_cls.expand(
            batch_size * n_modules,
            -1,
            -1,
        )
        cls_module_emb = self.module_embedding(module_ids).view(
            n_modules, 1, -1
        )
        cls_module_emb = (
            cls_module_emb.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * n_modules, 1, -1)
        )
        cls = cls + cls_module_emb

        sequence = torch.cat([cls, projected], dim=1)
        cls_valid = torch.ones(
            (batch_size * n_modules, 1),
            dtype=torch.bool,
            device=token_features.device,
        )
        valid_mask = torch.cat([cls_valid, flat_mask], dim=1)

        encoded = self.encoder(
            sequence,
            src_key_padding_mask=~valid_mask,
        )

        history = self.history_norm(encoded[:, 0])

        # Valid observation slots are contiguous and ordered earliest -> latest.
        # Position 0 in encoded is the module CLS token.
        lengths = flat_mask.sum(dim=1)
        latest_positions = lengths.clamp_min(1)
        row_indices = torch.arange(
            batch_size * n_modules,
            device=token_features.device,
        )
        latest = encoded[row_indices, latest_positions]
        latest = self.latest_norm(latest)

        fused = self.latest_history_fusion(
            torch.cat([latest, history], dim=-1)
        )
        return fused.reshape(batch_size, n_modules, self.config.d_model)


class GapControlledLatestAnchoredModularTemporalTransformer(nn.Module):
    """V6.1 candidate with availability-aware cross-module fusion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.module_encoder = LatestAnchoredSharedModuleTemporalEncoder(config)
        self.context_projection = nn.Sequential(
            nn.Linear(config.context_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.fusion_type_embedding = nn.Embedding(
            config.n_modules + 2,
            config.d_model,
        )
        self.global_cls = nn.Parameter(torch.zeros(1, 1, config.d_model))

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_layer,
            num_layers=config.fusion_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_intervals),
        )
        nn.init.normal_(self.global_cls, std=0.02)

    def forward(
        self,
        token_features: torch.Tensor,
        observation_mask: torch.Tensor,
        module_available: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        module_tokens = self.module_encoder(
            token_features,
            observation_mask,
        )
        context_token = self.context_projection(context_features).unsqueeze(1)
        batch_size = token_features.shape[0]

        global_token = self.global_cls.expand(batch_size, -1, -1)
        fusion_tokens = torch.cat(
            [global_token, context_token, module_tokens],
            dim=1,
        )

        type_ids = torch.arange(
            self.config.n_modules + 2,
            device=token_features.device,
        ).view(1, -1)
        fusion_tokens = (
            fusion_tokens + self.fusion_type_embedding(type_ids)
        )

        prefix_valid = torch.ones(
            (batch_size, 2),
            dtype=torch.bool,
            device=token_features.device,
        )
        valid_mask = torch.cat(
            [prefix_valid, module_available.bool()],
            dim=1,
        )
        fused = self.fusion_encoder(
            fusion_tokens,
            src_key_padding_mask=~valid_mask,
        )
        return self.head(fused[:, 0])


# Readable public aliases; frozen class names remain unchanged for checkpoints.
LongitudinalTransformer = GapControlledLatestAnchoredModularTemporalTransformer


def select_route(history_depth: int) -> str:
    """Route one assessment to Snapshot and repeated assessments to Transformer."""
    depth = int(history_depth)
    if depth < 1:
        raise ValueError("history_depth must be at least 1.")
    return "snapshot" if depth == 1 else "longitudinal_transformer"


__all__ = [
    "ModelConfig",
    "LatestAnchoredSharedModuleTemporalEncoder",
    "GapControlledLatestAnchoredModularTemporalTransformer",
    "LongitudinalTransformer",
    "select_route",
]
