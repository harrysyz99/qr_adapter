"""Surrogate models for multi-fidelity LF→HF performance prediction."""

from qr_adaptor.surrogate.mlp import GELUSurrogateNet, SurrogateMLPPromotion

__all__ = ["GELUSurrogateNet", "SurrogateMLPPromotion"]
