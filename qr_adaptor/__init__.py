"""
QR-Adaptor: Balancing Fidelity and Plasticity for Mixed-Precision Fine-Tuning.

Joint per-layer quantization bit-width and LoRA adapter rank optimization
via a three-phase framework:

  Phase I:   Fidelity Sensitivity Profiling
  Phase II:  Discrete Landscape Exploration (constrained NSGA-II)
  Phase III: Bayesian Frontier Refinement (Matern-5/2 GP + Expected Improvement)
"""

__version__ = "1.0.0"
__author__ = "QR-Adaptor Authors"

from qr_adaptor.core.config import QRAdaptorConfig, ConfigEncoding
from qr_adaptor.core.importance import Importance
from qr_adaptor.core.memory import MemoryModel
from qr_adaptor.evaluation.proxy import ProxyEvaluator
from qr_adaptor.evaluation.real_task import RealTaskEvaluator
from qr_adaptor.surrogate.mlp import GELUSurrogateNet, SurrogateMLPPromotion
from qr_adaptor.search.phase2_evolution import PhaseIIEvolution
from qr_adaptor.search.phase3_bo import PhaseIIIBO
from qr_adaptor.qradaptor import QRAdaptor

__all__ = [
    "QRAdaptorConfig",
    "ConfigEncoding",
    "Importance",
    "MemoryModel",
    "ProxyEvaluator",
    "RealTaskEvaluator",
    "GELUSurrogateNet",
    "SurrogateMLPPromotion",
    "PhaseIIEvolution",
    "PhaseIIIBO",
    "QRAdaptor",
]
