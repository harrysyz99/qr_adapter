"""Low-fidelity and high-fidelity evaluators for candidate configurations."""

from qr_adaptor.evaluation.proxy import ProxyEvaluator
from qr_adaptor.evaluation.real_task import RealTaskEvaluator

__all__ = ["ProxyEvaluator", "RealTaskEvaluator"]
