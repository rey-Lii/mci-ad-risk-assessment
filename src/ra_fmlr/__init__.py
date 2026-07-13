"""Compact public interface for V6.1-Hybrid-QC."""

from .data import PatientHistory, PatientVisit
from .evaluation import hazards_to_cumulative_risks
from .model import LongitudinalTransformer, ModelConfig, select_route

__all__ = [
    "PatientHistory", "PatientVisit", "ModelConfig",
    "LongitudinalTransformer", "select_route",
    "hazards_to_cumulative_risks",
]

__version__ = "0.6.1"
