"""
markov_hazard_fedavg
====================
マルコフ劣化ハザード推計 × FedAvg デモパッケージ
"""
from src.model import MarkovHazardModel, ALLOWED_TRANSITIONS, N_PARAMS
from src.likelihood import log_likelihood_batch_vectorized, compute_nll_and_grad
from src.data_utils import generate_synthetic_data, normalize_local, extract_transition_pairs
from src.client import FedAvgClient, ClientUpdate
from src.server import FedAvgServer, RoundLog

__all__ = [
    "MarkovHazardModel",
    "ALLOWED_TRANSITIONS",
    "N_PARAMS",
    "log_likelihood_batch_vectorized",
    "compute_nll_and_grad",
    "generate_synthetic_data",
    "normalize_local",
    "extract_transition_pairs",
    "FedAvgClient",
    "ClientUpdate",
    "FedAvgServer",
    "RoundLog",
]
