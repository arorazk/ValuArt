"""
utils.py – Shared utility helpers for ValuArt.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str = "valuart") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Path helpers ────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def outputs_path(*parts: str) -> str:
    """Build a path relative to outputs/."""
    return os.path.join(project_root(), "outputs", *parts)


# ── Metric helpers ──────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ── Plot helpers ────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure and close it."""
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
