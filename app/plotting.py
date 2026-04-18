"""Matplotlib plots for price, labels, and model probabilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_labels(df: pd.DataFrame, labels: pd.Series, out_path: Path) -> None:
    """Plot close price and highlight positive dip-zone labels."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], color="navy", linewidth=1.5, label="NIFTY Close")

    pos_idx = labels[labels == 1].index
    if len(pos_idx) > 0:
        ax.scatter(pos_idx, df.loc[pos_idx, "Close"], color="crimson", s=20, label="Historical Dip Zone")

    ax.set_title("NIFTY 50 Close with Historical Dip-Zone Labels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_buy_signals(df: pd.DataFrame, prob_series: pd.Series, out_path: Path, threshold: float = 0.65) -> None:
    """Plot close price and mark high-probability dip signals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], color="black", linewidth=1.2, label="NIFTY Close")

    signals = prob_series[prob_series >= threshold]
    if not signals.empty:
        ax.scatter(signals.index, df.loc[signals.index, "Close"], color="green", s=22, label="Buy Signal")

    ax.set_title("NIFTY 50 Buy Signals (Model Probability Threshold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability(prob_series: pd.Series, out_path: Path) -> None:
    """Plot dip probability over time."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(prob_series.index, prob_series.values, color="teal", linewidth=1.2)
    ax.axhline(0.65, color="crimson", linestyle="--", linewidth=1, label="DIP ZONE threshold")
    ax.axhline(0.45, color="orange", linestyle="--", linewidth=1, label="DIP WATCH threshold")

    ax.set_title("Dip Probability Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
