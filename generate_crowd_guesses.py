#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_all_data.py

1. Generates random LP instances → data/linear_programming_data_<n>.pkl
2. Calls generate_crowd_templates_from_pkl() to produce crowd-guess CSVs.
"""

import os
import pickle as pkl
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Helper: generate a single random LP instance
# ----------------------------------------------------------------------
def make_random_lp(
    n_vars: int,
    seed: Optional[int] = None,
    prob_positive: float = 0.4,
    c_scale: float = 10.0,
) -> dict:
    """
    Returns a dict that mimics the format expected by the crowd-template code:
        {'c': np.ndarray of shape (n_vars,) }
    """
    rng = np.random.default_rng(seed)

    # Objective coefficients: mix of positive, negative and zero
    c = rng.uniform(-c_scale, c_scale, size=n_vars)

    # Force a controllable fraction to be positive (optional)
    if prob_positive > 0:
        mask = rng.random(n_vars) < prob_positive
        c[mask] = np.abs(c[mask])

    return {"c": c.astype(float)}


# ----------------------------------------------------------------------
# 2. Generate a batch of .pkl files
# ----------------------------------------------------------------------
def generate_lp_pickles(
    data_folder: str = "data",
    sizes: List[int] = None,
    n_per_size: int = 1,
    seed: int = 123,
) -> None:
    """
    Creates `data/linear_programming_data_<n>.pkl` files.
    For each size in `sizes` we write `n_per_size` independent instances.
    """
    if sizes is None:
        sizes = [10, 20, 50, 100, 200]          # default test sizes

    data_path = Path(data_folder)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    base_seed = rng.integers(0, 2**32 - 1)

    print(f"Generating LP pickles → {data_path.resolve()}")
    for n in sizes:
        for i in range(n_per_size):
            file_seed = base_seed + i * 1000 + n   # deterministic per file
            lp = make_random_lp(n_vars=n, seed=file_seed)

            filename = f"linear_programming_data_{n}.pkl"
            if n_per_size > 1:
                filename = f"linear_programming_data_{n}_run{i+1}.pkl"

            pkl_path = data_path / filename
            with pkl_path.open("wb") as f:
                pkl.dump(lp, f)

            print(f"  → {pkl_path.name}  (n={n}, seed={file_seed})")
    print()


# ----------------------------------------------------------------------
# 3. Crowd-template generator (your original code, slightly cleaned)
# ----------------------------------------------------------------------
def generate_crowd_templates_from_pkl(
    data_folder: str = "data",
    output_subfolder: str = "crowd_templates",
    n_fake_guesses: int = 30,
    seed: Optional[int] = 42,
    max_active_per_guess: int = 5,
    min_val: int = 1,
    max_val: int = 5,
) -> None:
    """Same logic as in your question, now with a few safety checks."""
    rng = np.random.default_rng(seed)
    data_path = Path(data_folder)
    out_path = data_path / output_subfolder
    out_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"linear_programming_data_(\d+)(?:_run\d+)?\.pkl")
    pkl_files = sorted(data_path.glob("linear_programming_data_*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {data_folder}")
        return

    print(f"Found {len(pkl_files)} .pkl files. Generating crowd templates...\n")

    for pkl_file in pkl_files:
        m = pattern.search(pkl_file.name)
        if not m:
            print(f"Skipping malformed name: {pkl_file.name}")
            continue

        n_vars = int(m.group(1))
        print(f"Processing {pkl_file.name} → n = {n_vars}")

        # ---- load ----
        with pkl_file.open("rb") as f:
            data = pkl.load(f)

        c = np.asarray(data["c"], dtype=float)
        if c.shape != (n_vars,):
            raise ValueError(
                f"Size mismatch in {pkl_file.name}: filename says {n_vars}, c has {c.shape}"
            )

        # ---- probability vector (bias toward positive coeffs) ----
        pos_c = np.maximum(c, 0.0)
        if pos_c.sum() == 0:
            probs = None
            print("  All c[i] ≤ 0 → uniform sampling")
        else:
            probs = pos_c / pos_c.sum()
            n_pos = int((probs > 0).sum())
            print(f"  {n_pos} variables with positive c[i]")

        # ---- generate guesses ----
        rows: List[list] = []
        columns = [f"x{i}" for i in range(n_vars)]

        for _ in range(n_fake_guesses):
            x = np.zeros(n_vars, dtype=int)

            k = rng.integers(1, min(max_active_per_guess, n_vars) + 1)

            if probs is None:
                active = rng.choice(n_vars, size=k, replace=False)
            else:
                pos_idx = np.flatnonzero(probs > 0)
                if k <= len(pos_idx):
                    active = rng.choice(pos_idx, size=k, replace=False)
                else:
                    active = rng.choice(n_vars, size=k, replace=False, p=probs)

            values = rng.integers(min_val, max_val + 1, size=k)
            x[active] = values
            rows.append(x.tolist())

        # ---- save ----
        df = pd.DataFrame(rows, columns=columns)
        csv_path = out_path / f"crowd_guesses_{n_vars}.csv"
        # If multiple runs exist for the same n, keep only the *last* one (or change naming)
        df.to_csv(csv_path, index=False)
        print(f"  → Saved {csv_path.name} ({len(df)} rows, {n_vars} cols)")

    print(f"\nDone! Crowd templates → {out_path.resolve()}")


# ----------------------------------------------------------------------
# 4. Main driver – generate everything in one go
# ----------------------------------------------------------------------
def main() -> None:
    # ---- 1. Create the LP pickle files ----
    lp_sizes = [10, 20, 50, 100, 200]          # change / extend as you like
    generate_lp_pickles(
        data_folder="data",
        sizes=lp_sizes,
        n_per_size=2,          # 2 independent instances per size (optional)
        seed=123,
    )

    # ---- 2. Generate crowd-guess CSVs ----
    generate_crowd_templates_from_pkl(
        data_folder="data",
        output_subfolder="crowd_templates",
        n_fake_guesses=30,
        seed=42,
        max_active_per_guess=5,
        min_val=1,
        max_val=5,
    )


if __name__ == "__main__":
    main()