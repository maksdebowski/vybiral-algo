"""
Skrypt generujący wykresy porównawcze algorytmów do pracy magisterskiej.

Porównuje metodę Fornasiera-Schnassa-Vybirala (CS + l1) z:
1. Active Subspace Method (ASM) — Constantine, 2015
2. OMP-based Recovery (oracle s=s*) — Tropp & Gilbert, 2007
3. OMP-based Recovery (overestimated s=10)

Generuje figury w formacie JPG w katalogu figures/.

Użycie:
    python generate_comparison_figures.py           # wszystkie
    python generate_comparison_figures.py --fig 1   # tylko nr 1
"""

import argparse
import os
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from src.algorithms import algorithm1, algorithm2
from src.baselines import (
    active_subspace_method,
    omp_algorithm1,
    omp_algorithm2,
)
from src.test_functions import (
    make_sparse_ridge_function,
    make_k_ridge_function,
)

# ---------------------------------------------------------------------------
# Styl
# ---------------------------------------------------------------------------
matplotlib.use("Agg")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
    }
)

COLORS = {
    "Vybiral (CS+l1)": "#2176AE",
    "OMP (s=s*)": "#E04F5F",
    "OMP (s=10)": "#C0392B",
    "ASM-FD": "#57B894",
    "SIR": "#F0A500",
}
MARKERS = {
    "Vybiral (CS+l1)": "o",
    "OMP (s=s*)": "s",
    "OMP (s=10)": "v",
    "ASM-FD": "^",
    "SIR": "D",
}
METHOD_ORDER = ["Vybiral (CS+l1)", "OMP (s=s*)", "OMP (s=10)", "ASM-FD"]
FIGDIR = "figures"


def _ensure_dir():
    os.makedirs(FIGDIR, exist_ok=True)


def _save_jpg(fig, name):
    path = f"{FIGDIR}/{name}.jpg"
    fig.savefig(path, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Zapisano {path}")


def _subspace_error(A_true, A_hat):
    """Błąd podprzestrzeni ||A^T A - Â^T Â||_F."""
    P_true = A_true.T @ A_true
    P_hat = A_hat.T @ A_hat
    return np.linalg.norm(P_true - P_hat, "fro")


def _direction_error(a_true, a_hat):
    """Błąd kierunku min_± ||a - (±â)||_2."""
    return min(np.linalg.norm(a_hat - a_true), np.linalg.norm(a_hat + a_true))


# ===================================================================
# Figura C1 — k=1: błąd vs m_Φ (Vybiral vs OMP vs ASM vs SIR)
# ===================================================================
def figure_c1():
    print("=" * 60)
    print("Figura C1: Porównanie metod (k=1), błąd vs m_Φ")
    print("=" * 60)

    d = 100
    sparsity = 3
    m_X = 25
    epsilon = 0.1
    n_trials = 8
    seed = 42
    m_Phi_values = [15, 20, 30, 40, 50, 60]

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    results = {}

    # --- Vybiral (CS + l1) ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, epsilon, rng=rng_t)
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=s*=3) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD (centralne różnice skończone, 2d ewaluacji na punkt) ---
    # Dla uczciwego porównania: budżet ASM-FD = m_X*(m_Phi+1) (budżet Vybirala)
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(2, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=1, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                a_hat = A_hat[0]
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ (Vybiral/OMP) / proporcjonalny budżet (ASM-FD)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        r"Porównanie metod: rekonstrukcja $a$ ($d=100$, $k=1$, $s=3$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC1_comparison_k1_vs_mPhi")


# ===================================================================
# Figura C2 — k=1: błąd vs budżet ewaluacji (fair comparison)
# ===================================================================
def figure_c2():
    print("=" * 60)
    print("Figura C2: Porównanie metod (k=1), błąd vs budżet ewaluacji")
    print("=" * 60)

    d = 100
    sparsity = 3
    epsilon = 0.1
    n_trials = 8
    seed = 42

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    budgets = [200, 500, 1000, 1500, 2000, 3000]

    results = {}

    # --- Vybiral: budget = m_X*(m_Phi+1), set m_X=25, vary m_Phi ---
    label = "Vybiral (CS+l1)"
    m_X_v = 25
    means, stds = [], []
    for budget in budgets:
        m_Phi = budget // m_X_v - 1
        if m_Phi < 5 or m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + budget * 100)
            try:
                a_hat, _ = algorithm1(
                    func, d, m_Phi, m_X_v, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] budget={budget} (m_Phi={m_Phi}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=3) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for budget in budgets:
        m_Phi = budget // m_X_v - 1
        if m_Phi < 5 or m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + budget * 100)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X_v, sparsity, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] budget={budget} (m_Phi={m_Phi}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for budget in budgets:
        m_Phi = budget // m_X_v - 1
        if m_Phi < 5 or m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + budget * 100)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X_v, 10, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] budget={budget} (m_Phi={m_Phi}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD: budget = 2d * m_samples ---
    label = "ASM-FD"
    means, stds = [], []
    for budget in budgets:
        m_asm = budget // (2 * d)
        if m_asm < 2:
            means.append(np.nan)
            stds.append(np.nan)
            print(
                f"  [{label}] budget={budget}: pominięto (m_asm={m_asm} < 2)"
            )
            continue
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + budget * 100)
            try:
                A_hat = active_subspace_method(
                    func, d, k=1, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, A_hat[0]))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] budget={budget} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        valid = [i for i, v in enumerate(m) if not np.isnan(v)]
        ax.errorbar(
            [budgets[i] for i in valid],
            [m[i] for i in valid],
            yerr=[s[i] for i in valid],
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel("Budżet ewaluacji funkcji")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(r"Porównanie metod: błąd vs budżet ($d=100$, $k=1$, $s=3$)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC2_comparison_k1_budget")


# ===================================================================
# Figura C3 — k=2: błąd podprzestrzeni vs m_Φ / budżet
# ===================================================================
def figure_c3():
    print("=" * 60)
    print("Figura C3: Porównanie metod (k=2), błąd podprzestrzeni")
    print("=" * 60)

    d = 80
    k = 2
    sparsity = 4  # rozmiar wspólnego nośnika macierzy A (|active_indices|)
    m_X = 35
    epsilon = 0.1
    n_trials = 6
    seed = 42
    m_Phi_values = [20, 30, 40, 50, 60, 70]

    rng_setup = np.random.default_rng(seed)
    func, A_true = make_k_ridge_function(
        d,
        k,
        active_indices=[0, 1, 2, 3],
        rng=rng_setup,
    )

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat, _ = algorithm2(
                    func, d, k, m_Phi, m_X, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=4) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD ---
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=k, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ / proporcjonalny budżet")
    ax.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax.set_title(rf"Porównanie metod: podprzestrzeń ($d={d}$, $k={k}$)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC3_comparison_k2_subspace")


# ===================================================================
# Figura C4 — Odporność na szum: porównanie metod
# ===================================================================
def figure_c4():
    print("=" * 60)
    print("Figura C4: Porównanie odporności na szum (k=1)")
    print("=" * 60)

    d = 100
    sparsity = 3
    m_Phi = 50
    m_X = 25
    epsilon = 0.1
    n_trials = 6
    seed = 42
    noise_sigmas = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for sigma in noise_sigmas:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + int(sigma * 1e6))
            try:
                a_hat, _ = algorithm1(
                    func, d, m_Phi, m_X, epsilon, noise_sigma=sigma, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] sigma={sigma}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=3) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for sigma in noise_sigmas:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + int(sigma * 1e6))
            try:
                a_hat = omp_algorithm1(
                    func,
                    d,
                    m_Phi,
                    m_X,
                    sparsity,
                    epsilon,
                    noise_sigma=sigma,
                    rng=rng_t,
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] sigma={sigma}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for sigma in noise_sigmas:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + int(sigma * 1e6))
            try:
                a_hat = omp_algorithm1(
                    func,
                    d,
                    m_Phi,
                    m_X,
                    10,
                    epsilon,
                    noise_sigma=sigma,
                    rng=rng_t,
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] sigma={sigma}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD ---
    label = "ASM-FD"
    means, stds = [], []
    total_budget = m_X * (m_Phi + 1)
    m_asm = max(3, total_budget // (2 * d))
    for sigma in noise_sigmas:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + int(sigma * 1e6))
            try:
                A_hat = active_subspace_method(
                    func,
                    d,
                    k=1,
                    m_samples=m_asm,
                    epsilon=epsilon,
                    noise_sigma=sigma,
                    rng=rng_t,
                )
                errors.append(_direction_error(a_true, A_hat[0]))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] sigma={sigma}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    x_labels = [str(s) if s > 0 else "0" for s in noise_sigmas]
    x_pos = np.arange(len(noise_sigmas))

    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            x_pos,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$ (odchylenie standardowe szumu)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(rf"Odporność na szum ($d={d}$, $m_\Phi={m_Phi}$, $k=1$)")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC4_comparison_noise")


# ===================================================================
# Figura C5 — Czas obliczeń: porównanie metod
# ===================================================================
def figure_c5():
    print("=" * 60)
    print("Figura C5: Porównanie czasu obliczeń (k=1)")
    print("=" * 60)

    d = 100
    sparsity = 3
    m_X = 25
    epsilon = 0.1
    seed = 42
    m_Phi_values = [15, 20, 30, 40, 50]
    n_trials = 3

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    times_list = []
    for m_Phi in m_Phi_values:
        trial_times = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            t0 = time.time()
            try:
                algorithm1(func, d, m_Phi, m_X, epsilon, rng=rng_t)
            except RuntimeError:
                pass
            trial_times.append(time.time() - t0)
        times_list.append(np.mean(trial_times))
        print(f"  [{label}] m_Phi={m_Phi}: {times_list[-1]:.2f}s")
    results[label] = times_list

    # --- OMP (s=3) ---
    label = "OMP (s=s*)"
    times_list = []
    for m_Phi in m_Phi_values:
        trial_times = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            t0 = time.time()
            try:
                omp_algorithm1(
                    func, d, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
            except Exception:
                pass
            trial_times.append(time.time() - t0)
        times_list.append(np.mean(trial_times))
        print(f"  [{label}] m_Phi={m_Phi}: {times_list[-1]:.2f}s")
    results[label] = times_list

    # --- OMP (s=10) ---
    label = "OMP (s=10)"
    times_list = []
    for m_Phi in m_Phi_values:
        trial_times = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            t0 = time.time()
            try:
                omp_algorithm1(func, d, m_Phi, m_X, 10, epsilon, rng=rng_t)
            except Exception:
                pass
            trial_times.append(time.time() - t0)
        times_list.append(np.mean(trial_times))
        print(f"  [{label}] m_Phi={m_Phi}: {times_list[-1]:.2f}s")
    results[label] = times_list

    # --- ASM-FD ---
    label = "ASM-FD"
    times_list = []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        trial_times = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            t0 = time.time()
            try:
                active_subspace_method(
                    func, d, k=1, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
            except Exception:
                pass
            trial_times.append(time.time() - t0)
        times_list.append(np.mean(trial_times))
        print(
            f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {times_list[-1]:.2f}s"
        )
    results[label] = times_list

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        ax.plot(
            m_Phi_values,
            results[label],
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ / proporcjonalny budżet")
    ax.set_ylabel("Czas [s]")
    ax.set_title(rf"Czas obliczeń ($d={d}$, $k=1$)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC5_comparison_time")


# ===================================================================
# Figura C6 — Skalowanie z wymiarem d: porównanie metod
# ===================================================================
def figure_c6():
    print("=" * 60)
    print("Figura C6: Skalowanie z wymiarem d")
    print("=" * 60)

    d_values = [30, 50, 80, 120, 200]
    sparsity = 3
    m_Phi = 40
    m_X = 25
    epsilon = 0.1
    n_trials = 6
    seed = 42

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for d in d_values:
        if m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        rng_s = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_s,
        )
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + d * 10000)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, epsilon, rng=rng_t)
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] d={d}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=3) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for d in d_values:
        if m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        rng_s = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_s,
        )
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + d * 10000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] d={d}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for d in d_values:
        if m_Phi >= d:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        rng_s = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_s,
        )
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + d * 10000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] d={d}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD ---
    label = "ASM-FD"
    means, stds = [], []
    total_budget = m_X * (m_Phi + 1)
    for d in d_values:
        m_asm = max(3, total_budget // (2 * d))
        rng_s = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_s,
        )
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + d * 10000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=1, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, A_hat[0]))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] d={d} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        valid = [i for i, v in enumerate(m) if not np.isnan(v)]
        ax.errorbar(
            [d_values[i] for i in valid],
            [m[i] for i in valid],
            yerr=[s[i] for i in valid],
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$d$ (wymiar przestrzeni)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        rf"Skalowanie z wymiarem $d$ ($m_\Phi={m_Phi}$, $k=1$, $s=3$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC6_comparison_dimension")


# ===================================================================
# Figura C7 — nieliniowa funkcja k=2 (sin): błąd podprzestrzeni vs m_Φ
# ===================================================================
def figure_c7():
    print("=" * 60)
    print("Figura C7: Porównanie metod (k=2, g(y)=sin(pi*||y||^2))")
    print("=" * 60)

    # Funkcja k=2 grzbietowa: f(x) = sin(pi*(x[2]^2 + x[3]^2))
    # Trudniejsza od kwadratowej g(y)=||y||^2 z figury C3:
    # - silnie nieliniowa (sin), oscyluje → trudniejsza aproksymacja Taylora
    # - ale gradient jest wszędzie niezerowy i rzadki kanonicznie (s=2)
    # - spełnia warunki modelu Vybirala (C2, gradient ograniczony)
    #
    # Dla wszystkich metod: aktywna podprzestrzeń = span(e_2, e_3).
    # Porównujemy z C3 (g=||y||^2) żeby pokazać jak trudność f wpływa
    # na każdą metodę.
    d = 80
    k = 2
    sparsity = 4
    m_X = 35
    epsilon = 0.1
    n_trials = 6
    seed = 42
    m_Phi_values = [20, 30, 40, 50, 60, 70]

    rng_setup = np.random.default_rng(seed)
    func, A_true = make_k_ridge_function(
        d,
        k,
        active_indices=[0, 1, 2, 3],
        rng=rng_setup,
    )
    # Zastąp g: zamiast ||y||^2 użyj sin(pi*||y||^2)
    A_mat = A_true.copy()

    def func(x):  # noqa: F811
        y = A_mat @ x
        return float(np.sin(np.pi * np.sum(y**2)))

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat, _ = algorithm2(
                    func, d, k, m_Phi, m_X, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=2) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD ---
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=k, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ / proporcjonalny budżet (ASM-FD)")
    ax.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax.set_title(
        rf"Porównanie metod: $g(y)=\sin(\pi\|y\|^2)$ ($d={d}$, $k={k}$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC7_nonlinear_sin_subspace")


def figure_c8():
    print("=" * 60)
    print("Figura C8: Porównanie metod (k=2, g nieliniowa nieradialna)")
    print("=" * 60)

    # Funkcja k=2 grzbietowa z g(y1, y2) = sin(y1) + 0.5*cos(2*y2) + 0.2*y1*y2
    # Nieradialna i silnie nieliniowa — trudniejsza od ||y||^2 (C3) i sin(pi*||y||^2) (C7):
    # - brak symetrii radialnej → ASM i Vybiral muszą rozróżnić oba kierunki
    # - gradient niejednorodny i asymetryczny
    # - spełnia warunki Vybirala: C^inf, ograniczone pochodne na sferze
    d = 80
    k = 2
    sparsity = 4
    m_X = 35
    epsilon = 0.1
    n_trials = 6
    seed = 42
    m_Phi_values = [20, 30, 40, 50, 60, 70]

    rng_setup = np.random.default_rng(seed)
    _, A_true = make_k_ridge_function(
        d,
        k,
        active_indices=[0, 1, 2, 3],
        rng=rng_setup,
    )
    A_mat = A_true.copy()

    def func(x):  # noqa: F811
        y = A_mat @ x
        return float(
            np.sin(y[0]) + 0.5 * np.cos(2.0 * y[1]) + 0.2 * y[0] * y[1]
        )

    results = {}

    # --- Vybiral ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat, _ = algorithm2(
                    func, d, k, m_Phi, m_X, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=2) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, sparsity, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (overestimated s=10) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = omp_algorithm2(
                    func, d, k, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD ---
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=k, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_subspace_error(A_true, A_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ / proporcjonalny budżet (ASM-FD)")
    ax.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax.set_title(
        rf"Porównanie metod: $g(y_1,y_2)=\sin(y_1)+\frac{{1}}{{2}}\cos(2y_2)+0.2y_1 y_2$ ($d={d}$, $k={k}$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC8_nonlinear_nonradial_subspace")


# ===================================================================
# Figura C9 — Vybiral vs OMP: wpływ przeszacowania rzadkości s
# Vybiral (CS+l1) nie wymaga znajomości s — przewaga nad OMP(s>>s*)
# ===================================================================
def figure_c9():
    print("=" * 60)
    print("Figura C9: Vybiral vs OMP — przeszacowanie s (d=200, k=1)")
    print("=" * 60)

    # Scenariusz: s=3, d=200. Vybiral l1 automatycznie promuje rzadkość.
    # OMP z wyrocznim s=3 jest najlepszy z możliwych (dolna bariera).
    # Vybiral powinien dorównać OMP(s=3) i bić OMP z przeszacowanym s.
    # ASM-FD z uczciwym budżetem m_asm = total_budget / (2*d).
    d = 200
    sparsity_true = 3
    m_X = 30
    epsilon = 0.1
    n_trials = 8
    seed = 42
    m_Phi_values = [20, 30, 40, 50, 60, 80]

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[5, 17, 42],
        rng=rng_setup,
    )

    results = {}

    # --- Vybiral (CS+l1) — nie wymaga znajomości s ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, epsilon, rng=rng_t)
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP z wyroczną s=s*=3 (dolna bariera) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, sparsity_true, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP z przeszacowanym s=10 ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD z uczciwym budżetem ---
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func, d, k=1, m_samples=m_asm, epsilon=epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, A_hat[0]))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(rf"Vybiral vs OMP: $d={d}$, $k=1$, $s=3$, $g(t)=\cos(\pi t)$")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC9_vybiral_advantage_highd")


# ===================================================================
# Figura C10 — silnie oscylujące g: ASM-FD traci, Vybiral wygrywa
#
# g(t) = sin(ω·t) z dużym ω (wysoka częstotliwość).
# ASM-FD szacuje ∇f przez centralne różnice w każdym z d kierunków
# → koszt 2d ewaluacji/próbkę → przy ustalonym budżecie mało próbek.
# Dodatkowo: duże ω → ||g''|| duże → błąd Taylora O(ε²||g''||) duży
# → ASM musi użyć małego ε, ale małe ε wzmacnia szum num. → dylemat.
# Vybiral liczy tylko m_Phi ilorazów różnicowych/próbkę (m_Phi << d)
# i uśrednia po SVD → odporniejszy na duże ω i mały budżet.
# ===================================================================
def figure_c10():
    print("=" * 60)
    print("Figura C10: ASM-FD vs Vybiral — wysoka częstotliwość g")
    print("=" * 60)

    # d=300, s=2, g(t) = sin(omega*t), omega=15
    # Przy stałym budżecie total = m_X*(m_Phi+1):
    #   ASM:     m_asm = budget / (2*d) → małe m_asm przy dużym d
    #   Vybiral: m_X próbek, każda m_Phi rzutowań → koszt m_X*(m_Phi+1)
    d = 300
    k = 1
    sparsity_true = 2
    omega = 15.0
    m_X = 30
    epsilon = 0.05
    n_trials = 8
    seed = 42
    m_Phi_values = [20, 30, 40, 50, 60, 80]

    rng_setup = np.random.default_rng(seed)
    # Losowy rzadki wektor a z niezerowymi składowymi w indeksach 7 i 53
    a_true = np.zeros(d)
    vals = rng_setup.standard_normal(sparsity_true)
    a_true[[7, 53]] = vals
    a_true /= np.linalg.norm(a_true)

    def func(x):
        return float(np.sin(omega * float(a_true @ x)))

    results = {}

    # --- Vybiral (CS+l1) ---
    label = "Vybiral (CS+l1)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, epsilon, rng=rng_t)
                errors.append(_direction_error(a_true, a_hat))
            except RuntimeError:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (oracle s=2) ---
    label = "OMP (s=s*)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, sparsity_true, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- OMP (s=10 przeszacowane) ---
    label = "OMP (s=10)"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat = omp_algorithm1(
                    func, d, m_Phi, m_X, 10, epsilon, rng=rng_t
                )
                errors.append(_direction_error(a_true, a_hat))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi}: {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- ASM-FD: koszt 2d/próbkę → przy tym samym budżecie mało próbek
    # Używamy tego samego epsilon co Vybiral — tu tkwi dylemat ASM:
    # epsilon=0.05 jest za duże dla sin(15t) → błąd Taylora ~ω²ε²/6 ≈ 0.09
    label = "ASM-FD"
    means, stds = [], []
    for m_Phi in m_Phi_values:
        total_budget = m_X * (m_Phi + 1)
        m_asm = max(3, total_budget // (2 * d))
        errors = []
        for trial in range(n_trials):
            rng_t = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat = active_subspace_method(
                    func,
                    d,
                    k=k,
                    m_samples=m_asm,
                    epsilon=epsilon,
                    rng=rng_t,
                )
                errors.append(_direction_error(a_true, A_hat[0]))
            except Exception:
                errors.append(2.0)
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        print(f"  [{label}] m_Phi={m_Phi} (m_asm={m_asm}): {means[-1]:.4f}")
    results[label] = (means, stds)

    # --- Wykres ---
    fig, ax = plt.subplots()
    for label in METHOD_ORDER:
        m, s = results[label]
        ax.errorbar(
            m_Phi_values,
            m,
            yerr=s,
            marker=MARKERS[label],
            color=COLORS[label],
            label=label,
        )
    ax.set_xlabel(r"$m_\Phi$ / proporcjonalny budżet (ASM-FD)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        rf"Wysoka częst. $g(t)=\sin({int(omega)}\pi t)$:"
        rf" $d={d}$, $k=1$, $s=2$"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    _save_jpg(fig, "figC10_highfreq_asm_vs_vybiral")


FIGURES = {
    1: figure_c1,
    2: figure_c2,
    3: figure_c3,
    4: figure_c4,
    5: figure_c5,
    6: figure_c6,
    7: figure_c7,
    8: figure_c8,
    9: figure_c9,
    10: figure_c10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Wykresy porownawcze algorytmow"
    )
    parser.add_argument(
        "--fig",
        nargs="*",
        type=int,
        default=None,
        help="Numery figur (domyslnie: wszystkie)",
    )
    args = parser.parse_args()
    _ensure_dir()

    figs_to_run = args.fig if args.fig else sorted(FIGURES.keys())
    t_start = time.time()

    for fig_num in figs_to_run:
        if fig_num not in FIGURES:
            print(
                "UWAGA: Figura {fig_num} nie istnieje (dostepne:"
                + f" {sorted(FIGURES.keys())})"
            )
            continue
        t0 = time.time()
        FIGURES[fig_num]()
        print(f"  Czas: {time.time() - t0:.1f}s\n")

    print(f"\nGotowe! Wykresy w katalogu {FIGDIR}/")
    print(f"Laczny czas: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
