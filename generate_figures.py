"""
Skrypt generujący wykresy do pracy magisterskiej.

Generuje 7 figur w katalogu figures/ w formacie JPG
gotowych do zamieszczenia w pracy magisterskiej.

Użycie:
    python generate_figures.py           # wszystkie wykresy
    python generate_figures.py --fig 1   # tylko Rysunek nr 1
    python generate_figures.py --fig 1 3 5  # wybrane figury
"""

import argparse
import os
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.algorithms import algorithm1, algorithm2
from src.sampling import sample_sphere, bernoulli_matrix
from src.l1_solver import l1_minimize
from src.test_functions import (
    make_sparse_ridge_function,
    make_k_ridge_function,
)

# ---------------------------------------------------------------------------
# Styl wykresów — profesjonalny, czytelny, kompatybilny z LaTeX
# ---------------------------------------------------------------------------
matplotlib.use("Agg")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (7, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
    }
)

COLORS = ["#2176AE", "#E04F5F", "#57B894", "#F0A500", "#7B68EE", "#FF6B6B"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
FIGDIR = "figures"


def _ensure_dir():
    os.makedirs(FIGDIR, exist_ok=True)


# ===================================================================
# Rysunek 1 — Algorytm 1: zbieżność błędu vs m_Φ przy różnych d
# ===================================================================
def figure1():
    print("=" * 60)
    print("Rysunek 1: Algorytm 1 — błąd rekonstrukcji vs m_Φ")
    print("=" * 60)

    dims = [50, 100, 200]
    m_Phi_values = [15, 20, 30, 40, 50, 60, 70, 80]
    m_X = 30
    n_trials = 10
    seed = 42

    fig, ax = plt.subplots()

    for idx, d in enumerate(dims):
        rng_setup = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_setup,
        )

        mean_errors, std_errors = [], []
        for m_Phi in m_Phi_values:
            if m_Phi >= d:
                mean_errors.append(np.nan)
                std_errors.append(np.nan)
                continue
            errors = []
            for trial in range(n_trials):
                rng_trial = np.random.default_rng(
                    seed + trial + m_Phi * 1000 + d * 100000
                )
                try:
                    a_hat, _ = algorithm1(
                        func, d, m_Phi, m_X, epsilon=0.1, rng=rng_trial
                    )
                    err = min(
                        np.linalg.norm(a_hat - a_true),
                        np.linalg.norm(a_hat + a_true),
                    )
                    errors.append(err)
                except RuntimeError:
                    errors.append(2.0)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
            print(
                f"  d={d}, m_Φ={m_Phi}: {np.mean(errors):.4f}"
                + "+/-"
                + f"{np.std(errors):.4f}"
            )

        valid = [i for i, v in enumerate(mean_errors) if not np.isnan(v)]
        ax.errorbar(
            [m_Phi_values[i] for i in valid],
            [mean_errors[i] for i in valid],
            yerr=[std_errors[i] for i in valid],
            marker=MARKERS[idx],
            color=COLORS[idx],
            label=rf"$d = {d}$",
        )

    ax.set_xlabel(r"$m_\Phi$ (liczba kierunków pomiarowych)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        r"Algorytm 1: błąd rekonstrukcji wektora $a$ ($k=1$, $m_X=30$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig1_alg1_convergence.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  → Zapisano fig1_alg1_convergence.jpg\n")


# ===================================================================
# Rysunek 2 — Algorytm 1: porównanie prawdziwego i odzyskanego wektora
# ===================================================================
def figure2():
    print("=" * 60)
    print("Rysunek 2: Algorytm 1 — prawdziwy vs odzyskany wektor a")
    print("=" * 60)

    d = 100
    rng_setup = np.random.default_rng(0)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    rng = np.random.default_rng(42)
    a_hat, _ = algorithm1(func, d, m_Phi=50, m_X=25, epsilon=0.1, rng=rng)

    # Dopasuj znak
    if np.linalg.norm(a_hat + a_true) < np.linalg.norm(a_hat - a_true):
        a_hat = -a_hat

    # Pokaż pierwsze 15 współrzędnych (aktywne + reszta)
    n_show = 15
    indices = np.arange(n_show)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    width = 0.35
    ax1.bar(
        indices - width / 2,
        a_true[:n_show],
        width,
        color=COLORS[0],
        label=r"$a$ (prawdziwy)",
        alpha=0.85,
    )
    ax1.bar(
        indices + width / 2,
        a_hat[:n_show],
        width,
        color=COLORS[1],
        label=r"$\hat{a}$ (odzyskany)",
        alpha=0.85,
    )
    ax1.set_ylabel("Wartość współrzędnej")
    ax1.set_title(
        r"Algorytm 1: porównanie $a$ i $\hat{a}$ ($d=100$,"
        + r" $m_\Phi=50$, $m_X=25$)"
    )
    ax1.legend()
    ax1.axhline(0, color="gray", linewidth=0.5)

    # Błąd bezwzględny na wszystkich współrzędnych
    abs_diff = np.abs(a_hat - a_true)
    ax2.bar(np.arange(d), abs_diff, color=COLORS[2], alpha=0.7, width=1.0)
    ax2.set_xlabel("Indeks współrzędnej")
    ax2.set_ylabel(r"$|a_i - \hat{a}_i|$")
    ax2.set_title("Błąd bezwzględny na poszczególnych współrzędnych")
    ax2.set_xlim(-0.5, d - 0.5)

    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig2_alg1_vector.jpg", format="jpeg", bbox_inches="tight"
    )
    plt.close(fig)
    print(f"  Błąd: {np.linalg.norm(a_hat - a_true):.6f}")
    print("  → Zapisano fig2_alg1_vector.jpg\n")


# ===================================================================
# Rysunek 3 — Algorytm 2: zbieżność błędu podprzestrzeni vs m_Φ
# ===================================================================
def figure3():
    print("=" * 60)
    print("Rysunek 3: Algorytm 2 — błąd podprzestrzeni vs m_Φ")
    print("=" * 60)

    k_values = [2, 3]
    d = 80
    m_Phi_values = [20, 30, 40, 50, 60, 70]
    m_X = 40
    n_trials = 8
    seed = 42

    fig, ax = plt.subplots()

    for idx, k in enumerate(k_values):
        rng_setup = np.random.default_rng(seed)
        func, A_true = make_k_ridge_function(
            d,
            k,
            active_indices=list(range(k * 3)),
            rng=rng_setup,
        )
        P_true = A_true.T @ A_true

        mean_errors, std_errors = [], []
        for m_Phi in m_Phi_values:
            errors = []
            for trial in range(n_trials):
                rng_trial = np.random.default_rng(
                    seed + trial + m_Phi * 1000 + k * 500000
                )
                try:
                    A_hat, _ = algorithm2(
                        func, d, k, m_Phi, m_X, epsilon=0.1, rng=rng_trial
                    )
                    P_hat = A_hat.T @ A_hat
                    errors.append(np.linalg.norm(P_true - P_hat, "fro"))
                except RuntimeError:
                    errors.append(2.0)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
            print(
                f"  k={k}, m_Φ={m_Phi}: {np.mean(errors):.4f}"
                + " +/- "
                + f"{np.std(errors):.4f}"
            )

        ax.errorbar(
            m_Phi_values,
            mean_errors,
            yerr=std_errors,
            marker=MARKERS[idx],
            color=COLORS[idx],
            label=rf"$k = {k}$",
        )

    ax.set_xlabel(r"$m_\Phi$ (liczba kierunków pomiarowych)")
    ax.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax.set_title(
        "Algorytm 2: błąd rekonstrukcji podprzestrzeni"
        + rf" ($d={d}$, $m_X={m_X}$)"
    )
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig3_alg2_convergence.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  → Zapisano fig3_alg2_convergence.jpg\n")


# ===================================================================
# Rysunek 4 — Algorytm 2: wartości osobliwe X̂ (przerwa spektralna)
# ===================================================================
def figure4():
    print("=" * 60)
    print("Rysunek 4: Algorytm 2 — wartości osobliwe macierzy X̂")
    print("=" * 60)

    d = 80
    k = 2
    m_Phi = 50
    m_X = 40
    seed = 42

    rng_setup = np.random.default_rng(seed)
    func, A_true = make_k_ridge_function(
        d,
        k,
        active_indices=[0, 1, 2, 3],
        rng=rng_setup,
    )

    # Uruchom algorytm ręcznie, żeby wyciągnąć wartości osobliwe
    rng = np.random.default_rng(seed + 1)
    Xi = sample_sphere(d, m_X, rng)
    Phi = bernoulli_matrix(m_Phi, d, rng)

    from src.algorithms import _build_finite_differences

    Y = _build_finite_differences(func, Xi, Phi, epsilon=0.1)

    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        X_hat[:, j] = l1_minimize(Phi, Y[:, j])

    U, S, Vt = np.linalg.svd(X_hat.T, full_matrices=False)
    n_show = min(15, len(S))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        np.arange(1, n_show + 1), S[:n_show], color=COLORS[0], alpha=0.8
    )
    # Podświetl k pierwszych
    for i in range(min(k, n_show)):
        bars[i].set_color(COLORS[1])
        bars[i].set_alpha(0.9)

    # Linia przerwy spektralnej
    if k < n_show:
        gap_x = k + 0.5
        ax.axvline(
            gap_x, color="red", linestyle="--", linewidth=1.2, alpha=0.7
        )
        ax.annotate(
            rf"przerwa: $\sigma_{k}={S[k-1]:.3f}$,"
            + rf" $\sigma_{{{k+1}}}={S[k]:.3f}$",
            xy=(gap_x, S[k - 1] * 0.7),
            fontsize=9,
            color="red",
            ha="left",
        )

    ax.set_xlabel("Indeks wartości osobliwej")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title(
        r"Wartości osobliwe $\hat{{X}}^T$ — przerwa spektralna"
        + rf" ($d={d}$, $k={k}$, $m_\Phi={m_Phi}$)"
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Legenda
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor=COLORS[1],
                alpha=0.9,
                label=rf"$k={k}$ pierwszych (aktywne)",
            ),
            Patch(facecolor=COLORS[0], alpha=0.8, label="pozostałe"),
        ]
    )

    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig4_alg2_singular_values.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Wartości osobliwe (top-{n_show}): {np.round(S[:n_show], 4)}")
    print("  → Zapisano fig4_alg2_singular_values.jpg\n")


# ===================================================================
# Rysunek 5 — Wpływ parametru ε na jakość rekonstrukcji
# ===================================================================
def figure5():
    print("=" * 60)
    print("Rysunek 5: Wpływ parametru ε na błąd rekonstrukcji")
    print("=" * 60)

    d = 100
    m_Phi = 50
    m_X = 25
    epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    n_trials = 10
    seed = 42

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    # --- Krzywa 1: bez szumu ---
    mean_noiseless, std_noiseless = [], []
    for eps in epsilon_values:
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, eps, rng=rng_trial)
                err = min(
                    np.linalg.norm(a_hat - a_true),
                    np.linalg.norm(a_hat + a_true),
                )
                errors.append(err)
            except RuntimeError:
                errors.append(2.0)
        mean_noiseless.append(np.mean(errors))
        std_noiseless.append(np.std(errors))
        print(
            f"  [bez szumu] ε={eps:.4f}: {np.mean(errors):.4f}"
            + f" ± {np.std(errors):.4f}"
        )

    # --- Krzywa 2: z szumem sigma_f=0.01 ---
    # noise_sigma = sigma_f (szum pojedynczej obserwacji f(x))
    # Efektywny szum FD: sigma_fd = sqrt(2)*sigma_f/epsilon
    noise_sigma = 0.01
    mean_noisy, std_noisy = [], []
    for eps in epsilon_values:
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial + 77777)
            try:
                a_hat, _ = algorithm1(
                    func,
                    d,
                    m_Phi,
                    m_X,
                    eps,
                    noise_sigma=noise_sigma,
                    rng=rng_trial,
                )
                err = min(
                    np.linalg.norm(a_hat - a_true),
                    np.linalg.norm(a_hat + a_true),
                )
                errors.append(err)
            except RuntimeError:
                errors.append(2.0)
        mean_noisy.append(np.mean(errors))
        std_noisy.append(np.std(errors))
        sigma_fd = np.sqrt(2) * noise_sigma / eps
        print(
            f"[sigma_f={noise_sigma}, sigma_fd={sigma_fd:.4f}] eps={eps:.4f}:"
            + f" {np.mean(errors):.4f} +/- {np.std(errors):.4f}"
        )

    fig, ax = plt.subplots()
    ax.errorbar(
        epsilon_values,
        mean_noiseless,
        yerr=std_noiseless,
        marker="s",
        color=COLORS[0],
        label=r"bez szumu ($\sigma_f=0$, BPDN tol$=0.05\|y\|$)",
    )
    ax.errorbar(
        epsilon_values,
        mean_noisy,
        yerr=std_noisy,
        marker="^",
        color=COLORS[1],
        label=rf"z szumem ($\sigma_f={noise_sigma}$,"
        + r" $\sigma_{{\mathrm{{FD}}}}=\sqrt{{2}}\sigma_f/\epsilon$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\epsilon$ (krok różnicy skończonej)")
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        r"Wpływ parametru $\epsilon$ na jakość rekonstrukcji"
        + rf" ($d={d}$, $m_\Phi={m_Phi}$)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig5_epsilon_sensitivity.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  → Zapisano fig5_epsilon_sensitivity.jpg\n")


# ===================================================================
# Rysunek 6 — Algorytm 1: odporność na szum
# ===================================================================
def figure6():
    print("=" * 60)
    print("Rysunek 6: Algorytm 1 — odporność na szum (noise robustness)")
    print("=" * 60)

    d = 100
    m_Phi = 50
    m_X = 25
    noise_sigmas = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
    n_trials = 8
    seed = 42

    rng_setup = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng_setup,
    )

    eps_values = [0.1, 0.5]
    fig, ax = plt.subplots()

    for eps_idx, eps in enumerate(eps_values):
        mean_errors, std_errors = [], []
        for sigma in noise_sigmas:
            errors = []
            for trial in range(n_trials):
                rng_trial = np.random.default_rng(
                    seed + trial + int(sigma * 1e6) + int(eps * 1000)
                )
                try:
                    a_hat, _ = algorithm1(
                        func,
                        d,
                        m_Phi,
                        m_X,
                        epsilon=eps,
                        noise_sigma=sigma,
                        rng=rng_trial,
                    )
                    err = min(
                        np.linalg.norm(a_hat - a_true),
                        np.linalg.norm(a_hat + a_true),
                    )
                    errors.append(err)
                except RuntimeError:
                    errors.append(2.0)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
            sigma_fd = np.sqrt(2) * sigma / eps if sigma > 0 else 0.0
            print(
                f"  [eps={eps}, sigma_f={sigma:.4f}, sigma_fd={sigma_fd:.4f}]:"
                + f" {np.mean(errors):.4f} +/- {np.std(errors):.4f}"
            )

        x_labels = [str(s) if s > 0 else "0" for s in noise_sigmas]
        x_pos = np.arange(len(noise_sigmas))

        ax.errorbar(
            x_pos,
            mean_errors,
            yerr=std_errors,
            marker=MARKERS[eps_idx],
            color=COLORS[eps_idx],
            label=rf"$\epsilon = {eps}$",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_yscale("log")
    ax.set_xlabel(
        r"$\sigma_f$ (szum pojedynczej obserwacji $f(x)$"
        + r", efektywny FD: $\sqrt{2}\,\sigma_f/\epsilon$)"
    )
    ax.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax.set_title(
        "Algorytm 1: odporność na szum Gaussowski"
        + rf" ($d={d}$, $m_\Phi={m_Phi}$, $\epsilon$=0.1 lub 0.5)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig6_alg1_noise.jpg", format="jpeg", bbox_inches="tight"
    )
    plt.close(fig)
    print("  → Zapisano fig6_alg1_noise.jpg\n")


# ===================================================================
# Rysunek 7 — Skalowanie z wymiarem d (fixed m_Φ/log(d) ratio)
# ===================================================================
def figure7():
    print("=" * 60)
    print("Rysunek 7: Skalowanie błędu z wymiarem d")
    print("=" * 60)

    d_values = [30, 50, 80, 120, 200, 300]
    m_X = 25
    n_trials = 8
    seed = 42

    # Dwie strategie: fixed m_Φ i m_Φ = c·log(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Stałe m_Φ = 40
    m_Phi_fixed = 40
    mean_errors_fixed, std_errors_fixed = [], []
    for d in d_values:
        if m_Phi_fixed >= d:
            mean_errors_fixed.append(np.nan)
            std_errors_fixed.append(np.nan)
            continue
        rng_setup = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_setup,
        )
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial + d * 10000)
            try:
                a_hat, _ = algorithm1(
                    func, d, m_Phi_fixed, m_X, epsilon=0.1, rng=rng_trial
                )
                err = min(
                    np.linalg.norm(a_hat - a_true),
                    np.linalg.norm(a_hat + a_true),
                )
                errors.append(err)
            except RuntimeError:
                errors.append(2.0)
        mean_errors_fixed.append(np.mean(errors))
        std_errors_fixed.append(np.std(errors))
        print(f"  [stałe m_Φ={m_Phi_fixed}] d={d}: {np.mean(errors):.4f}")

    valid = [i for i, v in enumerate(mean_errors_fixed) if not np.isnan(v)]
    ax1.errorbar(
        [d_values[i] for i in valid],
        [mean_errors_fixed[i] for i in valid],
        yerr=[std_errors_fixed[i] for i in valid],
        marker="o",
        color=COLORS[0],
    )
    ax1.set_xlabel(r"$d$ (wymiar przestrzeni)")
    ax1.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax1.set_title(rf"(a) Stałe $m_\Phi = {m_Phi_fixed}$")
    ax1.set_yscale("log")

    # (b) m_Φ = 10·log(d)
    mean_errors_scaled, std_errors_scaled, m_Phi_used = [], [], []
    for d in d_values:
        m_Phi = max(15, int(10 * np.log(d)))
        if m_Phi >= d:
            mean_errors_scaled.append(np.nan)
            std_errors_scaled.append(np.nan)
            m_Phi_used.append(m_Phi)
            continue
        m_Phi_used.append(m_Phi)
        rng_setup = np.random.default_rng(seed)
        func, a_true = make_sparse_ridge_function(
            d,
            active_indices=[0, 1, 2],
            active_values=np.array([1.0, 0.5, 0.3]),
            rng=rng_setup,
        )
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial + d * 10000)
            try:
                a_hat, _ = algorithm1(
                    func, d, m_Phi, m_X, epsilon=0.1, rng=rng_trial
                )
                err = min(
                    np.linalg.norm(a_hat - a_true),
                    np.linalg.norm(a_hat + a_true),
                )
                errors.append(err)
            except RuntimeError:
                errors.append(2.0)
        mean_errors_scaled.append(np.mean(errors))
        std_errors_scaled.append(np.std(errors))
        print(f"  [m_Φ=10ln(d)={m_Phi}] d={d}: {np.mean(errors):.4f}")

    valid = [i for i, v in enumerate(mean_errors_scaled) if not np.isnan(v)]
    ax2.errorbar(
        [d_values[i] for i in valid],
        [mean_errors_scaled[i] for i in valid],
        yerr=[std_errors_scaled[i] for i in valid],
        marker="s",
        color=COLORS[1],
    )
    # Adnotacja z wartościami m_Φ
    for i in valid:
        ax2.annotate(
            rf"$m_\Phi\!=\!{m_Phi_used[i]}$",
            xy=(d_values[i], mean_errors_scaled[i]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color="gray",
        )
    ax2.set_xlabel(r"$d$ (wymiar przestrzeni)")
    ax2.set_ylabel(r"$\min_{\pm}\|a - (\pm\hat{a})\|_2$")
    ax2.set_title(r"(b) $m_\Phi = \lceil 10 \ln d \rceil$")
    ax2.set_yscale("log")

    fig.suptitle(
        r"Skalowanie błędu Algorytmu 1 z wymiarem $d$ ($k=1$, rzadkość $s=3$)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig7_dimension_scaling.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  → Zapisano fig7_dimension_scaling.jpg\n")


# ===================================================================
# Rysunek 8 — Algorytm 2: odporność na szum
# ===================================================================
def figure8():
    print("=" * 60)
    print("Rysunek 8: Algorytm 2 — odporność na szum (noise robustness)")
    print("=" * 60)

    d = 80
    k = 2
    m_Phi = 60
    m_X = 40
    # noise_sigma = sigma_f (szum pojedynczej obserwacji f(x))
    # efektywny szum FD: sigma_fd = sqrt(2)*sigma_f/epsilon
    noise_sigmas = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
    n_trials = 8
    seed = 42

    rng_setup = np.random.default_rng(seed)
    func, A_true = make_k_ridge_function(
        d, k, active_indices=list(range(k * 3)), rng=rng_setup
    )
    P_true = A_true.T @ A_true

    eps_values = [0.1, 0.5]
    fig, ax = plt.subplots()

    for eps_idx, eps in enumerate(eps_values):
        mean_errors, std_errors = [], []
        for sigma in noise_sigmas:
            errors = []
            for trial in range(n_trials):
                rng_trial = np.random.default_rng(
                    seed + trial + int(sigma * 1e6) + int(eps * 1000)
                )
                try:
                    A_hat, _ = algorithm2(
                        func,
                        d,
                        k,
                        m_Phi,
                        m_X,
                        epsilon=eps,
                        noise_sigma=sigma,
                        rng=rng_trial,
                    )
                    P_hat = A_hat.T @ A_hat
                    errors.append(np.linalg.norm(P_true - P_hat, "fro"))
                except Exception:
                    errors.append(2.0)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
            sigma_fd = np.sqrt(2) * sigma / eps if sigma > 0 else 0.0
            print(
                f"  [eps={eps}, sigma_f={sigma:.4f}, sigma_fd={sigma_fd:.4f}]:"
                + f" {np.mean(errors):.4f} +/- {np.std(errors):.4f}"
            )

        x_labels = [str(s) if s > 0 else "0" for s in noise_sigmas]
        x_pos = np.arange(len(noise_sigmas))

        ax.errorbar(
            x_pos,
            mean_errors,
            yerr=std_errors,
            marker=MARKERS[eps_idx],
            color=COLORS[eps_idx],
            label=rf"$\epsilon = {eps}$",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30)
    ax.set_yscale("log")
    ax.set_xlabel(
        r"$\sigma_f$ (szum pojedynczej obserwacji $f(x)$,"
        r" efektywny FD: $\sqrt{2}\,\sigma_f/\epsilon$)"
    )
    ax.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax.set_title(
        "Algorytm 2: odporność na szum Gaussowski"
        + rf" ($d={d}$, $k={k}$, $m_\Phi={m_Phi}$, $\epsilon$=0.1 lub 0.5)"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig8_alg2_noise.jpg", format="jpeg", bbox_inches="tight"
    )
    plt.close(fig)
    print("  → Zapisano fig8_alg2_noise.jpg\n")


# ===================================================================
# Rysunek 9 — Algorytm 2: skalowanie błędu podprzestrzeni z wymiarem d
# ===================================================================
def figure9():
    print("=" * 60)
    print("Rysunek 9: Algorytm 2 — skalowanie błędu podprzestrzeni vs d")
    print("=" * 60)

    k = 2
    d_values = [30, 50, 80, 120, 200, 300]
    m_X = 40
    n_trials = 8
    seed = 42

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Stałe m_Φ = 60
    m_Phi_fixed = 60
    mean_errors_fixed, std_errors_fixed = [], []
    for d in d_values:
        if m_Phi_fixed >= d:
            mean_errors_fixed.append(np.nan)
            std_errors_fixed.append(np.nan)
            continue
        rng_setup = np.random.default_rng(seed)
        func, A_true = make_k_ridge_function(
            d, k, active_indices=list(range(k * 3)), rng=rng_setup
        )
        P_true = A_true.T @ A_true
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial + d * 10000)
            try:
                A_hat, _ = algorithm2(
                    func, d, k, m_Phi_fixed, m_X, epsilon=0.1, rng=rng_trial
                )
                P_hat = A_hat.T @ A_hat
                errors.append(np.linalg.norm(P_true - P_hat, "fro"))
            except Exception:
                errors.append(2.0)
        mean_errors_fixed.append(np.mean(errors))
        std_errors_fixed.append(np.std(errors))
        print(f"  [stałe m_Φ={m_Phi_fixed}] d={d}: {np.mean(errors):.4f}")

    valid = [i for i, v in enumerate(mean_errors_fixed) if not np.isnan(v)]
    ax1.errorbar(
        [d_values[i] for i in valid],
        [mean_errors_fixed[i] for i in valid],
        yerr=[std_errors_fixed[i] for i in valid],
        marker="o",
        color=COLORS[0],
    )
    ax1.set_xlabel(r"$d$ (wymiar przestrzeni)")
    ax1.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax1.set_title(rf"(a) Stałe $m_\Phi = {m_Phi_fixed}$")
    ax1.set_yscale("log")

    # (b) m_Φ = 15·log(d)
    mean_errors_scaled, std_errors_scaled, m_Phi_used = [], [], []
    for d in d_values:
        m_Phi = max(20, int(15 * np.log(d)))
        if m_Phi >= d:
            mean_errors_scaled.append(np.nan)
            std_errors_scaled.append(np.nan)
            m_Phi_used.append(m_Phi)
            continue
        m_Phi_used.append(m_Phi)
        rng_setup = np.random.default_rng(seed)
        func, A_true = make_k_ridge_function(
            d, k, active_indices=list(range(k * 3)), rng=rng_setup
        )
        P_true = A_true.T @ A_true
        errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(seed + trial + d * 10000)
            try:
                A_hat, _ = algorithm2(
                    func, d, k, m_Phi, m_X, epsilon=0.1, rng=rng_trial
                )
                P_hat = A_hat.T @ A_hat
                errors.append(np.linalg.norm(P_true - P_hat, "fro"))
            except Exception:
                errors.append(2.0)
        mean_errors_scaled.append(np.mean(errors))
        std_errors_scaled.append(np.std(errors))
        print(f"  [m_Φ=15ln(d)={m_Phi}] d={d}: {np.mean(errors):.4f}")

    valid = [i for i, v in enumerate(mean_errors_scaled) if not np.isnan(v)]
    ax2.errorbar(
        [d_values[i] for i in valid],
        [mean_errors_scaled[i] for i in valid],
        yerr=[std_errors_scaled[i] for i in valid],
        marker="s",
        color=COLORS[1],
    )
    for i in valid:
        ax2.annotate(
            rf"$m_\Phi\!=\!{m_Phi_used[i]}$",
            xy=(d_values[i], mean_errors_scaled[i]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color="gray",
        )
    ax2.set_xlabel(r"$d$ (wymiar przestrzeni)")
    ax2.set_ylabel(r"$\|A^T A - \hat{A}^T \hat{A}\|_F$")
    ax2.set_title(r"(b) $m_\Phi = \lceil 15 \ln d \rceil$")
    ax2.set_yscale("log")

    fig.suptitle(
        rf"Skalowanie błędu Algorytmu 2 z wymiarem $d$ ($k={k}$, $m_X={m_X}$)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        f"{FIGDIR}/fig9_alg2_dimension_scaling.jpg",
        format="jpeg",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  → Zapisano fig9_alg2_dimension_scaling.jpg\n")


# ===================================================================
# main
# ===================================================================
FIGURES = {
    1: figure1,
    2: figure2,
    3: figure3,
    4: figure4,
    5: figure5,
    6: figure6,
    7: figure7,
    8: figure8,
    9: figure9,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generowanie wykresów do pracy magisterskiej"
    )
    parser.add_argument(
        "--fig",
        nargs="*",
        type=int,
        default=None,
        help="Numery figur do wygenerowania (domyślnie: wszystkie)",
    )
    args = parser.parse_args()
    _ensure_dir()

    figs_to_run = args.fig if args.fig else sorted(FIGURES.keys())
    t_start = time.time()

    for fig_num in figs_to_run:
        if fig_num not in FIGURES:
            print(
                f"UWAGA: Rysunek {fig_num} nie istnieje"
                + f"(dostępne: {sorted(FIGURES.keys())})"
            )
            continue
        t0 = time.time()
        FIGURES[fig_num]()
        print(f"  Czas: {time.time() - t0:.1f}s\n")

    print(f"\nGotowe! Wszystkie wykresy w katalogu {FIGDIR}/")
    print(f"Łączny czas: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
