"""
Moduł eksperymentów numerycznych odtwarzających wyniki z pracy
Fornasiera, Schnassa i Vybirala oraz rozdziału 3 pracy magisterskiej.

Eksperymenty:
1. Weryfikacja dokładności rekonstrukcji (przypadek bezszumowy)
   — Algorytm 1 i Algorytm 2 na funkcjach testowych.
2. Doświadczenia numeryczne w obecności szumu
   — Mapy sukcesu rekonstrukcji aktywnych współrzędnych (Figure 2).
3. Analiza wpływu hiperparametrów (m_Φ, m_X, ε).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .algorithms import algorithm1, algorithm2, identify_active_coordinates
from .test_functions import (
    vybiral_test_function,
    make_sparse_ridge_function,
    make_k_ridge_function,
)


def experiment_algorithm1_noiseless(d: int = 100,
                                     m_Phi_values: list[int] | None = None,
                                     m_X: int = 30,
                                     epsilon: float = 0.1,
                                     n_trials: int = 10,
                                     seed: int = 42) -> dict:
    """
    Eksperyment 1: Weryfikacja Algorytmu 1 (k=1) bez szumu.

    Bada błąd rekonstrukcji ||a - sign·â||_2 w zależności od m_Φ.
    Funkcja testowa: f(x) = cos(π·a·x) z rzadkim wektorem a
    (3 aktywne współrzędne z d).

    Zwraca
    -------
    dict z kluczami: 'm_Phi_values', 'mean_errors', 'std_errors'
    """
    if m_Phi_values is None:
        m_Phi_values = [10, 20, 30, 50, 70, 100]

    rng = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d, active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng
    )

    results = {'m_Phi_values': m_Phi_values, 'mean_errors': [], 'std_errors': []}

    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, epsilon, rng=trial_rng)
                # Błąd z uwzględnieniem nieznaku (sign ambiguity)
                err1 = np.linalg.norm(a_hat - a_true)
                err2 = np.linalg.norm(a_hat + a_true)
                errors.append(min(err1, err2))
            except RuntimeError:
                errors.append(2.0)  # max error

        results['mean_errors'].append(np.mean(errors))
        results['std_errors'].append(np.std(errors))
        print(f"  m_Φ = {m_Phi}: błąd = {np.mean(errors):.6f} ± {np.std(errors):.6f}")

    return results


def experiment_algorithm2_noiseless(d: int = 50,
                                     k: int = 2,
                                     m_Phi_values: list[int] | None = None,
                                     m_X: int = 40,
                                     epsilon: float = 0.1,
                                     n_trials: int = 5,
                                     seed: int = 42) -> dict:
    """
    Eksperyment 2: Weryfikacja Algorytmu 2 (k≥1) bez szumu.

    Bada błąd rekonstrukcji podprzestrzeni ||A^T A - Â^T Â||_F
    w zależności od m_Φ.

    Zwraca
    -------
    dict z kluczami: 'm_Phi_values', 'mean_errors', 'std_errors'
    """
    if m_Phi_values is None:
        m_Phi_values = [20, 40, 60, 80, 100]

    rng = np.random.default_rng(seed)
    func, A_true = make_k_ridge_function(
        d, k, active_indices=list(range(k * 2)), rng=rng
    )
    P_true = A_true.T @ A_true  # rzut ortogonalny

    results = {'m_Phi_values': m_Phi_values, 'mean_errors': [], 'std_errors': []}

    for m_Phi in m_Phi_values:
        errors = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(seed + trial + m_Phi * 1000)
            try:
                A_hat, _ = algorithm2(func, d, k, m_Phi, m_X, epsilon, rng=trial_rng)
                P_hat = A_hat.T @ A_hat
                errors.append(np.linalg.norm(P_true - P_hat, 'fro'))
            except RuntimeError:
                errors.append(2.0)

        results['mean_errors'].append(np.mean(errors))
        results['std_errors'].append(np.std(errors))
        print(f"  m_Φ = {m_Phi}: ||P - P̂||_F = {np.mean(errors):.6f} ± {np.std(errors):.6f}")

    return results


def experiment_vybiral_noise(d: int = 1000,
                              m_X_values: list[int] | None = None,
                              m_Phi_values: list[int] | None = None,
                              noise_levels: list[float] | None = None,
                              epsilon: float = 0.1,
                              n_trials: int = 20,
                              threshold: float = 0.1,
                              seed: int = 42) -> dict:
    """
    Eksperyment 3: Reprodukcja Figure 2 z pracy Vybirala.

    Funkcja testowa (63): f(x) = max(1 - 5√((x₃-½)² + (x₄-½)²), 0)³
    w wymiarze d=1000.

    Identyfikacja aktywnych współrzędnych {3, 4} z szumem Gaussa.

    Parametry
    ----------
    d : int
        Wymiar (domyślnie 1000 jak w pracy).
    m_X_values : list[int]
        Wartości m_X do testowania (oś x na mapie).
    m_Phi_values : list[int]
        Wartości m_Φ do testowania (oś y na mapie).
    noise_levels : list[float]
        Poziomy szumu ν ∈ {0.1, 0.01, 0.001}.
    n_trials : int
        Liczba prób dla każdej kombinacji parametrów.
    threshold : float
        Próg identyfikacji aktywnych współrzędnych.
    seed : int
        Ziarno losowości.

    Zwraca
    -------
    dict
        Słownik z macierzami wskaźników sukcesu dla każdego poziomu szumu.
    """
    if m_X_values is None:
        m_X_values = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    if m_Phi_values is None:
        m_Phi_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    if noise_levels is None:
        noise_levels = [0.1, 0.01, 0.001]

    true_active = {2, 3}  # indeksowanie od 0 (współrzędne x_3, x_4)

    results = {}
    for nu in noise_levels:
        print(f"\n--- Szum ν = {nu} ---")
        success_map = np.zeros((len(m_Phi_values), len(m_X_values)))

        for i, m_Phi in enumerate(m_Phi_values):
            for j_idx, m_X in enumerate(m_X_values):
                n_success = 0
                for trial in range(n_trials):
                    trial_rng = np.random.default_rng(
                        seed + trial + m_Phi * 10000 + m_X * 100
                    )
                    try:
                        active = identify_active_coordinates(
                            vybiral_test_function, d, k=2,
                            m_Phi=m_Phi, m_X=m_X,
                            epsilon=epsilon,
                            noise_sigma=nu,
                            threshold=threshold,
                            rng=trial_rng
                        )
                        if set(active) == true_active:
                            n_success += 1
                    except Exception:
                        pass

                success_map[i, j_idx] = n_success / n_trials * 100
                print(f"  m_Φ={m_Phi:3d}, m_X={m_X:2d}: "
                      f"sukces = {success_map[i, j_idx]:.0f}%")

        results[nu] = success_map

    results['m_X_values'] = m_X_values
    results['m_Phi_values'] = m_Phi_values
    return results


def experiment_epsilon_sensitivity(d: int = 100,
                                    m_Phi: int = 60,
                                    m_X: int = 30,
                                    epsilon_values: list[float] | None = None,
                                    n_trials: int = 10,
                                    seed: int = 42) -> dict:
    """
    Eksperyment 4: Wpływ parametru ε (krok różnicy skończonej)
    na jakość rekonstrukcji.

    Zgodnie z Uwagą 4(iii) z pracy Vybirala — ε nie powinno być
    zbyt małe ze względu na stabilność numeryczną.

    Zwraca
    -------
    dict z kluczami: 'epsilon_values', 'mean_errors', 'std_errors'
    """
    if epsilon_values is None:
        epsilon_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    rng = np.random.default_rng(seed)
    func, a_true = make_sparse_ridge_function(
        d, active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=rng
    )

    results = {'epsilon_values': epsilon_values, 'mean_errors': [], 'std_errors': []}

    for eps in epsilon_values:
        errors = []
        for trial in range(n_trials):
            trial_rng = np.random.default_rng(seed + trial)
            try:
                a_hat, _ = algorithm1(func, d, m_Phi, m_X, eps, rng=trial_rng)
                err1 = np.linalg.norm(a_hat - a_true)
                err2 = np.linalg.norm(a_hat + a_true)
                errors.append(min(err1, err2))
            except RuntimeError:
                errors.append(2.0)

        results['mean_errors'].append(np.mean(errors))
        results['std_errors'].append(np.std(errors))
        print(f"  ε = {eps:.4f}: błąd = {np.mean(errors):.6f} ± {np.std(errors):.6f}")

    return results


# ---------------------------------------------------------------------------
# Funkcje wizualizacji
# ---------------------------------------------------------------------------

def plot_reconstruction_error(results: dict, title: str = "",
                               save_path: str | None = None):
    """Wykres błędu rekonstrukcji vs m_Φ."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results['m_Phi_values'],
        results['mean_errors'],
        yerr=results['std_errors'],
        marker='o', capsize=4, linewidth=2
    )
    ax.set_xlabel(r'$m_\Phi$ (liczba kierunków pomiarowych)', fontsize=12)
    ax.set_ylabel(r'Błąd rekonstrukcji', fontsize=12)
    ax.set_title(title or 'Błąd rekonstrukcji vs. $m_\\Phi$', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_epsilon_sensitivity(results: dict, save_path: str | None = None):
    """Wykres wpływu ε na błąd rekonstrukcji."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        results['epsilon_values'],
        results['mean_errors'],
        yerr=results['std_errors'],
        marker='s', capsize=4, linewidth=2, color='darkorange'
    )
    ax.set_xlabel(r'$\epsilon$ (krok różnicy skończonej)', fontsize=12)
    ax.set_ylabel(r'Błąd rekonstrukcji', fontsize=12)
    ax.set_title(r'Wpływ parametru $\epsilon$ na jakość rekonstrukcji', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_success_maps(results: dict, save_path: str | None = None):
    """
    Mapy sukcesu rekonstrukcji aktywnych współrzędnych
    (reprodukcja Figure 2 z pracy Vybirala).
    """
    noise_levels = [k for k in results.keys()
                    if isinstance(k, (float, int)) and k > 0]
    noise_levels.sort(reverse=True)

    n_plots = len(noise_levels)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    m_X_vals = results['m_X_values']
    m_Phi_vals = results['m_Phi_values']

    for idx, nu in enumerate(noise_levels):
        ax = axes[idx]
        im = ax.imshow(
            results[nu],
            aspect='auto',
            origin='lower',
            cmap='gray_r',
            vmin=0, vmax=100,
            extent=[
                m_X_vals[0], m_X_vals[-1],
                m_Phi_vals[0], m_Phi_vals[-1]
            ]
        )
        ax.set_xlabel(r'$m_X$ (punkty próbkowania)', fontsize=11)
        ax.set_ylabel(r'$m_\Phi$ (kierunki)', fontsize=11)
        ax.set_title(rf'$\nu = {nu}$', fontsize=13)

    fig.colorbar(im, ax=axes, label='Wskaźnik sukcesu [%]',
                 shrink=0.8, pad=0.02)
    fig.suptitle('Mapa sukcesu rekonstrukcji aktywnych współrzędnych\n'
                 r'$f(x) = \max(1 - 5\sqrt{(x_3-\frac{1}{2})^2 + (x_4-\frac{1}{2})^2}, 0)^3$, '
                 r'$d=1000$',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
