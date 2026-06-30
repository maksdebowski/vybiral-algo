"""
Punkt wejścia do eksperymentów numerycznych dla pracy magisterskiej:
"Uczenie funkcji wysoko-wymiarowych o niewielkiej liczbie parametrów liniowych"

Uruchamia eksperymenty z Rozdziału 3:
  3.3 — Weryfikacja dokładności rekonstrukcji (bezszumowy)
  3.4 — Doświadczenia numeryczne w obecności szumu

Użycie:
    python main.py --experiment all
    python main.py --experiment alg1
    python main.py --experiment alg2
    python main.py --experiment epsilon
    python main.py --experiment vybiral
    python main.py --experiment quick   # szybki test poprawności
"""

import argparse
import time
import numpy as np

from src.experiments import (
    experiment_algorithm1_noiseless,
    experiment_algorithm2_noiseless,
    experiment_vybiral_noise,
    experiment_epsilon_sensitivity,
    plot_reconstruction_error,
    plot_epsilon_sensitivity,
    plot_success_maps,
)
from src.algorithms import algorithm1, algorithm2
from src.test_functions import (
    make_sparse_ridge_function,
    make_k_ridge_function,
)


def run_quick_test():
    """Szybki test poprawności algorytmów na małym wymiarze."""
    print("=" * 60)
    print("SZYBKI TEST POPRAWNOŚCI")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Test Algorytmu 1 (k=1) ---
    print("\n--- Algorytm 1 (k=1), d=100 ---")
    d = 100
    func, a_true = make_sparse_ridge_function(
        d,
        active_indices=[0, 1, 2],
        active_values=np.array([1.0, 0.5, 0.3]),
        rng=np.random.default_rng(0),
    )
    print(f"  Prawdziwe a (niezerowe): {a_true[a_true != 0]}")
    print(f"  Aktywne indeksy: {np.nonzero(a_true)[0]}")

    t0 = time.time()
    a_hat, f_hat = algorithm1(
        func, d, m_Phi=40, m_X=20, epsilon=0.1, rng=rng, verbose=True
    )
    dt = time.time() - t0

    err1 = np.linalg.norm(a_hat - a_true)
    err2 = np.linalg.norm(a_hat + a_true)
    error = min(err1, err2)
    print(
        f"  Odzyskane â (top-5): indeksy={np.argsort(-np.abs(a_hat))[:5]}, "
        f"wartości={a_hat[np.argsort(-np.abs(a_hat))[:5]]}"
    )
    print(f"  Błąd ||a - ±â||_2 = {error:.6f}")
    print(f"  Czas: {dt:.2f}s")

    # --- Test Algorytmu 2 (k=2) ---
    print("\n--- Algorytm 2 (k=2), d=100 ---")
    func2, A_true = make_k_ridge_function(
        d, k=2, active_indices=[0, 1, 2, 3], rng=np.random.default_rng(0)
    )
    P_true = A_true.T @ A_true
    print(f"  Rząd A: {np.linalg.matrix_rank(A_true)}")

    t0 = time.time()
    A_hat, f_hat2 = algorithm2(
        func2, d, k=2, m_Phi=40, m_X=30, epsilon=0.1, rng=rng, verbose=True
    )
    dt = time.time() - t0

    P_hat = A_hat.T @ A_hat
    error_sub = np.linalg.norm(P_true - P_hat, "fro")
    print(f"  Błąd podprzestrzeni ||A^T A - Â^T Â||_F = {error_sub:.6f}")
    print(f"  Czas: {dt:.2f}s")

    print("\n✓ Szybki test zakończony pomyślnie.")


def run_experiment_alg1():
    """Eksperyment 3.3a: Algorytm 1 bez szumu."""
    print("=" * 60)
    print("EKSPERYMENT 3.3a: Algorytm 1 (k=1), przypadek bezszumowy")
    print("=" * 60)
    results = experiment_algorithm1_noiseless(
        d=100,
        m_Phi_values=[10, 20, 30, 50, 70, 100],
        m_X=30,
        epsilon=0.1,
        n_trials=10,
        seed=42,
    )
    plot_reconstruction_error(
        results,
        title=r"Algorytm 1: Błąd $\|a - \hat{a}\|_2$ vs."
        + r" $m_\Phi$ ($d=100$, $k=1$)",
        save_path="figures/alg1_noiseless.png",
    )
    return results


def run_experiment_alg2():
    """Eksperyment 3.3b: Algorytm 2 bez szumu."""
    print("=" * 60)
    print("EKSPERYMENT 3.3b: Algorytm 2 (k=2), przypadek bezszumowy")
    print("=" * 60)
    results = experiment_algorithm2_noiseless(
        d=50,
        k=2,
        m_Phi_values=[20, 40, 60, 80, 100],
        m_X=40,
        epsilon=0.1,
        n_trials=5,
        seed=42,
    )
    plot_reconstruction_error(
        results,
        title=r"Algorytm 2: $\|A^T A - \hat{A}^T \hat{A}\|_F$ vs."
        + r" $m_\Phi$ ($d=50$, $k=2$)",
        save_path="figures/alg2_noiseless.png",
    )
    return results


def run_experiment_epsilon():
    """Eksperyment 3.3c: Wpływ parametru ε."""
    print("=" * 60)
    print("EKSPERYMENT: Wpływ parametru ε na rekonstrukcję")
    print("=" * 60)
    results = experiment_epsilon_sensitivity(
        d=100,
        m_Phi=60,
        m_X=30,
        epsilon_values=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        n_trials=10,
        seed=42,
    )
    plot_epsilon_sensitivity(
        results, save_path="figures/epsilon_sensitivity.png"
    )
    return results


def run_experiment_vybiral():
    """
    Eksperyment 3.4: Reprodukcja Figure 2 z pracy Vybirala.
    Mapa sukcesu rekonstrukcji, d=1000, szum Gaussa.

    UWAGA: Ten eksperyment jest bardzo kosztowny obliczeniowo.
    Dla pełnej reprodukcji (d=1000, 100 prób) potrzeba wielu godzin.
    Poniżej ustawiona jest wersja ze zmniejszonymi parametrami.
    """
    print("=" * 60)
    print("EKSPERYMENT 3.4: Mapa sukcesu (Vybiral Figure 2)")
    print("=" * 60)
    print("UWAGA: Wersja uproszczona (mniejsze parametry).")
    print("Dla pełnej reprodukcji zmień parametry w kodzie.\n")

    results = experiment_vybiral_noise(
        d=1000,
        m_X_values=[6, 12, 18, 24, 30],
        m_Phi_values=[20, 40, 60, 80, 100],
        noise_levels=[0.1, 0.01, 0.001],
        epsilon=0.1,
        n_trials=5,
        threshold=0.1,
        seed=42,
    )
    plot_success_maps(results, save_path="figures/vybiral_success_maps.png")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Eksperymenty numeryczne — aproksymacja funkcji"
        + " grzbietowych"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="quick",
        choices=["quick", "alg1", "alg2", "epsilon", "vybiral", "all"],
        help="Który eksperyment uruchomić (domyślnie: quick)",
    )
    args = parser.parse_args()

    import os

    os.makedirs("figures", exist_ok=True)

    if args.experiment == "quick":
        run_quick_test()
    elif args.experiment == "alg1":
        run_experiment_alg1()
    elif args.experiment == "alg2":
        run_experiment_alg2()
    elif args.experiment == "epsilon":
        run_experiment_epsilon()
    elif args.experiment == "vybiral":
        run_experiment_vybiral()
    elif args.experiment == "all":
        run_quick_test()
        print("\n")
        run_experiment_alg1()
        print("\n")
        run_experiment_alg2()
        print("\n")
        run_experiment_epsilon()
        print("\n")
        run_experiment_vybiral()


if __name__ == "__main__":
    main()
