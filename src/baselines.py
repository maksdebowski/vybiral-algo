"""
Implementacja algorytmów bazowych (baselines) do porównania
z metodą Fornasiera-Schnassa-Vybirala (compressed sensing).

Zaimplementowane metody:
1. Active Subspace Method (ASM) — Constantine (2015)
2. OMP-based Recovery — Compressed sensing z Orthogonal Matching Pursuit
3. Sliced Inverse Regression (SIR) — Li (1991)
"""

import numpy as np
from numpy.typing import NDArray

from .sampling import sample_sphere, bernoulli_matrix


# ===================================================================
# 1. Active Subspace Method (ASM) — Constantine, 2015
# ===================================================================


def _estimate_gradient_fd(
    f: callable, x: NDArray, d: int, epsilon: float
) -> NDArray:
    """
    Estymacja gradientu ∇f(x) za pomocą centralnych różnic skończonych.
    Wymaga 2d ewaluacji funkcji.
    """
    grad = np.zeros(d)
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        grad[i] = (f(x + epsilon * e_i) - f(x - epsilon * e_i)) / (
            2.0 * epsilon
        )
    return grad


def active_subspace_method(
    f: callable,
    d: int,
    k: int,
    m_samples: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> NDArray:
    """
    Active Subspace Method (ASM) — Constantine, 2015.

    Estymuje aktywną podprzestrzeń za pomocą macierzy
    C = E[∇f(x) ∇f(x)^T] ≈ (1/m) Σ ∇f(ξ_j) ∇f(ξ_j)^T.

    Dekompozycja spektralna C daje k dominujących wektorów własnych
    jako estymatory aktywnych kierunków.

    Koszt: 2d x m_samples ewaluacji funkcji (centralne różnice skończone
    do estymacji pełnego gradientu w R^d).

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki f: R^d → R.
    d : int
        Wymiar przestrzeni.
    k : int
        Wymiar aktywnej podprzestrzeni.
    m_samples : int
        Liczba punktów próbkowania na sferze.
    epsilon : float
        Krok różnicy skończonej.
    noise_sigma : float
        Odchylenie standardowe szumu Gaussa dodawanego
        do ewaluacji funkcji.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    NDArray o kształcie (k, d)
        Estymata macierzy aktywnych kierunków Â.
    """
    if rng is None:
        rng = np.random.default_rng()

    Xi = sample_sphere(d, m_samples, rng)

    # Estymacja gradientów
    G = np.zeros((d, m_samples))
    for j in range(m_samples):
        grad = _estimate_gradient_fd(f, Xi[j], d, epsilon)
        if noise_sigma > 0:
            grad += rng.normal(0, noise_sigma / epsilon, size=d)
        G[:, j] = grad

    # Macierz C = (1/m) G G^T
    C = G @ G.T / m_samples

    # Dekompozycja spektralna
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # eigh zwraca wartości rosnąco — bierzemy k ostatnich
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if verbose:
        print(
            f"  ASM: wartości własne (top-{min(k+2, len(eigenvalues))}): "
            f"{eigenvalues[:min(k+2, len(eigenvalues))]}"
        )
        print(f"  ASM: ewaluacji funkcji = {2 * d * m_samples}")

    A_hat = eigenvectors[:, :k].T  # kształt (k, d)
    return A_hat


# ===================================================================
# 2. OMP-based Recovery
# ===================================================================


def _omp_solve(Phi: NDArray, y: NDArray, sparsity: int) -> NDArray:
    """
    Orthogonal Matching Pursuit (OMP) — Tropp & Gilbert, 2007.

    Rozwiązuje problem sparse recovery:
        x̂ = argmin ||x||_0  subject to  Φx ≈ y
    używając zachłannego algorytmu OMP z zadaną rzadkością.

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa.
    y : NDArray o kształcie (m,)
        Wektor obserwacji.
    sparsity : int
        Maksymalna rzadkość (liczba niezerowych współrzędnych).

    Zwraca
    -------
    NDArray o kształcie (d,)
        Rozwiązanie rzadkie.
    """
    m, d = Phi.shape
    residual = y.copy()
    support = []

    for _ in range(sparsity):
        # Wybierz kolumnę Φ najbardziej skorelowaną z residuum
        correlations = np.abs(Phi.T @ residual)
        # Wyklucz już wybrane
        correlations[support] = -1
        j_best = np.argmax(correlations)
        support.append(j_best)

        # Rozwiąż LS na wybranym support
        Phi_S = Phi[:, support]
        x_S = np.linalg.lstsq(Phi_S, y, rcond=None)[0]

        # Aktualizuj residuum
        residual = y - Phi_S @ x_S

    # Zapisz rozwiązanie
    x_out = np.zeros(d)
    x_out[support] = x_S
    return x_out


def omp_algorithm1(
    f: callable,
    d: int,
    m_Phi: int,
    m_X: int,
    sparsity: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> NDArray:
    """
    Algorytm odzyskiwania wektora kierunkowego a z użyciem OMP
    zamiast minimalizacji l1.

    Struktura identyczna jak Algorytm 1 Vybirala, ale krok (3) zastąpiony
    algorytmem OMP (Tropp & Gilbert, 2007).

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki f: R^d → R.
    d : int
        Wymiar przestrzeni.
    m_Phi : int
        Liczba kierunków pomiarowych.
    m_X : int
        Liczba punktów próbkowania.
    sparsity : int
        Zakładana rzadkość wektora a.
    epsilon : float
        Krok różnicy skończonej.
    noise_sigma : float
        Odchylenie standardowe szumu.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    NDArray o kształcie (d,)
        Estymata wektora kierunkowego â.
    """
    if rng is None:
        rng = np.random.default_rng()

    Xi = sample_sphere(d, m_X, rng)
    Phi = bernoulli_matrix(m_Phi, d, rng)

    from .algorithms import _build_finite_differences

    Y = _build_finite_differences(f, Xi, Phi, epsilon)

    if noise_sigma > 0:
        W = rng.normal(0, noise_sigma, size=Y.shape)
        Y = Y + W / epsilon

    # OMP dla każdej kolumny
    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        X_hat[:, j] = _omp_solve(Phi, Y[:, j], sparsity)
        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  OMP: kolumna {j+1}/{m_X} zakończona")

    # Wybór optymalnego kierunku i normalizacja
    norms = np.linalg.norm(X_hat, axis=0)
    j0 = np.argmax(norms)

    a_hat = X_hat[:, j0]
    if np.linalg.norm(a_hat) > 1e-12:
        a_hat = a_hat / np.linalg.norm(a_hat)
    else:
        a_hat = np.zeros(d)

    if verbose:
        print(f"  OMP: j0={j0}, ||x̂_j0||_2 = {norms[j0]:.6f}")

    return a_hat


def omp_algorithm2(
    f: callable,
    d: int,
    k: int,
    m_Phi: int,
    m_X: int,
    sparsity: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> NDArray:
    """
    Algorytm odzyskiwania podprzestrzeni A z użyciem OMP + SVD.
    Struktura jak Algorytm 2 Vybirala, z OMP zamiast l1.

    Zwraca
    -------
    NDArray o kształcie (k, d)
        Estymata macierzy parametrów Â.
    """
    if rng is None:
        rng = np.random.default_rng()

    Xi = sample_sphere(d, m_X, rng)
    Phi = bernoulli_matrix(m_Phi, d, rng)

    from .algorithms import _build_finite_differences

    Y = _build_finite_differences(f, Xi, Phi, epsilon)

    if noise_sigma > 0:
        W = rng.normal(0, noise_sigma, size=Y.shape)
        Y = Y + W / epsilon

    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        X_hat[:, j] = _omp_solve(Phi, Y[:, j], sparsity)
        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  OMP: kolumna {j+1}/{m_X} zakończona")

    # SVD
    U, S, Vt = np.linalg.svd(X_hat.T, full_matrices=False)

    if verbose:
        print(
            f"  OMP+SVD: wartości osobliwe (top-{min(k+2, len(S))}): "
            f"{S[:min(k+2, len(S))]}"
        )

    A_hat = Vt[:k, :]
    return A_hat


# ===================================================================
# 3. Sliced Inverse Regression (SIR) — Li, 1991
# ===================================================================


def sliced_inverse_regression(
    f: callable,
    d: int,
    k: int,
    m_samples: int,
    n_slices: int = 10,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> NDArray:
    """
    Sliced Inverse Regression (SIR) — Li, 1991.

    Metoda statystyczna estymacji wystarczających kierunków redukcji
    (sufficient dimension reduction directions) na podstawie
    odwrotnej regresji E[X | Y].

    Kroki:
    1. Wylosuj m punktów X z rozkładu na sferze.
    2. Oblicz Y = f(X) dla każdego punktu.
    3. Podziel zakres Y na H plasterków (slices).
    4. Oblicz średnią ̄x_h w każdym plasterku.
    5. Ważona kowariancja Σ_SIR = Σ_h p_h (x̄_h - x̄)(x̄_h - x̄)^T.
    6. Dekompozycja spektralna → k dominujących wektorów własnych.

    Koszt: m_samples ewaluacji funkcji (nie wymaga pochodnych).

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki f: R^d → R.
    d : int
        Wymiar przestrzeni.
    k : int
        Wymiar aktywnej podprzestrzeni.
    m_samples : int
        Liczba punktów próbkowania.
    n_slices : int
        Liczba plasterków (domyślnie 10).
    noise_sigma : float
        Odchylenie standardowe szumu dodawanego do ewaluacji.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    NDArray o kształcie (k, d)
        Estymata macierzy aktywnych kierunków Â.
    """
    if rng is None:
        rng = np.random.default_rng()

    # (1) Próbkowanie
    X = sample_sphere(d, m_samples, rng)

    # (2) Ewaluacja funkcji
    Y = np.array([f(X[i]) for i in range(m_samples)])
    if noise_sigma > 0:
        Y += rng.normal(0, noise_sigma, size=m_samples)

    # (3) Plasterkowanie (slicing)
    sorted_idx = np.argsort(Y)
    slice_size = max(1, m_samples // n_slices)

    # (4-5) Ważona kowariancja plasterków
    Sigma_SIR = np.zeros((d, d))
    x_bar = np.mean(X, axis=0)

    for h in range(n_slices):
        start = h * slice_size
        end = min((h + 1) * slice_size, m_samples)
        if start >= m_samples:
            break
        X_slice = X[sorted_idx[start:end]]
        n_h = len(X_slice)
        p_h = n_h / m_samples  # waga plasterka

        x_h_mean = np.mean(X_slice, axis=0)
        diff = (x_h_mean - x_bar).reshape(-1, 1)
        Sigma_SIR += p_h * (diff @ diff.T)

    # (6) Dekompozycja spektralna
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_SIR)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if verbose:
        print(
            f"  SIR: wartości własne (top-{min(k+2, len(eigenvalues))}): "
            f"{eigenvalues[:min(k+2, len(eigenvalues))]}"
        )
        print(f"  SIR: ewaluacji funkcji = {m_samples}")

    A_hat = eigenvectors[:, :k].T  # kształt (k, d)
    return A_hat
