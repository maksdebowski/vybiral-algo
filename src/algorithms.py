"""
Implementacja Algorytmu 1 (k=1) i Algorytmu 2 (k≥1)
z pracy Fornasiera, Schnassa i Vybirala:
"Learning Functions of Few Arbitrary Linear Parameters in High Dimensions"

Algorytm 1 — odzyskiwanie pojedynczego wektora kierunkowego a
              dla funkcji grzbietowej f(x) = g(a · x).

Algorytm 2 — odzyskiwanie macierzy parametrów A (k × d)
              dla k-grzbietowej f(x) = g(Ax) za pomocą SVD.
"""

import numpy as np
from numpy.typing import NDArray

from .sampling import sample_sphere, bernoulli_matrix
from .l1_solver import l1_minimize, l1_minimize_noisy


def _build_finite_differences(f: callable,
                               Xi: NDArray,
                               Phi_rows: NDArray,
                               epsilon: float) -> NDArray:
    """
    Konstruuje macierz Y ilorazów różnicowych (finite differences).

    y_{ij} = [f(ξ_j + ε·φ_i) - f(ξ_j)] / ε

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki f: R^d → R.
    Xi : NDArray o kształcie (m_X, d)
        Punkty próbkowania na sferze.
    Phi_rows : NDArray o kształcie (m_Phi, d)
        Kierunki pomiarowe (wiersze macierzy Φ, ale NIEPRZESKALOWANE,
        tj. φ_i z norma ≈ √(d/m_Phi)).
    epsilon : float
        Krok różnicy skończonej ε > 0.

    Zwraca
    -------
    NDArray o kształcie (m_Phi, m_X)
        Macierz Y.
    """
    m_Phi = Phi_rows.shape[0]
    m_X = Xi.shape[0]
    Y = np.zeros((m_Phi, m_X))

    for j in range(m_X):
        f_xi = f(Xi[j])
        for i in range(m_Phi):
            f_shifted = f(Xi[j] + epsilon * Phi_rows[i])
            Y[i, j] = (f_shifted - f_xi) / epsilon

    return Y


def algorithm1(f: callable,
               d: int,
               m_Phi: int,
               m_X: int,
               epsilon: float = 0.1,
               noise_sigma: float = 0.0,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> tuple[NDArray, callable]:
    """
    Algorytm 1 — aproksymacja funkcji grzbietowej f(x) = g(a · x)
    dla przypadku k = 1.

    Kroki:
    1. Wylosuj zbiory Φ (m_Φ kierunków Bernoulliego) i X (m_X punktów na sferze).
    2. Skonstruuj macierz Y ilorazów różnicowych.
    3. Dla każdej kolumny y_j rozwiąż min ||z||_{l1} s.t. Φz = y_j → x̂_j.
    4. Znajdź j_0 = argmax_j ||x̂_j||_{l2}.
    5. Znormalizuj: â = x̂_{j0} / ||x̂_{j0}||_{l2}.
    6. Zdefiniuj ĝ(y) = f(â^T · y) i f̂(x) = ĝ(â · x).

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki f: R^d → R.
    d : int
        Wymiar przestrzeni.
    m_Phi : int
        Liczba kierunków pomiarowych.
    m_X : int
        Liczba punktów próbkowania na sferze.
    epsilon : float
        Krok różnicy skończonej (domyślnie 0.1).
    noise_sigma : float
        Odchylenie standardowe szumu Gaussa dodawanego do pomiarów
        (0.0 = brak szumu).
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    a_hat : NDArray o kształcie (d,)
        Estymata wektora kierunkowego â.
    f_hat : callable
        Funkcja aproksymująca f̂(x) = f(â^T (â · x)).
    """
    if rng is None:
        rng = np.random.default_rng()

    # (1) Losowanie zbiorów
    Xi = sample_sphere(d, m_X, rng)                    # m_X × d
    Phi = bernoulli_matrix(m_Phi, d, rng)               # m_Phi × d

    # Kierunki φ_i z (16): wiersze Φ to już φ_i / √m_Phi,
    # ale w ilorazie różnicowym potrzebujemy φ_i ∈ B(√(d/m_Phi)).
    # Wiersze Phi mają normę ≈ √(d/m_Phi) po przeskalowaniu 1/√m_Phi.
    # Używamy wierszy Phi bezpośrednio jako kierunków w finite diff.
    Phi_directions = Phi  # norma wiersza ≈ √(d/m_Phi)

    # (2) Macierz Y ilorazów różnicowych
    Y = _build_finite_differences(f, Xi, Phi_directions, epsilon)

    # Opcjonalnie: dodanie szumu do Y
    if noise_sigma > 0:
        W = rng.normal(0, noise_sigma, size=Y.shape)
        Y = Y + W / epsilon

    # (3) Minimalizacja l1 dla każdej kolumny
    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        y_j = Y[:, j]
        if noise_sigma > 0:
            # Dla danych zaszumionych: relaxed BP
            tol = noise_sigma * np.sqrt(m_Phi) / epsilon
            X_hat[:, j] = l1_minimize_noisy(Phi, y_j, sigma=tol)
        else:
            X_hat[:, j] = l1_minimize(Phi, y_j)

        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  l1-min: kolumna {j+1}/{m_X} zakończona")

    # (4) Wybór optymalnego kierunku
    norms = np.linalg.norm(X_hat, axis=0)
    j0 = np.argmax(norms)

    if verbose:
        print(f"  j0 = {j0}, ||x̂_j0||_2 = {norms[j0]:.6f}")

    # (5) Normalizacja
    a_hat = X_hat[:, j0]
    if np.linalg.norm(a_hat) > 1e-12:
        a_hat = a_hat / np.linalg.norm(a_hat)
    else:
        raise RuntimeError(
            "Algorytm 1: nie udało się odzyskać wektora kierunkowego "
            "(norma ≈ 0). Zwiększ m_Φ lub m_X."
        )

    # (6) Konstrukcja f̂
    def f_hat(x):
        return f(a_hat * (a_hat @ x))

    return a_hat, f_hat


def algorithm2(f: callable,
               d: int,
               k: int,
               m_Phi: int,
               m_X: int,
               epsilon: float = 0.1,
               noise_sigma: float = 0.0,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> tuple[NDArray, callable]:
    """
    Algorytm 2 — aproksymacja funkcji k-grzbietowej f(x) = g(Ax)
    dla przypadku k ≥ 1.

    Kroki:
    1. Wylosuj zbiory Φ i X, skonstruuj macierz Y.
    2. Dla każdej kolumny y_j: x̂_j = argmin ||z||_{l1} s.t. Φz = y_j.
    3. SVD transpozycji X̂^T = Û₁ Σ̂₁ V̂₁^T + Û₂ Σ̂₂ V̂₂^T.
    4. Â = V̂₁^T (k prawych wektorów osobliwych).
    5. ĝ(y) = f(Â^T y), f̂(x) = ĝ(Âx).

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki.
    d : int
        Wymiar przestrzeni.
    k : int
        Wymiar ukryty (liczba aktywnych kierunków).
    m_Phi : int
        Liczba kierunków pomiarowych.
    m_X : int
        Liczba punktów próbkowania.
    epsilon : float
        Krok różnicy skończonej.
    noise_sigma : float
        Odchylenie standardowe szumu (0 = brak).
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    A_hat : NDArray o kształcie (k, d)
        Estymata macierzy parametrów Â.
    f_hat : callable
        Funkcja aproksymująca f̂(x) = f(Â^T (Â x)).
    """
    if rng is None:
        rng = np.random.default_rng()

    # (1) Losowanie zbiorów
    Xi = sample_sphere(d, m_X, rng)
    Phi = bernoulli_matrix(m_Phi, d, rng)
    Phi_directions = Phi

    # (2) Macierz Y
    Y = _build_finite_differences(f, Xi, Phi_directions, epsilon)

    if noise_sigma > 0:
        W = rng.normal(0, noise_sigma, size=Y.shape)
        Y = Y + W / epsilon

    # (3) Minimalizacja l1
    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        y_j = Y[:, j]
        if noise_sigma > 0:
            tol = noise_sigma * np.sqrt(m_Phi) / epsilon
            X_hat[:, j] = l1_minimize_noisy(Phi, y_j, sigma=tol)
        else:
            X_hat[:, j] = l1_minimize(Phi, y_j)

        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  l1-min: kolumna {j+1}/{m_X} zakończona")

    # (4) SVD transpozycji X̂^T
    # X̂ ma wymiar d × m_X, więc X̂^T ma wymiar m_X × d
    U, S, Vt = np.linalg.svd(X_hat.T, full_matrices=False)

    if verbose:
        print(f"  Wartości osobliwe X̂^T: {S[:min(k+2, len(S))]}")
        if k < len(S):
            print(f"  Przerwa spektralna: σ_{k} = {S[k-1]:.6f}, "
                  f"σ_{k+1} = {S[k] if k < len(S) else 0:.6f}")

    # k prawych wektorów osobliwych (wiersze V^T)
    A_hat = Vt[:k, :]  # kształt (k, d)

    # (5) Konstrukcja f̂
    def f_hat(x):
        return f(A_hat.T @ (A_hat @ x))

    return A_hat, f_hat


def identify_active_coordinates(f: callable,
                                 d: int,
                                 k: int,
                                 m_Phi: int,
                                 m_X: int,
                                 epsilon: float = 0.1,
                                 noise_sigma: float = 0.0,
                                 threshold: float = 0.1,
                                 rng: np.random.Generator | None = None) -> list[int]:
    """
    Identyfikacja aktywnych współrzędnych funkcji f na podstawie
    odzyskanej macierzy Â.

    Współrzędna j jest uznana za aktywną, jeśli norma kolumny j
    macierzy Â^T przekracza próg (threshold).

    Procedura odpowiada eksperymentowi z Figure 2 pracy Vybirala.

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki.
    d : int
        Wymiar przestrzeni.
    k : int
        Wymiar ukryty.
    m_Phi : int
        Liczba kierunków.
    m_X : int
        Liczba punktów próbkowania.
    epsilon : float
        Krok różnicy skończonej.
    noise_sigma : float
        Poziom szumu.
    threshold : float
        Próg aktywności współrzędnej.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.

    Zwraca
    -------
    list[int]
        Lista indeksów aktywnych współrzędnych.
    """
    if k == 1:
        a_hat, _ = algorithm1(f, d, m_Phi, m_X, epsilon, noise_sigma, rng)
        active = [i for i in range(d) if abs(a_hat[i]) > threshold]
    else:
        A_hat, _ = algorithm2(f, d, k, m_Phi, m_X, epsilon, noise_sigma, rng)
        col_norms = np.linalg.norm(A_hat, axis=0)
        active = [i for i in range(d) if col_norms[i] > threshold]

    return active
