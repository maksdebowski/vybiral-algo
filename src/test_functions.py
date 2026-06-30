"""
Moduł definiujący funkcje testowe wykorzystywane w eksperymentach numerycznych.

Zawiera:
- Funkcję testową z równania (63) pracy Vybirala et al.
- Proste funkcje grzbietowe (ridge functions) do walidacji Algorytmu 1
- Funkcje k-grzbietowe do walidacji Algorytmu 2
"""

import numpy as np
from numpy.typing import NDArray


class RidgeFunction:
    """
    Funkcja grzbietowa f(x) = g(a · x), gdzie a ∈ S^{d-1}.

    Parametry
    ----------
    a : NDArray o kształcie (d,)
        Wektor kierunkowy (znormalizowany do ||a||=1).
    g : callable
        Funkcja jednej zmiennej g: R → R.
    """

    def __init__(self, a: NDArray, g: callable):
        self.a = a / np.linalg.norm(a)
        self.g = g
        self.d = len(a)
        self.k = 1

    def __call__(self, x: NDArray) -> float:
        return self.g(self.a @ x)


class KRidgeFunction:
    """
    Funkcja k-grzbietowa f(x) = g(Ax), gdzie A ∈ M_{k×d}, AA^T = I_k.

    Parametry
    ----------
    A : NDArray o kształcie (k, d)
        Macierz parametrów liniowych z ortonormalnymi wierszami.
    g : callable
        Funkcja g: R^k → R.
    """

    def __init__(self, A: NDArray, g: callable):
        self.A = A
        self.g = g
        self.k = A.shape[0]
        self.d = A.shape[1]

    def __call__(self, x: NDArray) -> float:
        return self.g(self.A @ x)


def vybiral_test_function(x: NDArray) -> float:
    """
    Funkcja testowa z równania (63) pracy Vybirala et al.:
        f(x) = max(1 - 5 * sqrt((x_3 - 1/2)^2 + (x_4 - 1/2)^2), 0)^3

    Funkcja zdefiniowana na R^d (d ≥ 4), zależy tylko od
    współrzędnych x[2] i x[3] (indeksowanie od 0).

    To jest funkcja 2-grzbietowa z aktywnymi współrzędnymi {3, 4}
    (indeksowanie od 1, jak w pracy).

    Parametry
    ----------
    x : NDArray o kształcie (d,)
        Punkt ewaluacji, d ≥ 4.

    Zwraca
    -------
    float
        Wartość f(x).
    """
    r = np.sqrt((x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2)
    val = 1.0 - 5.0 * r
    return max(val, 0.0) ** 3


def make_sparse_ridge_function(
    d: int,
    active_indices: list[int],
    active_values: NDArray | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Tworzy funkcję grzbietową f(x) = g(a · x) z rzadkim wektorem a,
    gdzie niezerowe współrzędne to active_indices.

    g(t) = cos(π·t) — prosta nieliniowa funkcja spełniająca
    warunki gładkości z modelu (C2, wartości oraz pochodne ograniczone).

    Parametry
    ----------
    d : int
        Wymiar przestrzeni.
    active_indices : list[int]
        Indeksy niezerowych współrzędnych wektora a.
    active_values : NDArray, opcjonalnie
        Wartości niezerowych współrzędnych. Jeśli None, losowe.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.

    Zwraca
    -------
    RidgeFunction
        Funkcja grzbietowa z rzadkim wektorem a.
    NDArray
        Prawdziwy wektor a (ground truth).
    """
    if rng is None:
        rng = np.random.default_rng()

    a = np.zeros(d)
    if active_values is not None:
        a[active_indices] = active_values
    else:
        a[active_indices] = rng.standard_normal(len(active_indices))
    a /= np.linalg.norm(a)

    def g(t):
        return np.cos(np.pi * t)

    return RidgeFunction(a, g), a


def make_k_ridge_function(
    d: int,
    k: int,
    active_indices: list[int] | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Tworzy funkcję k-grzbietową f(x) = g(Ax) z rzadką macierzą A.

    g(y) = ||y||^2 — prosta funkcja radialna spełniająca warunki modelu
    (C2, wartości oraz pochodne ograniczone).

    Parametry
    ----------
    d : int
        Wymiar przestrzeni.
    k : int
        Wymiar ukryty (liczba wierszy macierzy A).
    active_indices : list[int], opcjonalnie
        Indeksy aktywnych współrzędnych. Jeśli None, pierwsze k*2.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.

    Zwraca
    -------
    KRidgeFunction
        Funkcja k-grzbietowa.
    NDArray
        Macierz A (ground truth) o kształcie (k, d).
    """
    if rng is None:
        rng = np.random.default_rng()

    if active_indices is None:
        active_indices = list(range(min(k * 2, d)))

    n_active = len(active_indices)
    A = np.zeros((k, d))
    A_sub = rng.standard_normal((k, n_active))
    # Ortonormalizacja wierszy (QR na A_sub^T)
    Q, _ = np.linalg.qr(A_sub.T)
    A[:, active_indices] = Q[:, :k].T

    def g(y):
        return np.sum(y**2)

    return KRidgeFunction(A, g), A
