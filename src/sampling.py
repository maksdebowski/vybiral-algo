"""
Moduł odpowiedzialny za generowanie losowych zbiorów próbkowania
wykorzystywanych w algorytmach aproksymacji funkcji grzbietowych.

Zawiera:
- Losowanie punktów z jednostajnej miary powierzchniowej na sferze S^{d-1}
- Generowanie macierzy pomiarowej Φ (Bernoulli, przeskalowana 1/√m)
"""

import numpy as np
from numpy.typing import NDArray


def sample_sphere(d: int, m: int, rng: np.random.Generator | None = None) -> NDArray:
    """
    Losuje m punktów z jednostajnej miary powierzchniowej μ_{S^{d-1}}
    na sferze jednostkowej S^{d-1} w R^d.

    Metoda: normalizacja wektorów z rozkładu N(0, I_d).

    Parametry
    ----------
    d : int
        Wymiar przestrzeni.
    m : int
        Liczba punktów do wylosowania.
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.

    Zwraca
    -------
    NDArray o kształcie (m, d)
        Macierz, której wiersze to punkty na sferze S^{d-1}.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = rng.standard_normal((m, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def bernoulli_matrix(m: int, d: int, rng: np.random.Generator | None = None) -> NDArray:
    """
    Generuje macierz pomiarową Φ o wymiarze m × d,
    składającą się z niezależnych zmiennych Bernoulliego
    przeskalowanych przez 1/√m.

    Φ_{iℓ} = (1/√m) * {+1 z pr. 1/2, -1 z pr. 1/2}

    Zgodnie z (16) z pracy Fornasiera, Schnassa i Vybirala.

    Parametry
    ----------
    m : int
        Liczba wierszy (kierunków pomiarowych m_Φ).
    d : int
        Liczba kolumn (wymiar przestrzeni).
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.

    Zwraca
    -------
    NDArray o kształcie (m, d)
        Macierz pomiarowa Φ.
    """
    if rng is None:
        rng = np.random.default_rng()
    Phi = rng.choice([-1, 1], size=(m, d)).astype(np.float64)
    Phi /= np.sqrt(m)
    return Phi
