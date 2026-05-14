"""
Moduł rozwiązywania problemów minimalizacji normy l1
(basis pursuit) wykorzystywanych w compressed sensing.

Problem:
    x̂ = arg min ||z||_{l1}  subject to  Φz = y

Implementacja oparta na solverze CVXPY.
"""

import numpy as np
import cvxpy as cp
from numpy.typing import NDArray


def l1_minimize(
    Phi: NDArray, y: NDArray, solver: str = "SCS", tol: float = 1e-4
) -> NDArray:
    """
    Rozwiązuje problem basis pursuit (z niewielką tolerancją numeryczną):
        x̂ = argmin ||z||_{l1^d}  subject to  ||Φz - y||_2 ≤ tol

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa (Bernoulli, przeskalowana).
    y : NDArray o kształcie (m,)
        Wektor obserwacji (kolumna macierzy Y).
    solver : str
        Solver CVXPY do użycia (domyślnie "SCS").
    tol : float
        Tolerancja na ograniczenie równościowe.

    Zwraca
    -------
    NDArray o kształcie (d,)
        Rozwiązanie x̂ — estymata wektora kierunkowego.
    """
    d = Phi.shape[1]
    z = cp.Variable(d)
    objective = cp.Minimize(cp.norm(z, 1))
    constraints = [cp.norm(Phi @ z - y, 2) <= tol]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if z.value is None:
        raise RuntimeError("Solver l1 nie znalazł rozwiązania.")
    return np.array(z.value).flatten()


def l1_minimize_noisy(
    Phi: NDArray, y: NDArray, sigma: float, solver: str = "SCS"
) -> NDArray:
    """
    Rozwiązuje problem Dantzig selector / LASSO dla danych zaszumionych:
        x̂ = argmin ||z||_{l1}  subject to  ||Φz - y||_2 ≤ σ√m

    Używany w przypadku pomiarów z szumem (sekcja 5.1 Vybirala).

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa.
    y : NDArray o kształcie (m,)
        Zaszumiony wektor obserwacji.
    sigma : float
        Poziom szumu (odchylenie standardowe).
    solver : str
        Solver CVXPY.

    Zwraca
    -------
    NDArray o kształcie (d,)
        Rozwiązanie x̂.
    """
    m, d = Phi.shape
    z = cp.Variable(d)
    objective = cp.Minimize(cp.norm(z, 1))
    constraints = [cp.norm(Phi @ z - y, 2) <= sigma * np.sqrt(m)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if z.value is None:
        raise RuntimeError("Solver l1 (noisy) nie znalazł rozwiązania.")
    return np.array(z.value).flatten()
