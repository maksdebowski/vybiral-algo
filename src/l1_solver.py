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
    Phi: NDArray, y: NDArray, solver: str = "CLARABEL", tol: float | None = None
) -> NDArray:
    """
    Rozwiązuje problem basis pursuit (BPDN):
        x̂ = argmin ||z||_{l1^d}  subject to  ||Φz - y||_2 ≤ tol

    Jeśli tol=None, tolerancja jest estymowana automatycznie jako
    ε_mach · ||y||_2 · √d, co odpowiada spodziewanemu residuum
    aproksymacji Taylora w różnicach skończonych.

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa (Bernoulli, przeskalowana).
    y : NDArray o kształcie (m,)
        Wektor obserwacji (kolumna macierzy Y).
    solver : str
        Solver CVXPY do użycia (domyślnie "CLARABEL").
    tol : float | None
        Tolerancja na ograniczenie. Jeśli None — auto-estymacja.

    Zwraca
    -------
    NDArray o kształcie (d,)
        Rozwiązanie x̂ — estymata wektora kierunkowego.
    """
    m, d = Phi.shape
    if tol is None:
        # Heurystyka: tolerancja proporcjonalna do siły sygnału.
        # Residuum Taylora ≈ O(ε · ||∇²f|| · d/m) → w praktyce ~5–10% ||y||.
        tol = 0.05 * np.linalg.norm(y) + 1e-6
    z = cp.Variable(d)
    objective = cp.Minimize(cp.norm(z, 1))
    constraints = [cp.norm(Phi @ z - y, 2) <= tol]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if z.value is None:
        raise RuntimeError("Solver l1 nie znalazł rozwiązania.")
    return np.array(z.value).flatten()


def l1_minimize_noisy(
    Phi: NDArray, y: NDArray, sigma: float, solver: str = "CLARABEL"
) -> NDArray:
    """
    Rozwiązuje problem BPDN dla danych zaszumionych:
        x̂ = argmin ||z||_{l1}  subject to  ||Φz - y||_2 ≤ tol

    Tolerancja łączy szum pomiarowy i residuum Taylora:
        tol = σ·√m + 0.1·||y||₂

    Używany w przypadku pomiarów z szumem (sekcja 5.1 Vybirala).

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa.
    y : NDArray o kształcie (m,)
        Zaszumiony wektor obserwacji.
    sigma : float
        Poziom szumu (efektywny, po podzieleniu przez ε).
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
    # Łączymy szum pomiarowy i residuum Taylora
    tol = sigma * np.sqrt(m) + 0.05 * np.linalg.norm(y) + 1e-6
    constraints = [cp.norm(Phi @ z - y, 2) <= tol]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=False)

    if z.value is None:
        raise RuntimeError("Solver l1 (noisy) nie znalazł rozwiązania.")
    return np.array(z.value).flatten()
