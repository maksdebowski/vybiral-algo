"""
Moduł rozwiązywania problemów minimalizacji normy l1
wykorzystywanych w compressed sensing.

We wszystkich eksperymentach stosowane jest BPDN (Basis Pursuit DeNoising):
    $\\hat{x} = \\mathrm{argmin} \\|z\\|_{\\ell_1} \\quad \\text{s.t.}
    \\quad \\|\\Phi z - y\\|_2 \\leq \\mathrm{tol}$

Nawet w przypadku bezszumowym używamy BPDN z tolerancją wynikającą
z residuum aproksymacji Taylora, a nie dokładnego ograniczenia równościowego
$\\Phi z = y$. Jest to świadoma decyzja implementacyjna — dokładne BP
(equal-constraint) jest numerycznie wrażliwe i wymaga idealnie spójnych
pomiarów. Tolerancja jest heurystyką opartą na błędzie Taylora pierwszego
rzędu ilorazu różnicowego.

Implementacja oparta na solverze CVXPY.
"""

import warnings

import numpy as np
import cvxpy as cp
from numpy.typing import NDArray


def l1_minimize(
    Phi: NDArray,
    y: NDArray,
    solver: str = "CLARABEL",
    tol: float | None = None,
    tol_alpha: float = 0.05,
) -> NDArray:
    """
    Rozwiązuje problem BPDN (Basis Pursuit DeNoising):
        $\\hat{x} = \\mathrm{argmin} \\|z\\|_{\\ell_1^d} \\quad
        \\text{s.t.} \\quad \\|\\Phi z - y\\|_2 \\leq \\mathrm{tol}$

    UWAGA: We wszystkich eksperymentach — zarówno bezszumowych, jak
    i zaszumionych — stosowane jest BPDN, nie dokładne Basis Pursuit
    z ograniczeniem równościowym $\\Phi z = y$.

    Jeśli tol=None, tolerancja wyznaczana jest jako:
        $\\mathrm{tol} = \\alpha \\cdot \\|y\\|_2 + 10^{-6}$
    gdzie $\\alpha$ = tol_alpha (domyślnie 0.05) jest parametrem
    heurystycznym odpowiadającym przybliżonemu rzędowi błędu Taylora
    pierwszego rzędu ilorazu różnicowego:
        $\\frac{f(\\xi + \\epsilon \\phi) - f(\\xi)}{\\epsilon}
        - (\\nabla f(\\xi) \\cdot \\phi)
        = \\mathcal{O}\\left(\\epsilon \\|\\nabla^2 f\\|_\\infty
        \\|\\phi\\|_2\\right)$.
    Wartość $\\alpha = 0.05$ jest heurystyką — nie wynika bezpośrednio
    z żadnego twierdzenia i może wpływać na obserwowane plateau błędu.
    Do analizy wrażliwości na ten parametr służy
    `experiment_tolerance_sensitivity`.

    Parametry
    ----------
    Phi : NDArray o kształcie (m, d)
        Macierz pomiarowa (Bernoulli, przeskalowana).
    y : NDArray o kształcie (m,)
        Wektor obserwacji (kolumna macierzy Y).
    solver : str
        Solver CVXPY do użycia (domyślnie "CLARABEL").
    tol : float | None
        Tolerancja na ograniczenie. Jeśli None -- wyznaczana z tol_alpha.
    tol_alpha : float
        Współczynnik heurystyki tolerancji (domyślnie 0.05).
        Używany tylko gdy tol=None.

    Zwraca
    -------
    NDArray o kształcie (d,)
        Rozwiązanie $\\hat{x}$ -- estymata wektora kierunkowego.
    """
    m, d = Phi.shape
    if tol is None:
        # Heurystyka tolerancji BPDN: tol = alpha * ||y||_2 + 1e-6.
        # Uzasadnienie: residuum Taylora ilorazu różnicowego wynosi
        # O(epsilon * ||nabla^2 f|| * ||phi||), co w praktyce daje
        # ~alpha * ||y||. Wartość alpha=0.05 jest heurystyką;
        # patrz docstring powyżej oraz experiment_tolerance_sensitivity.
        tol = tol_alpha * np.linalg.norm(y) + 1e-6

    if np.linalg.norm(y) < 1e-10:
        return np.zeros(d)

    z = cp.Variable(d)
    objective = cp.Minimize(cp.norm(z, 1))
    constraints = [cp.norm(Phi @ z - y, 2) <= tol]
    problem = cp.Problem(objective, constraints)

    solvers_to_try = [solver, "SCS"]
    for slv in solvers_to_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=slv, verbose=False)
            if z.value is not None:
                return np.array(z.value).flatten()
        except (cp.error.SolverError, Exception):
            pass

    raise RuntimeError("Solver l1 nie znalazł rozwiązania.")
