"""
Implementacja Algorytmu 1 ($k=1$) i Algorytmu 2 ($k \\geq 1$)
z pracy Fornasiera, Schnassa i Vybirala:
"Learning Functions of Few Arbitrary Linear Parameters in High Dimensions"

Algorytm 1 -- odzyskiwanie pojedynczego wektora kierunkowego $a$
              dla funkcji grzbietowej $f(x) = g(a \\cdot x)$.

Algorytm 2 -- odzyskiwanie macierzy parametrów $A$ ($k \\times d$)
              dla $k$-grzbietowej $f(x) = g(Ax)$ za pomocą SVD.
"""

import numpy as np
from numpy.typing import NDArray

from .sampling import sample_sphere, bernoulli_matrix
from .l1_solver import l1_minimize


def _build_finite_differences(
    f: callable, Xi: NDArray, Phi_rows: NDArray, epsilon: float
) -> NDArray:
    """
    Konstruuje macierz $Y$ ilorazów różnicowych (finite differences).

    \\[
    y_{ij} = \\frac{f(\\xi_j + \\epsilon \\cdot \\phi_i) -
    f(\\xi_j)}{\\epsilon}
    \\]

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}$.
    Xi : NDArray o kształcie (m_X, d)
        Punkty próbkowania na sferze.
    Phi_rows : NDArray o kształcie (m_Phi, d)
        Kierunki pomiarowe (wiersze macierzy $\\Phi$, ale NIEPRZESKALOWANE,
        tj. $\\phi_i$ z normą $\\approx \\sqrt{d/m_\\Phi}$).
    epsilon : float
        Krok różnicy skończonej $\\epsilon > 0$.

    Zwraca
    -------
    NDArray o kształcie (m_Phi, m_X)
        Macierz $Y$.
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


def algorithm1(
    f: callable,
    d: int,
    m_Phi: int,
    m_X: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[NDArray, callable]:
    """
    Algorytm 1 -- aproksymacja funkcji grzbietowej $f(x) = g(a \\cdot x)$
    dla przypadku $k = 1$.

    Kroki:
    1. Wylosuj zbiory $\\Phi$ ($m_\\Phi$ kierunków Bernoulliego) i $X$
    ($m_X$ punktów na sferze).
    2. Skonstruuj macierz $Y$ ilorazów różnicowych.
    3. Dla każdej kolumny $y_j$ rozwiąż BPDN (nie BP) z tolerancją
    zależną od szumu.
    4. Znajdź $j_0 = \\mathrm{argmax}_j \\|\\hat{x}_j\\|_2$.
    5. Znormalizuj: $\\hat{a} = \\hat{x}_{j_0} / \\|\\hat{x}_{j_0}\\|_2$.
    6. Zdefiniuj $\\hat{f}(x) = f(\\hat{a}^T (\\hat{a} \\cdot x))$.

    Model szumu: `noise_sigma` ($\\sigma_f$) to odchylenie standardowe
    szumu jednotej obserwacji $f(x)$. Do każdej z dwóch ewaluacji
    tworzących iloraz różnicowy dodawany jest niezależny szum
    $\\mathcal{N}(0, \\sigma_f^2)$. Szum na ilorazie różnicowym ma wtedy
    odchylenie $\\sigma_{\\mathrm{FD}} = \\sqrt{2}\\,\\sigma_f / \\epsilon$.
    Ten sam punkt bazowy $f(\\xi_j)$ jest współdzielony przez wszystkie
    kierunki $\\phi_i$ dla danego $j$, co wprowadza korelację błędów
    w jednej kolumnie $y_j$ macierzy $Y$.

    Parametry
    ----------
    f : callable
        Funkcja czarnej skrzynki $f: \\mathbb{R}^d \\rightarrow \\mathbb{R}$.
    d : int
        Wymiar przestrzeni.
    m_Phi : int
        Liczba kierunków pomiarowych.
    m_X : int
        Liczba punktów próbkowania na sferze.
    epsilon : float
        Krok różnicy skończonej (domyślnie 0.1).
    noise_sigma : float
        Odchylenie standardowe szumu Gaussa na pojedynczej obserwacji
        $f(x)$ (0.0 = brak szumu). Szum na ilorazie różnicowym wynosi
        $\\sqrt{2}\\,\\sigma_f / \\epsilon$ (gdzie $\\sigma_f$ = noise_sigma).
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    a_hat : NDArray o kształcie (d,)
        Estymata wektora kierunkowego $\\hat{a}$.
    f_hat : callable
        Funkcja aproksymująca
        $\\hat{f}(x) = f(\\hat{a}^T (\\hat{a} \\cdot x))$.
    """
    if rng is None:
        rng = np.random.default_rng()

    # (1) Losowanie zbiorów
    Xi = sample_sphere(d, m_X, rng)  # m_X \\times d
    Phi = bernoulli_matrix(m_Phi, d, rng)  # m_Phi \\times d

    # Kierunki $\\phi_i$ z (16): wiersze $\\Phi$ to już
    # $\\phi_i / \\sqrt{m_\\Phi}$,
    # ale w ilorazie różnicowym potrzebujemy
    # $\\phi_i \\in B(\\sqrt{d/m_\\Phi})$.
    # Wiersze Phi mają normę $\\approx \\sqrt{d/m_\\Phi}$
    # po przeskalowaniu $1/\\sqrt{m_\\Phi}$.
    # Używamy wierszy Phi bezpośrednio jako kierunków w finite diff.
    Phi_directions = Phi  # norma wiersza $\\approx \\sqrt{d/m_\\Phi}$

    # (2) Macierz Y ilorazów różnicowych
    Y = _build_finite_differences(f, Xi, Phi_directions, epsilon)

    # Opcjonalnie: dodanie szumu do Y.
    # Szum jest dodawany niezależnie do każdej ewaluacji funkcji f.
    # Współdzielony f(xi_j) wprowadza korelację w kolumnie Y[:,j].
    # Efektywny szum FD: N(0, (sqrt(2)*noise_sigma/epsilon)^2).
    if noise_sigma > 0:
        # Szum na f(xi_j + eps*phi_i) -- niezależny dla każdego (i,j)
        noise_shifted = rng.normal(0, noise_sigma, size=(m_Phi, m_X))
        # Szum na f(xi_j) -- współdzielony przez wszystkie kierunki i
        noise_base = rng.normal(0, noise_sigma, size=(1, m_X))
        Y = Y + (noise_shifted - noise_base) / epsilon

    # (3) Minimalizacja l1 dla każdej kolumny (BPDN we wszystkich przypadkach)
    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        y_j = Y[:, j]
        if noise_sigma > 0:
            # sigma_fd = sqrt(2)*sigma_f/epsilon (szum ilorazu różnicowego)
            sigma_fd = np.sqrt(2) * noise_sigma / epsilon
            noise_tol = sigma_fd * np.sqrt(m_Phi)
            taylor_tol = 0.05 * np.linalg.norm(y_j)
            X_hat[:, j] = l1_minimize(
                Phi, y_j, tol=noise_tol + taylor_tol + 1e-6
            )
        else:
            X_hat[:, j] = l1_minimize(Phi, y_j)

        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  l1-min: kolumna {j+1}/{m_X} zakończona")

    # (4) Wybór optymalnego kierunku
    norms = np.linalg.norm(X_hat, axis=0)
    j0 = np.argmax(norms)

    if verbose:
        print(f"j0 = {j0}" + ", \\|\\hat{x}_{j_0}\\|_2 =" + f"{norms[j0]:.6f}")

    # (5) Normalizacja
    a_hat = X_hat[:, j0]
    if np.linalg.norm(a_hat) > 1e-12:
        a_hat = a_hat / np.linalg.norm(a_hat)
    else:
        raise RuntimeError(
            "Algorytm 1: nie udało się odzyskać wektora kierunkowego "
            "(norma $\\\\approx$ 0). Zwiększ $m_\\\\Phi$ lub $m_X$."
        )

    # (6) Konstrukcja $\\hat{f}$
    def f_hat(x):
        return f(a_hat * (a_hat @ x))

    return a_hat, f_hat


def algorithm2(
    f: callable,
    d: int,
    k: int,
    m_Phi: int,
    m_X: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[NDArray, callable]:
    """
    Algorytm 2 -- aproksymacja funkcji k-grzbietowej $f(x) = g(Ax)$
    dla przypadku $k \\geq 1$.

    Kroki:
    1. Wylosuj zbiory $\\Phi$ i $X$, skonstruuj macierz $Y$.
    2. Dla każdej kolumny $y_j$: BPDN (nie BP).
    3. SVD transpozycji $\\hat{X}^T = \\hat{U}_1
    \\hat{\\Sigma}_1 \\hat{V}_1^T + \\hat{U}_2 \\hat{\\Sigma}_2 \\hat{V}_2^T$.
    4. $\\hat{A} = \\hat{V}_1^T$ ($k$ prawych wektorów osobliwych).
    5. $\\hat{f}(x) = f(\\hat{A}^T (\\hat{A} x))$.

    Model szumu: analogiczny jak w `algorithm1`. `noise_sigma` ($\\sigma_f$)
    to odchylenie standardowe szumu pojedynczej obserwacji $f(x)$.
    Szum na ilorazie różnicowym: $\\sigma_{\\mathrm{FD}} =
    \\sqrt{2}\\,\\sigma_f / \\epsilon$. Punkt bazowy $f(\\xi_j)$ jest
    współdzielony przez wszystkie wiersze dla danego $j$.

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
        Odchylenie standardowe szumu na pojedynczej obserwacji $f(x)$
        (0 = brak szumu).
    rng : np.random.Generator, opcjonalnie
        Generator liczb losowych.
    verbose : bool
        Wypisuj informacje diagnostyczne.

    Zwraca
    -------
    A_hat : NDArray o kształcie (k, d)
        Estymata macierzy parametrów $\\hat{A}$.
    f_hat : callable
        Funkcja aproksymująca $\\hat{f}(x) = f(\\hat{A}^T (\\hat{A} x))$.
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
        noise_shifted = rng.normal(0, noise_sigma, size=(m_Phi, m_X))
        noise_base = rng.normal(0, noise_sigma, size=(1, m_X))
        Y = Y + (noise_shifted - noise_base) / epsilon

    # (3) Minimalizacja l1 (BPDN we wszystkich przypadkach)
    X_hat = np.zeros((d, m_X))
    for j in range(m_X):
        y_j = Y[:, j]
        if noise_sigma > 0:
            sigma_fd = np.sqrt(2) * noise_sigma / epsilon
            noise_tol = sigma_fd * np.sqrt(m_Phi)
            taylor_tol = 0.05 * np.linalg.norm(y_j)
            X_hat[:, j] = l1_minimize(
                Phi, y_j, tol=noise_tol + taylor_tol + 1e-6
            )
        else:
            X_hat[:, j] = l1_minimize(Phi, y_j)

        if verbose and (j + 1) % max(1, m_X // 5) == 0:
            print(f"  l1-min: kolumna {j+1}/{m_X} zakończona")

    # (4) SVD transpozycji $\\hat{X}^T$
    # $\\hat{X}$ ma wymiar $d \\times m_X$, więc $\\hat{X}^T$ ma wymiar
    # $m_X \\times d$
    U, S, Vt = np.linalg.svd(X_hat.T, full_matrices=False)

    if verbose:
        print(f"  Wartości osobliwe \\hat{{X}}^T: {S[:min(k+2, len(S))]}")
        if k < len(S):
            print(
                f"  Przerwa spektralna: \\sigma_{k} = {S[k-1]:.6f}, "
                f"\\sigma_{{k+1}} = {S[k] if k < len(S) else 0:.6f}"
            )

    # k prawych wektorów osobliwych (wiersze V^T)
    A_hat = Vt[:k, :]  # kształt (k, d)

    # (5) Konstrukcja $\\hat{f}$
    def f_hat(x):
        return f(A_hat.T @ (A_hat @ x))

    return A_hat, f_hat


def identify_active_coordinates(
    f: callable,
    d: int,
    k: int,
    m_Phi: int,
    m_X: int,
    epsilon: float = 0.1,
    noise_sigma: float = 0.0,
    threshold: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Identyfikacja aktywnych współrzędnych funkcji $f$ na podstawie
    odzyskanej macierzy $\\hat{A}$.

    Współrzędna $j$ jest uznana za aktywną, jeśli norma kolumny $j$
    macierzy $\\hat{A}^T$ przekracza próg (threshold).

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
