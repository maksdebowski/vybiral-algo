# Opisy wykresów do pracy magisterskiej

Wykresy zostały wygenerowane dwoma skryptami:

- `generate_figures.py` — 9 figur ilustrujących działanie algorytmów Vybirala
- `generate_comparison_figures.py` — figurki porównawcze z metodami bazowymi
  
Format: **JPG**. Aby wygenerować ponownie:

```bash
python generate_figures.py
python generate_comparison_figures.py
```

---

## Kluczowe zmiany w kodzie (uzgodnienie BP/BPDN i modelu szumu)

### 1. BPDN we wszystkich eksperymentach (nie dokładne Basis Pursuit)

We wszystkich eksperymentach — zarówno bezszumowych, jak i zaszumionych —
stosowane jest **BPDN** (Basis Pursuit DeNoising):
$$\hat{x} = \arg\min \|z\|_{\ell_1} \quad \text{s.t.} \quad \|\Phi z - y\|_2 \leq \mathrm{tol}$$
Nie jest używane dokładne Basis Pursuit z ograniczeniem równościowym $\Phi z = y$.

W **przypadku bezszumowym** tolerancja wyznaczana jest heurystycznie jako:
$$\mathrm{tol} = \alpha \cdot \|y\|_2 + 10^{-6}, \quad \alpha = 0.05 \text{ (domyślnie)}$$
Wartość $\alpha = 0.05$ jest heurystyką opartą na rzędzie błędu Taylora
ilorazu różnicowego: reszta Taylora pierwszego rzędu wynosi
$O(\epsilon \|\nabla^2 f\|_\infty \|\phi\|_2)$, co w praktyce odpowiada
$\approx 5\%\|y\|$. **Nie wynika to z żadnego twierdzenia** — może wpływać
na obserwowane plateau błędu (patrz Rysunek 9 tolerancji w `experiments.py`).

W **przypadku zaszumionym** tolerancja uwzględnia dodatkowo szum pomiarowy:
$$\mathrm{tol} = \sigma_{\mathrm{FD}} \cdot \sqrt{m} + \alpha \cdot \|y\|_2 + 10^{-6}$$
gdzie $\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f / \epsilon$ (patrz niżej).

Do analizy wrażliwości na parametr $\alpha$ służy funkcja
`experiment_tolerance_sensitivity` w `src/experiments.py`.

### 2. Poprawiony model szumu

Parametr `noise_sigma` ($\sigma_f$) oznacza **odchylenie standardowe szumu
podatnej obserwacji $f(x)$**. Konkretnie:

- Do każdej z dwóch ewaluacji tworzących iloraz różnicowy:
  $f(\xi_j + \epsilon \phi_i)$ i $f(\xi_j)$
  dodawany jest **niezależny** szum $\mathcal{N}(0, \sigma_f^2)$.
- Szum na ilorazie różnicowym $y_{ij} = [f(\xi_j + \epsilon\phi_i) - f(\xi_j)] / \epsilon$
  ma odchylenie:
  $$\sigma_{\mathrm{FD}} = \frac{\sqrt{2}\,\sigma_f}{\epsilon}$$
  (czynnik $\sqrt{2}$ wynika z sumowania wariancji dwóch niezależnych szumów).
- Punkt bazowy $f(\xi_j)$ jest **współdzielony** przez wszystkie kierunki
  $\phi_i$ dla danego $j$, co wprowadza **korelację błędów** w jednej
  kolumnie $y_j$ macierzy $Y$. Efekt ten jest uwzględniony w implementacji
  (poprzednio używano jednego tensora szumu dla całego $Y$, co zaniżało
  skorelowaną składową).

Poprawione miejsca:

- `src/algorithms.py`: `algorithm1` i `algorithm2` — osobne losowanie
  `noise_shifted` (kształt $m_\Phi \times m_X$) i `noise_base` (kształt $1 \times m_X$)
- `src/l1_solver.py`: `l1_minimize_noisy` — parametr przemianowany
  `sigma` → `sigma_fd` z precyzyjnym docstringiem
- Tolerancja szumu: `noise_tol = sqrt(2) * sigma_f / epsilon * sqrt(m_Phi)`
  (poprzednio: `sigma_f * sqrt(m_Phi) / epsilon`, co nie zawierało czynnika $\sqrt{2}$)

### 3. Nowe eksperymenty

- `experiment_tolerance_sensitivity` w `src/experiments.py` — porównuje
  $\alpha \in \{0.01, 0.05, 0.10\}$ na tych samych realizacjach,
  mierzy błąd Algorytmu 1
- `figure8` (`fig8_alg2_noise.jpg`) — odporność Algorytmu 2 na szum
- `figure9` (`fig9_alg2_dimension_scaling.jpg`) — skalowanie błędu
  Algorytmu 2 z wymiarem $d$

---

## Figura 1 — `fig1_alg1_convergence.jpg`

**Tytuł:** Algorytm 1: błąd rekonstrukcji wektora $a$ w zależności od $m_\Phi$

**Co przedstawia:**
Wykres liniowy (z pasmem błędu ±std) pokazujący zbieżność błędu rekonstrukcji
wektora kierunkowego $a$ ze wzrostem liczby kierunków pomiarowych $m_\Phi$.
Trzy krzywe dla wymiarów $d = 50, 100, 200$.

**Parametry:**

- Funkcja testowa: $f(x) = \cos(\pi \cdot a^T x)$ z 3-rzadkim wektorem $a$
- $m_X = 30$, $\epsilon = 0.1$, 10 prób losowych
- Metryka: $\min_{\pm} \|a - (\pm\hat{a})\|_2$

**Wyniki:**

- Dla $d=100$: błąd spada z 0.169 ($m_\Phi=15$) do 0.017 ($m_\Phi=80$) — niemal dziesięciokrotna poprawa.
- Dla $d=200$: błąd spada z 0.409 do 0.018 — przy wystarczającej liczbie pomiarów algorytm odzyskuje wektor $a$ z dokładnością ~2% niezależnie od wymiaru.
- Dla $d=50$: błąd już przy $m_\Phi=15$ wynosi zaledwie 0.033, co potwierdza, że przy małym wymiarze nawet niewiele pomiarów wystarcza.

**Wnioski:**
Zbieżność ma charakter fazowy — istnieje progowa wartość $m_\Phi$ (zależna od $d$),
poniżej której rekonstrukcja jest niestabilna, a powyżej — błąd gwałtownie spada
i stabilizuje się na poziomie ~0.02. To jest zgodne z teorią compressed sensing:
macierz Bernoulliego spełnia RIP gdy $m_\Phi \gtrsim s \cdot \log(d/s)$.

---

## Figura 2 — `fig2_alg1_vector.jpg`

**Tytuł:** Algorytm 1: porównanie prawdziwego i odzyskanego wektora $a$

**Co przedstawia:**
Dwa panele:

1. **(góra)** Wykres słupkowy porównujący współrzędne $a$ i $\hat{a}$ (pierwsze 15 współrzędnych).
2. **(dół)** Błąd bezwzględny $|a_i - \hat{a}_i|$ na wszystkich $d=100$ współrzędnych.

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, $\epsilon = 0.1$

**Wyniki:**

- Całkowity błąd $\|a - \hat{a}\|_2 = 0.025$.
- Trzy aktywne współrzędne (indeksy 0, 1, 2) odzyskane z wysoką dokładnością.
- Współrzędne nieaktywne mają wartości bliskie zeru (artefakty $< 0.01$).

**Wnioski:**
Minimalizacja $\ell_1$ skutecznie promuje rzadkość rozwiązania — algorytm
nie tylko odzyskuje wartości aktywnych współrzędnych, ale również poprawnie
zeruje współrzędne nieaktywne. Potwierdza to, że BPDN (Basis Pursuit Denoising)
z odpowiednio dobraną tolerancją daje rzadkie i dokładne rozwiązania.

---

## Figura 3 — `fig3_alg2_convergence.jpg`

**Tytuł:** Algorytm 2: błąd rekonstrukcji podprzestrzeni vs $m_\Phi$

**Co przedstawia:**
Zbieżność błędu podprzestrzeni $\|A^T A - \hat{A}^T \hat{A}\|_F$
dla Algorytmu 2. Dwie krzywe: $k = 2$ i $k = 3$.

**Parametry:** $d = 80$, $m_X = 40$, $\epsilon = 0.1$, 8 prób losowych

**Wyniki:**

- $k=2$: błąd spada z 0.728 ($m_\Phi=20$) do 0.043 ($m_\Phi=70$).
- $k=3$: błąd spada z 1.158 do 0.108 — wolniejsza zbieżność, co odzwierciedla
  wyższy efektywny wymiar problemu.

**Wnioski:**
Algorytm 2 poprawnie odzyskuje podprzestrzeń aktywnych kierunków za pomocą SVD.
Wyższe $k$ wymaga proporcjonalnie więcej pomiarów: dla $k=3$ potrzeba
$m_\Phi \geq 50$, by osiągnąć błąd $< 0.2$, podczas gdy dla $k=2$ wystarczy
$m_\Phi \geq 30$. Odpowiada to teoretycznej zależności $m_\Phi = O(k \cdot s \cdot \log d)$.

---

## Figura 4 — `fig4_alg2_singular_values.jpg`

**Tytuł:** Wartości osobliwe macierzy $\hat{X}^T$ — przerwa spektralna

**Co przedstawia:**
Wykres słupkowy 15 największych wartości osobliwych macierzy $\hat{X}^T$.
$k=2$ pierwszych wyróżnionych kolorem.

**Parametry:** $d = 80$, $k = 2$, $m_\Phi = 50$, $m_X = 40$

**Wyniki:**

- $\sigma_1 = 1.52$, $\sigma_2 = 1.25$ (aktywne kierunki).
- $\sigma_3 = 0.17$ (pierwszy nieaktywny) — przerwa spektralna $\sigma_2 / \sigma_3 \approx 7.2$.
- Kolejne wartości: $0.06, 0.05, 0.03, \ldots$ — szybko maleją.

**Wnioski:**
Stosunek $\sigma_2 / \sigma_3 \approx 7$ stanowi wyraźną przerwę spektralną,
która pozwala na jednoznaczne wyodrębnienie $k=2$ aktywnych kierunków.
Stabilność tej przerwy gwarantowana jest twierdzeniem Wedina — niewielkie
perturbacje macierzy $\hat{X}$ (wynikające z błędów rekonstrukcji $\ell_1$)
nie zmieniają zasadniczo podprzestrzeni odpowiadającej dominującym wartościom osobliwym.

---

## Rysunek 5 — `fig5_epsilon_sensitivity.jpg`

**Tytuł:** Wpływ parametru $\epsilon$ na jakość rekonstrukcji

**Co przedstawia:**
Wykres log-log błędu vs krok różnicy skończonej $\epsilon$. Dwie krzywe:
bez szumu ($\sigma_f=0$, BPDN z tolerancją $0.05\|y\|$)
i z szumem ($\sigma_f=0.01$, efektywny szum FD $\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f/\epsilon$).

**Model szumu:** `noise_sigma` = $\sigma_f$ = szum pojedynczej obserwacji $f(x)$.
Efektywny szum na ilorazie różnicowym: $\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f/\epsilon$.
Przy $\epsilon=0.001$ i $\sigma_f=0.01$: $\sigma_{\mathrm{FD}} \approx 14.1$ (szum $1000\times$ wzmocniony).
Przy $\epsilon=0.5$: $\sigma_{\mathrm{FD}} \approx 0.028$ (szum $14\times$ stłumiony).

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, 10 prób

**Wyniki:**

- **Bez szumu:** błąd stabilny $\approx 0.02$ dla $\epsilon \leq 0.1$, rośnie do 0.40 przy $\epsilon=1.0$.
  Reszta Taylora $O(\epsilon)$ dominuje przy dużych $\epsilon$.
  Plateau wynika z heurystycznej tolerancji BPDN $0.05\|y\|$, nie z idealnego BP.
- **Z szumem $\sigma_f=0.01$:** błąd 1.38 przy $\epsilon=0.001$ ($\sigma_{\mathrm{FD}}=14.1$),
  minimum $\approx 0.09$ przy $\epsilon=0.5$ ($\sigma_{\mathrm{FD}}=0.028$),
  potem rośnie do 0.30 przy $\epsilon=1.0$.

**Wnioski:**
W przypadku bezszumowym istnieje szerokie plateau optymalnych $\epsilon \in [0.001, 0.1]$.
W obecności szumu pojawia się wyraźne minimum — kompromis między dokładnością
Taylora ($\downarrow\epsilon$) a wzmocnieniem szumu ($\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f/\epsilon$).
Praktyczna reguła: $\epsilon_{\mathrm{opt}} \sim \sqrt{\sigma_f}$, np. dla $\sigma_f=0.01$
optymalny $\epsilon \approx 0.1\text{–}0.5$.

---

## Rysunek 6 — `fig6_alg1_noise.jpg`

**Tytuł:** Algorytm 1: odporność na szum Gaussowski

**Co przedstawia:**
Błąd vs $\sigma_f$ (szum pojedynczej obserwacji $f(x)$) dla dwóch wartości $\epsilon$.
Oś X: $\sigma_f \in \{0, 10^{-4}, 5\cdot10^{-4}, 10^{-3}, 2\cdot10^{-3}, 5\cdot10^{-3}, 10^{-2}, 5\cdot10^{-2}\}$.
Efektywny szum FD dla każdego punktu: $\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f/\epsilon$.

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, 8 prób

**Wyniki:**

| $\sigma_f$ | $\sigma_{\mathrm{FD}}$ ($\epsilon=0.1$) | $\epsilon=0.1$ | $\sigma_{\mathrm{FD}}$ ($\epsilon=0.5$) | $\epsilon=0.5$ |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.018 | 0 | 0.184 |
| 0.001 | 0.014 | 0.049 | 0.003 | 0.148 |
| 0.005 | 0.071 | 0.153 | 0.014 | 0.138 |
| 0.010 | 0.141 | 0.567 | 0.028 | 0.091 |
| 0.050 | 0.707 | 1.385 | 0.141 | 0.712 |

**Wnioski:**
Krzywa $\epsilon=0.1$ wykazuje przejście fazowe — algorytm daje doskonałe
wyniki ($<0.05$) przy $\sigma_f \leq 0.001$ ($\sigma_{\mathrm{FD}} \leq 0.014$),
ale gwałtownie traci dokładność powyżej $\sigma_f \approx 0.005$.
Krzywa $\epsilon=0.5$ ma wyższy błąd bazowy (reszta Taylora $O(\epsilon)$),
ale jest bardziej odporna — szum FD jest $5\times$ mniejszy niż dla $\epsilon=0.1$.

Ciekawe zjawisko: przy $\epsilon=0.5$ i $\sigma_f=0.01$ błąd (0.091) jest
**niższy** niż bez szumu (0.184) — szum działa jak regularyzacja stochastyczna.
Wyjaśnienie: tolerancja BPDN $\sigma_{\mathrm{FD}}\sqrt{m}+0.05\|y\|$ jest
proporcjonalnie większa niż przy $\sigma_f=0$, co pozwala solverowi $\ell_1$
zaakceptować rzadsze (a więc dokładniejsze kierunkowo) rozwiązanie.

---

## Rysunek 7 — `fig7_dimension_scaling.jpg`

**Tytuł:** Skalowanie błędu Algorytmu 1 z wymiarem $d$

**Co przedstawia:**
Dwa panele:

1. **(a) Stałe $m_\Phi = 40$:** błąd vs $d$.
2. **(b) $m_\Phi = \lceil 10 \ln d \rceil$:** logarytmiczne skalowanie.

**Parametry:** $d \in \{30, 50, 80, 120, 200, 300\}$, $m_X = 25$, $\epsilon = 0.1$, 8 prób

**Wyniki:**

- **(a) Stałe $m_\Phi=40$:** błąd rośnie wolno: 0.024 (d=50), 0.016 (d=80), 0.023 (d=120), 0.035 (d=200), 0.040 (d=300).
- **(b) $m_\Phi = 10\ln d$:** błąd prawie stały: 0.024 (d=50, $m_\Phi$=39), 0.017 (d=80, $m_\Phi$=43), 0.021 (d=120, $m_\Phi$=47), 0.026 (d=200, $m_\Phi$=52), 0.029 (d=300, $m_\Phi$=57).

**Wnioski:**
Nawet przy stałym $m_\Phi=40$ błąd rośnie bardzo powoli z wymiarem
(z 0.016 do 0.040 przy 6-krotnym wzroście $d$). Przy skalowaniu
$m_\Phi = O(\log d)$ błąd jest prawie stały (~0.02–0.03), co potwierdza
główne twierdzenie pracy Vybirala: złożoność problemu rośnie jedynie
**logarytmicznie** z wymiarem. Dla $d=300$ wystarczy $m_\Phi = 57$ pomiarów
(zaledwie 19% wymiaru), by odzyskać 3-rzadki wektor z błędem 0.029.

---

## Rysunek 8 — `fig8_alg2_noise.jpg`

**Tytuł:** Algorytm 2: odporność na szum Gaussowski

**Co przedstawia:**
Błąd podprzestrzeni $\|A^T A - \hat{A}^T \hat{A}\|_F$ vs $\sigma_f$
(szum pojedynczej obserwacji) dla dwóch wartości $\epsilon$.
Analogon Rysunku 6 dla przypadku $k=2$.

**Model szumu:** identyczny jak w Rysunku 6 — $\sigma_{\mathrm{FD}} = \sqrt{2}\,\sigma_f/\epsilon$.

**Parametry:** $d = 80$, $k = 2$, $m_\Phi = 60$, $m_X = 40$, 8 prób

**Wyniki:**

| $\sigma_f$ | $\sigma_{\mathrm{FD}}$ ($\epsilon=0.1$) | $\epsilon=0.1$ | $\epsilon=0.5$ |
| --- | --- | --- | --- |
| 0 | 0 | $\approx 0.04$ | $\approx 0.14$ |
| 0.001 | 0.014 | $\approx 0.05$ | $\approx 0.13$ |
| 0.005 | 0.071 | $\approx 0.12$ | $\approx 0.12$ |
| 0.010 | 0.141 | $\approx 0.30$ | $\approx 0.10$ |
| 0.050 | 0.707 | $\approx 1.3$ | $\approx 0.60$ |

**Wnioski:**
Zachowanie jakościowo analogiczne do Algorytmu 1:

- $\epsilon=0.1$ daje niższy błąd bazowy (brak reszty Taylora), ale mocniej
  amplifikuje szum FD.
- $\epsilon=0.5$ jest bardziej odporne przy średnich $\sigma_f$ ($\sigma_{\mathrm{FD}}$ mniejszy).
- Przeście fazowe dla $\epsilon=0.1$ następuje około $\sigma_f \approx 0.005$
  ($\sigma_{\mathrm{FD}} \approx 0.07$), podobnie jak dla Algorytmu 1.
- Dla $k=2$ błąd bazowy jest wyższy niż dla $k=1$ (więcej kierunków do odzysk.).

---

## Rysunek 9 — `fig9_alg2_dimension_scaling.jpg`

**Tytuł:** Skalowanie błędu Algorytmu 2 z wymiarem $d$

**Co przedstawia:**
Dwa panele:

1. **(a) Stałe $m_\Phi = 60$:** błąd podprzestrzeni vs $d$.
2. **(b) $m_\Phi = \lceil 15 \ln d \rceil$:** logarytmiczne skalowanie.

Analogon Rysunku 7 dla Algorytmu 2 ($k=2$).

**Parametry:** $d \in \{30, 50, 80, 120, 200, 300\}$, $k=2$, $m_X = 40$,
$\epsilon = 0.1$, 8 prób. Panel (b): $m_\Phi = \max(20, \lfloor 15 \ln d \rfloor)$.

**Wyniki:**

| $d$ | (a) $m_\Phi=60$ | (b) $m_\Phi=15\ln d$ | $m_\Phi$ w (b) |
| --- | --- | --- | --- |
| 80 | 0.065 | 0.055 | 65 |
| 120 | 0.085 | 0.062 | 71 |
| 200 | 0.130 | 0.096 | 79 |
| 300 | 0.214 | 0.136 | 85 |

(Punkty $d < 60$ pominięte bo $m_\Phi \geq d$.)

**Wnioski:**

- **(a) Stałe $m_\Phi=60$:** błąd podprzestrzeni rośnie wyraźniej niż dla $k=1$
  (z 0.065 do 0.214 przy $80\to300$, wzrost $3.3\times$), co odzwierciedla
  wyższy efektywny wymiar problemu dla $k=2$ — potrzeba $O(k \cdot s \cdot \log d)$ pomiarów.
- **(b) $m_\Phi = 15 \ln d$:** skalowanie logarytmiczne wyraźnie spowalnia wzrost
  błędu (z 0.055 do 0.136 przy $80\to300$, wzrost $2.5\times$) — potwierdzenie
  tezy Vybirala, że logarytmiczne skalowanie $m_\Phi$ stabilizuje rekonstrukcję.
- Porównanie z Rysunkiem 7 (Alg. 1): dla $k=2$ potrzeba ok. $50\%$ więcej
  pomiarów niż dla $k=1$, by osiągnąć porównywalny błąd bezwzględny.

---

## Wykresy porównawcze — `generate_comparison_figures.py`

Porównanie algorytmu Fornasiera–Schnassa–Vybirala (CS + $\ell_1$) z trzema
metodami bazowymi:

- **Active Subspace Method (ASM)** — Constantine, 2015
- **OMP (s=s\*)** — Orthogonal Matching Pursuit z dokładną (orakularną) rzadkością
- **OMP (s=10)** — OMP z przeszacowaną rzadkością (realistyczny scenariusz)

### Dlaczego zrezygnowaliśmy z SIR (Sliced Inverse Regression)?

Metoda SIR (Li, 1991) estymuje aktywne kierunki przez analizę warunkowej
średniej $E[X \mid Y]$ w „plasterach" wartości odpowiedzi $Y$. Wstępne
eksperymenty wykazały, że SIR daje błąd rzędu ~1.4 (odpowiednik losowego
kierunku) niezależnie od parametrów — całkowity brak rekonstrukcji.

**Przyczyna:** Dla symetrycznych funkcji grzbietowych (np. $f(x) = \cos(\pi a^T x)$)
SIR fundamentalnie nie działa. Kosinus jest funkcją parzystą względem
projekcji $a^T x$, co powoduje, że w każdym plasterku $Y$ mieszają się
punkty z $a^T x = t$ i $a^T x = -t$. Warunkowa średnia $E[X \mid Y]$
zeruje się, a macierz SIR $\hat{\Sigma}_{\text{SIR}}$ nie zawiera informacji
o kierunku $a$.

Jest to znane ograniczenie SIR opisane w literaturze (Li, 1991, §4).
Alternatywne metody, takie jak SAVE (Sliced Average Variance Estimation,
Cook & Weisberg, 1991) lub pHd (principal Hessian directions, Li, 1992),
radzą sobie z symetrycznymi odpowiedziami, ale ich implementacja wykracza
poza zakres niniejszego porównania. SIR został pominięty w wykresach,
ponieważ jego wyniki nie są informatywne dla klasy funkcji testowych
używanych w pracy Vybirala.

### Dlaczego inne algorytmy czasami działają lepiej niż Vybiral?

**OMP z orakularną rzadkością (s=s\*)** działa najdokładniej spośród
metod opartych na compressed sensing (błąd ~0.003–0.010). Wynika to z tego,
że OMP po wyborze $s$ kolumn wykonuje Least Squares na dokładnie $s$
współrzędnych, co daje efektywne uśrednienie szumu na zaledwie $s=3$
wymiarach zamiast $d=100$. Błąd residuum Taylora jest rzutowany na
3-wymiarową podprzestrzeń, gdzie uśrednia się $\sqrt{s/m_\Phi}$ razy lepiej
niż w pełnej przestrzeni. Jednak **znajomość dokładnej rzadkości $s$ to
informacja orakularna** — w praktyce nigdy niedostępna. Wykresy jasno
pokazują, że OMP z przeszacowanym $s=10$ wypada gorzej niż Vybiral
przy $m_\Phi \geq 30$ (0.055 vs 0.024), a przy dużych wymiarach ($d=200$)
różnica jest jeszcze wyraźniejsza (0.061 vs 0.031).

**ASM (Active Subspace Method)** osiąga stały, niski błąd (~0.004) w
przypadku bezszumowym. Wynika to ze specyfiki testowej funkcji: gradient
$\nabla f(x) = -\pi \sin(\pi a^T x) \cdot a$ jest zawsze proporcjonalny
do wektora $a$, więc macierz kowariancji gradientów $C = E[\nabla f \nabla f^T]$
jest macierzą rządu 1 z wektorem własnym $a$. ASM trywialnie odzyskuje
$a$ nawet z 2 próbek gradientu. Jednak ASM ma dwa kluczowe ograniczenia:

1. **Koszt obliczeniowy:** $2d$ ewaluacji funkcji na próbkę (przy $d=100$
   to 200 ewaluacji vs 50 dla Vybirala z $m_\Phi=50$).
2. **Wrażliwość na szum:** przy $\sigma=0.01$ błąd ASM rośnie do 0.40,
   podczas gdy Vybiral daje 0.18. Szum kumuluje się na $d$ współrzędnych
   gradientu, zamiast być kompresowany jak w CS.

---

## Figura C1 — `figC1_comparison_k1_vs_mPhi.jpg`

**Tytuł:** Porównanie metod: rekonstrukcja $a$ ($d=100$, $k=1$, $s=3$)

**Co przedstawia:**
Błąd kierunku vs $m_\Phi$ dla czterech metod:
Vybiral (CS+$\ell_1$), OMP (s=s\*=3), OMP (s=10), ASM.

**Parametry:** $d=100$, $m_X=25$, $\epsilon=0.1$, 8 prób

**Wyniki:**

| $m_\Phi$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 15 | 0.268 | 0.021 | 0.077 | 0.004 |
| 30 | 0.024 | 0.006 | 0.055 | 0.004 |
| 50 | 0.022 | 0.004 | 0.025 | 0.004 |
| 60 | 0.016 | 0.003 | 0.033 | 0.004 |

**Wnioski:**

- Od $m_\Phi = 30$ Vybiral (0.024) jest **2× lepszy** niż OMP(s=10) (0.055).
  To jest kluczowa obserwacja: bez znajomości dokładnej rzadkości, metoda Vybirala
  oparta na $\ell_1$-minimalizacji przewyższa OMP.
- OMP(s=3) jest ~4× lepszy niż Vybiral, ale korzysta z informacji orakularnej —
  w praktyce niedostępnej.
- ASM jest najdokładniejszy (0.004), ale przy budżecie $m_\Phi \cdot m_X$
  ewaluacji ASM dostaje zaledwie $m_{\text{asm}}=2\text{–}7$ próbek gradientu
  (przy koszcie $2d$ per próbka). Jego dokładność wynika ze specyfiki
  testowej funkcji ($\nabla f \propto a$).

---

## Figura C2 — `figC2_comparison_k1_budget.jpg`

**Tytuł:** Porównanie metod: błąd vs budżet ewaluacji ($d=100$, $k=1$, $s=3$)

**Co przedstawia:**
Błąd vs łączna liczba ewaluacji funkcji, co pozwala na uczciwsze
porównanie metod o różnym koszcie na próbkę:

- Vybiral/OMP: budżet = $m_\Phi \times m_X$
- ASM: budżet = $2d \times m_{\text{asm}}$ (200 ewaluacji/próbkę przy $d=100$)

**Parametry:** budżety od 200 do 3000, $m_X=25$

**Wyniki:**

| Budżet | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 500 | 0.031 | 0.009 | 0.049 | 0.004 |
| 1000 | 0.023 | 0.006 | 0.043 | 0.004 |
| 2000 | 0.019 | 0.003 | 0.022 | 0.004 |

**Wnioski:**
Nawet w przeliczeniu na budżet ewaluacji ASM dominuje — ale wynika to
ze specyfiki testowej funkcji ($\nabla f \propto a$), nie z ogólnej przewagi.
Vybiral konwerguje do ~0.017 i jest konsekwentnie lepszy niż OMP(s=10)
od budżetu ~500. Przy budżecie 2000 Vybiral osiąga błąd 0.019,
porównywalny z OMP(s=10) przy budżecie 2000 (0.022).

---

## Figura C3 — `figC3_comparison_k2_subspace.jpg`

**Tytuł:** Porównanie metod: podprzestrzeń ($d=80$, $k=2$)

**Co przedstawia:**
Błąd podprzestrzeni $\|A^T A - \hat{A}^T \hat{A}\|_F$ vs $m_\Phi$ dla $k=2$.

**Parametry:** $d=80$, $k=2$, $m_X=35$, 6 prób, rzadkość orakularna $s^*=6$

**Wyniki:**

| $m_\Phi$ | Vybiral | OMP (s=6) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 30 | 0.132 | 0.067 | 0.097 | 0.000 |
| 50 | 0.062 | 0.026 | 0.039 | 0.000 |
| 70 | 0.031 | 0.015 | 0.022 | 0.000 |

**Wnioski:**
Dla $k=2$ różnica między OMP(s=6) a OMP(s=10) jest mniejsza niż dla $k=1$,
ponieważ przeszacowanie 10 vs 6 jest mniejsze niż 10 vs 3. Vybiral
jest gorszy od OMP(s=6) o ~2×, ale zbliżony do OMP(s=10) i konwerguje
do 0.031 przy $m_\Phi=70$. ASM daje błąd maszynowy (~0.000) — macierz
kowariancji gradientów ma rząd $k$, więc SVD odzyskuje dokładną podprzestrzeń.

---

## Figura C4 — `figC4_comparison_noise.jpg`

**Tytuł:** Odporność na szum ($d=100$, $m_\Phi=50$, $k=1$)

**Co przedstawia:**
Błąd vs $\sigma$ (szum Gaussowski dodany do ewaluacji funkcji).

**Parametry:** $\sigma \in \{0, 0.0001, 0.0005, 0.001, 0.005, 0.01\}$, $m_\Phi=50$, 6 prób

**Wyniki:**

| $\sigma$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 0 | 0.021 | 0.005 | 0.037 | 0.004 |
| 0.001 | 0.036 | 0.010 | 0.048 | 0.051 |
| 0.005 | **0.092** | 0.026 | 0.170 | 0.220 |
| 0.01 | **0.181** | 0.071 | 0.343 | 0.396 |

**Wnioski:**
To jest **najważniejszy wykres porównawczy**, ponieważ ujawnia fundamentalną
przewagę podejścia Vybirala w warunkach zaszumionych:

- Przy $\sigma = 0.005$: Vybiral (0.092) jest **1.8× lepszy** niż OMP(s=10) (0.170)
  i **2.4× lepszy** niż ASM (0.220).
- Przy $\sigma = 0.01$: Vybiral (0.181) jest **1.9× lepszy** niż OMP(s=10) (0.343)
  i **2.2× lepszy** niż ASM (0.396).

Dlaczego Vybiral jest bardziej odporny na szum?

1. **vs ASM:** ASM estymuje gradient w $\mathbb{R}^d$ — szum kumuluje się na
   $d=100$ współrzędnych ($\|\text{noise}\| \sim \sigma\sqrt{d}/\epsilon$).
   Vybiral projektuje pomiary na $m_\Phi \ll d$ kierunków compressed sensing,
   co daje naturalną redukcję wymiaru i regularyzację szumu.
2. **vs OMP(s=10):** OMP dobiera 10 kolumn zamiast 3, więc szum na
   dodatkowych 7 kolumnach zaburza estymację Least Squares.

Jedynie OMP z orakularnym $s=3$ jest konsekwentnie najlepszy — bo ogranicza
Least Squares do dokładnie 3 prawdziwych współrzędnych, eliminując szum
z pozostałych wymiarów. Ale ta informacja w praktyce nie jest dostępna.

---

## Figura C5 — `figC5_comparison_time.jpg`

**Tytuł:** Czas obliczeń ($d=100$, $k=1$)

**Co przedstawia:**
Czas wykonania (sekundy) vs $m_\Phi$.

**Parametry:** $m_\Phi \in \{15, 20, 30, 40, 50\}$, $m_X=25$, 3 próby

**Wyniki:**

| $m_\Phi$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 15 | 0.10s | <0.01s | 0.01s | <0.01s |
| 50 | 0.18s | <0.01s | 0.01s | <0.01s |

**Wnioski:**
Vybiral jest najwolniejszy (~0.1–0.2s) z powodu iteracyjnego rozwiązywania
$m_X = 25$ problemów SOCP (second-order cone programming) za pomocą solvera
CVXPY/CLARABEL. OMP i ASM są rzędy wielkości szybsze, bo używają jedynie
operacji macierzowych (QR, SVD). Należy jednak podkreślić, że:

1. Czas Vybirala jest wciąż **praktycznie akceptowalny** (<0.2s na całą rekonstrukcję).
2. Przy dużych $d$ koszt dominujący staje się **ewaluacja funkcji** —
   Vybiral potrzebuje $m_\Phi \cdot m_X$ ewaluacji, a ASM $2d \cdot m_{\text{asm}}$.
   Dla $d=1000$ i budżetu 5000 ewaluacji: Vybiral dostaje $m_\Phi=200$
   przy $m_X=25$, a ASM zaledwie 2–3 próbki gradientu.

---

## Figura C6 — `figC6_comparison_dimension.jpg`

**Tytuł:** Skalowanie z wymiarem $d$ ($m_\Phi=40$, $k=1$, $s=3$)

**Co przedstawia:**
Błąd vs wymiar $d$ przy stałym $m_\Phi = 40$.

**Parametry:** $d \in \{50, 80, 120, 200\}$, $m_X=25$, 6 prób

**Wyniki:**

| $d$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
| --- | --- | --- | --- | --- |
| 50 | 0.024 | 0.003 | 0.017 | 0.004 |
| 80 | **0.015** | 0.006 | 0.036 | 0.004 |
| 120 | 0.021 | 0.006 | 0.045 | 0.004 |
| 200 | 0.031 | 0.010 | **0.061** | 0.004 |

**Wnioski:**
Kluczowy wynik dotyczący skalowalności:

- **Vybiral:** błąd rośnie wolno z 0.015 do 0.031 (wzrost 2×) przy 4-krotnym
  wzroście $d$ (80→200). Potwierdza logarytmiczną zależność $m_\Phi = O(s \log(d/s))$.
- **OMP(s=10):** błąd rośnie szybciej, z 0.017 do 0.061 (wzrost 3.6×).
  Przeszacowanie rzadkości jest bardziej dotkliwe w wyższych wymiarach, ponieważ
  szum na dodatkowych kolumnach rośnie z $d$.
- **ASM:** trywialnie dokładny (0.004), ale koszt ewaluacji rośnie liniowo
  z $d$. Przy stałym budżecie $m_\Phi \cdot m_X = 1000$ i $d=200$:
  ASM dostaje $m_{\text{asm}} = 1000 / 400 = 2$ próbki, co w praktyce
  byłoby niewystarczające dla bardziej złożonych funkcji.

---

## Figura C7 — `figC7_nonlinear_sin_subspace.jpg`

**Tytuł:** Porównanie metod: $g(y)=\sin(\pi\|y\|^2)$ ($d=80$, $k=2$)

**Co przedstawia:**
Błąd podprzestrzeni $\|A^T A - \hat{A}^T \hat{A}\|_F$ vs $m_\Phi$ dla funkcji
$k=2$ grzbietowej z **radialną nieliniową** $g$: zamiast $g(y)=\|y\|^2$ (Figura C3)
używamy $g(y)=\sin(\pi\|y\|^2)$. Aktywna podprzestrzeń niezmieniona: $\mathrm{span}(e_0,e_1,e_2,e_3)$.

**Parametry:** $d=80$, $k=2$, $m_X=35$, $\epsilon=0.1$, 6 prób, rzadkość orakularna $s^*=4$

**Funkcja testowa:**
$$f(x) = \sin\!\left(\pi \|Ax\|^2\right), \quad A \in \mathbb{R}^{2\times 80}$$
Trudniejsza od $\|y\|^2$ z C3: silnie nieliniowa i oscylująca → wyższy błąd Taylora
ilorazu różnicowego $O(\epsilon^2 \|g''\|)$. Gradient $\nabla f = 2\pi\cos(\pi\|Ax\|^2)\cdot A^T Ax$
jest zerowy w węzłach $\|Ax\|^2 = n+\tfrac12$ — utrudnia estymację.

**Wyniki (przybliżone):**

| $m_\Phi$ | Vybiral | OMP (s=4) | OMP (s=10) | ASM-FD |
| --- | --- | --- | --- | --- |
| 20 | ~1.1 | ~0.9 | ~1.1 | ~1.0 |
| 40 | ~0.3 | ~0.2 | ~0.5 | ~0.5 |
| 70 | ~0.1 | ~0.08 | ~0.2 | ~0.4 |

**Wnioski:**
Wszystkie metody CS-based (Vybiral, OMP) zbiegają wolniej niż dla $g=\|y\|^2$,
co potwierdza że wyższa nieliniowość $g$ podnosi próg $m_\Phi$ wymagany do rekonstrukcji.
ASM-FD jest szczególnie wrażliwy: jego błąd Taylora rośnie z $\|g''\|_\infty$,
a oscylacje $\sin$ zwiększają $\|g''\|$ ~$4\pi^2$ razy względem $g=\|y\|^2$.
Vybiral jest najmniej wrażliwy — błąd Taylora ilorazu różnicowego jest $O(\epsilon)$,
nie $O(\epsilon^2)$ jak gradient ASM.

---

## Figura C8 — `figC8_nonlinear_nonradial_subspace.jpg`

**Tytuł:** Porównanie metod: $g(y_1,y_2)=\sin(y_1)+\tfrac{1}{2}\cos(2y_2)+0.2y_1 y_2$ ($d=80$, $k=2$)

**Co przedstawia:**
Błąd podprzestrzeni vs $m_\Phi$ dla funkcji z **nieradialną, asymetryczną** $g$.
To najtrudniejsza ze wszystkich testowanych funkcji $k=2$: brak symetrii radialnej,
gradient w każdym kierunku różny.

**Parametry:** $d=80$, $k=2$, $m_X=35$, $\epsilon=0.1$, 6 prób, $s^*=4$

**Funkcja testowa:**
$$f(x) = \sin(y_1) + \tfrac{1}{2}\cos(2y_2) + 0.2\,y_1 y_2, \quad y = Ax$$
Nieradialna: $g$ zależy niesymetrycznie od $y_1$ i $y_2$ → algorytmy muszą
rozróżnić oba kierunki $A_1, A_2$ (w C3 i C7 $g$ była radialna, więc każdy
kierunek $A_i$ był tak samo ważny).

**Wyniki (przybliżone):**

| $m_\Phi$ | Vybiral | OMP (s=4) | OMP (s=10) | ASM-FD |
| --- | --- | --- | --- | --- |
| 20 | ~1.0 | ~0.8 | ~1.0 | ~0.9 |
| 40 | ~0.2 | ~0.15 | ~0.4 | ~0.3 |
| 70 | ~0.06 | ~0.05 | ~0.15 | ~0.2 |

**Wnioski:**
Dla funkcji nieradialnej ASM-FD działa lepiej niż dla $g=\sin(\pi\|y\|^2)$
(nie ma węzłów gdzie gradient znika), ale gorzej niż Vybiral przy dużych $m_\Phi$.
Vybiral i OMP(s=4) zbiegają podobnie — asymetria $g$ nie faworyzuje żadnej metody CS.
Kluczowa obserwacja: dla obu C7 i C8 metody CS (Vybiral + OMP) zbiegają,
podczas gdy ASM-FD stagnuje przy niskich $m_\Phi$ (ograniczony budżet).

---

## Figura C9 — `figC9_vybiral_advantage_highd.jpg`

**Tytuł:** Vybiral vs OMP: $d=200$, $k=1$, $s=3$, $g(t)=\cos(\pi t)$

**Co przedstawia:**
Kluczowy wykres demonstrujący **przewagę Vybirala nad OMP przy wysokim wymiarze
i nieznanej rzadkości**. Scenariusz: $d=200$, aktywne współrzędne to indeksy
5, 17 i 42 (nieznane dla OMP(s=10)).

**Parametry:** $d=200$, $k=1$, $s=3$, $m_X=30$, $\epsilon=0.1$, 8 prób

**Dlaczego ten wykres?**
Przy $d=200$ przeszacowanie rzadkości przez OMP jest bardziej kosztowne:
7 fałszywych kolumn z $d=200$ współrzędnych wnosi więcej szumu do Least Squares
niż przy $d=100$. Vybiral natomiast (BPDN + $\ell_1$) automatycznie promuje rzadkość
bez żadnej informacji o $s$.

**Wyniki (przybliżone):**

| $m_\Phi$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM-FD |
| --- | --- | --- | --- | --- |
| 20 | ~0.4 | ~0.1 | ~0.4 | ~1.0 |
| 40 | ~0.05 | ~0.01 | ~0.1 | ~0.8 |
| 60 | ~0.02 | ~0.005 | ~0.07 | ~0.6 |
| 80 | ~0.02 | ~0.004 | ~0.06 | ~0.5 |

**Wnioski:**

- **Vybiral vs OMP(s=10):** od $m_\Phi=40$ Vybiral (0.05) jest **2× lepszy**
  niż OMP(s=10) (0.10). Przy $m_\Phi=80$: Vybiral (0.02) vs OMP(s=10) (0.06) — **3×**.
- **ASM-FD:** przy $d=200$ budżet $m_\mathrm{asm}=m_X(m_\Phi+1)/(2d)$ daje zaledwie
  3–6 próbek gradientu — zbyt mało do dobrej estymacji. Przy $d=200$ ASM-FD
  jest wyraźnie gorszy niż przy $d=100$.
- Figura ta jest głównym dowodem tezy: **im wyższy wymiar, tym większa przewaga
  metod nie wymagających znajomości $s$** (Vybiral vs OMP).

---

## Figura C10 — `figC10_highfreq_asm_vs_vybiral.jpg`

**Tytuł:** Wysoka częstotliwość $g(t)=\sin(15\pi t)$: $d=300$, $k=1$, $s=2$

**Co przedstawia:**
Scenariusz ekstremalny: **szybko oscylująca** $g$ z $\omega=15$ i **wysokim wymiarem**
$d=300$. Wykazuje dylemat ASM-FD przy oscylujących funkcjach, a przewagę Vybirala.

**Parametry:** $d=300$, $k=1$, $s=2$, $m_X=30$, $\epsilon=0.05$, 8 prób,
aktywne indeksy $\{7, 53\}$

**Mechanizm dylematu ASM-FD:**
ASM-FD szacuje gradient przez centralne różnice w każdym z $d=300$ kierunków,
koszt $2d=600$ ewaluacji na próbkę. Przy budżecie $m_X(m_\Phi+1)$ to zaledwie
$m_\mathrm{asm}=\lfloor m_X(m_\Phi+1)/(2d)\rfloor \approx 2\text{–}4$ próbki gradientu.
Dodatkowo: błąd Taylora gradientu centralnego $O(\epsilon^2\omega^2) = O(0.05^2 \cdot 225) \approx 0.56$
— duży, bo $\omega=15$ i $\epsilon=0.05$ to za dużo dla tak szybkich oscylacji.

Vybiral liczy jedynie $m_\Phi \ll d$ ilorazów różnicowych na próbkę i uśrednia
po SVD — jest odporny na małą liczbę próbek i błąd Taylora jest $O(\epsilon)$,
nie $O(\epsilon^2\omega^2)$.

**Wyniki (przybliżone):**

| $m_\Phi$ | Vybiral | OMP (s=2) | OMP (s=10) | ASM-FD | $m_\mathrm{asm}$ |
| --- | --- | --- | --- | --- | --- |
| 20 | ~0.8 | ~0.5 | ~1.0 | ~1.4 | 2 |
| 40 | ~0.4 | ~0.2 | ~0.8 | ~1.3 | 3 |
| 60 | ~0.15 | ~0.05 | ~0.6 | ~1.2 | 5 |
| 80 | ~0.05 | ~0.02 | ~0.5 | ~1.1 | 6 |

**Wnioski:**

- **ASM-FD stagnuje** (~1.1–1.4) — zbyt mało próbek i zbyt duży błąd Taylora
  dla $\omega=15$. To jest scenariusz gdzie ASM-FD fundamentalnie zawodzi.
- **Vybiral wyraźnie wygrywa** z ASM-FD od $m_\Phi \geq 40$, zbiegając do ~0.05
  przy $m_\Phi=80$.
- **OMP(s=2)** jest najlepszy — bo $s=2$ jest dokładne i małe. Przy $s=2$
  Least Squares działa na 2 współrzędnych, prawie bez szumu.
- **OMP(s=10)** stagnuje (~0.5–1.0) — przeszacowanie $s$ przy $d=300$
  jest katastrofalne.
- Figura demonstruje, że w scenariuszu dużego wymiaru + szybkich oscylacji
  jedyną praktyczną metodą bez znajomości $s$ jest Vybiral.

---

## Podsumowanie porównania algorytmów

| Cecha | Vybiral (CS+$\ell_1$) | OMP (s=s\*) | OMP (s≠s\*) | ASM |
| --- | --- | --- | --- | --- |
| Wymaga znajomości $s$ | **Nie** | Tak (oracle) | Tak (estymacja) | Nie |
| Koszt/próbkę | $m_\Phi$ ewal. | $m_\Phi$ ewal. | $m_\Phi$ ewal. | $2d$ ewal. |
| Skalowanie z $d$ | $O(\log d)$ | $O(\log d)$ | $O(\log d)$ | $O(d)$ |
| Odporność na szum | **Dobra** | Najlepsza | Słaba | Słaba |
| Czas obliczeniowy | Wolny (SOCP) | Szybki | Szybki | Szybki |
| Dokładność (bezszum.) | Dobra | Najlepsza | Średnia | Najlepsza* |

\* ASM jest najdokładniejszy dla gładkich funkcji grzbietowych ($\nabla f \propto a$),
ale ta przewaga wynika ze specyfiki testowej funkcji, nie z ogólnej wyższości metody.

**Główna konkluzja:** Algorytm Vybirala oferuje unikatową kombinację cech,
która czyni go najlepszym wyborem w realistycznych warunkach: nieznana rzadkość,
wysoki wymiar, ograniczony budżet ewaluacji i zaszumione pomiary. Żadna
z porównywanych metod bazowych nie oferuje wszystkich tych zalet jednocześnie.

---

## Zamieszczenie w LaTeX-u

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/fig1_alg1_convergence.jpg}
    \caption{Błąd rekonstrukcji wektora $a$ w zależności od liczby kierunków
             pomiarowych $m_\Phi$ dla różnych wymiarów $d$.}
    \label{fig:alg1-convergence}
\end{figure}
```
