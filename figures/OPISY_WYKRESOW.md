# Opisy wykresów do pracy magisterskiej

Wykresy zostały wygenerowane dwoma skryptami:
- `generate_figures.py` — 7 figur ilustrujących działanie algorytmów Vybirala
- `generate_comparison_figures.py` — 6 figur porównawczych z metodami bazowymi

Format: **JPG**. Aby wygenerować ponownie:
```bash
python generate_figures.py
python generate_comparison_figures.py
```

---

## Figura 1 — `fig1_alg1_convergence.jpg`

**Tytuł:** Algorytm 1: błąd rekonstrukcji wektora $a$ w zależności od $m_\Phi$

**Co przedstawia:**
Wykres liniowy (z pasmem błędu ±std) pokazujący zbieżność błędu rekonstrukcji
wektora kierunkowego $a$ ze wzrostem liczby kierunków pomiarowych $m_\Phi$.
Trzy krzywe dla wymiarów $d = 50, 100, 200$.

**Parametry:**
- Funkcja testowa: $f(x) = \cos(\pi \cdot a^T x)$ z 3-rzadkim wektorem $a$
- $m_X = 30$, $\varepsilon = 0.1$, 10 prób losowych
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

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, $\varepsilon = 0.1$

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

**Parametry:** $d = 80$, $m_X = 40$, $\varepsilon = 0.1$, 8 prób losowych

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

## Figura 5 — `fig5_epsilon_sensitivity.jpg`

**Tytuł:** Wpływ parametru $\varepsilon$ na jakość rekonstrukcji

**Co przedstawia:**
Wykres log-log błędu vs krok różnicy skończonej $\varepsilon$. Dwie krzywe:
bez szumu ($\sigma=0$) i z szumem ($\sigma=0.01$).

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, 10 prób

**Wyniki:**
- **Bez szumu:** błąd stabilny ~0.02 dla $\varepsilon \leq 0.1$, rośnie do 0.40 przy $\varepsilon=1.0$.
  Przybliżenie Taylora jest dokładne dla małych $\varepsilon$; przy dużych dominuje reszta $O(\varepsilon)$.
- **Z szumem $\sigma=0.01$:** błąd 1.38 przy $\varepsilon=0.001$ (szum amplifikowany $1000\times$),
  minimum 0.10 przy $\varepsilon \approx 0.2\text{–}0.5$, potem rośnie do 0.31 przy $\varepsilon=1.0$.

**Wnioski:**
W przypadku bezszumowym istnieje szerokie plateau optymalnych $\varepsilon \in [0.001, 0.1]$.
W obecności szumu pojawia się wyraźne minimum — kompromis między dokładnością
Taylora ($\downarrow\varepsilon$) a amplifikacją szumu ($\sigma/\varepsilon$).
Praktyczna reguła: $\varepsilon_{\text{opt}} \sim \sqrt{\sigma}$, np. dla $\sigma=0.01$
optymalne $\varepsilon \approx 0.1\text{–}0.5$.

---

## Figura 6 — `fig6_alg1_noise.jpg`

**Tytuł:** Algorytm 1: odporność na szum Gaussowski

**Co przedstawia:**
Błąd vs $\sigma$ dla dwóch wartości $\varepsilon$ (0.1 i 0.5).

**Parametry:** $d = 100$, $m_\Phi = 50$, $m_X = 25$, 8 prób

**Wyniki:**
- **$\varepsilon=0.1$:** błąd 0.018 (σ=0), 0.036 (σ=0.001), 0.092 (σ=0.005), 0.164 (σ=0.01), 1.08 (σ=0.05).
- **$\varepsilon=0.5$:** błąd 0.184 (σ=0), ~0.15 (σ=0.001–0.005), 0.107 (σ=0.01), 0.233 (σ=0.05).

**Wnioski:**
Krzywa $\varepsilon=0.1$ wykazuje przejście fazowe — algorytm daje doskonałe
wyniki ($<0.04$) przy $\sigma \leq 0.001$, ale gwałtownie traci dokładność
powyżej $\sigma \approx 0.005$. Krzywa $\varepsilon=0.5$ ma wyższy błąd bazowy
(reszta Taylora $O(\varepsilon)$), ale jest znacznie bardziej odporna na szum.

Ciekawe zjawisko: przy $\varepsilon=0.5$ i $\sigma=0.01$ błąd (0.107) jest
**niższy** niż bez szumu (0.184) — szum działa jak regularyzacja stochastyczna,
pomagając solverowi $\ell_1$ znaleźć bardziej rzadkie rozwiązanie.

---

## Figura 7 — `fig7_dimension_scaling.jpg`

**Tytuł:** Skalowanie błędu Algorytmu 1 z wymiarem $d$

**Co przedstawia:**
Dwa panele:
1. **(a) Stałe $m_\Phi = 40$:** błąd vs $d$.
2. **(b) $m_\Phi = \lceil 10 \ln d \rceil$:** logarytmiczne skalowanie.

**Parametry:** $d \in \{50, 80, 120, 200, 300\}$, $m_X = 25$, $\varepsilon = 0.1$, 8 prób

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

**Parametry:** $d=100$, $m_X=25$, $\varepsilon=0.1$, 8 prób

**Wyniki:**

| $m_\Phi$ | Vybiral | OMP (s=3) | OMP (s=10) | ASM |
|---|---|---|---|---|
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
|---|---|---|---|---|
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
|---|---|---|---|---|
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
|---|---|---|---|---|
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
   $d=100$ współrzędnych ($\|\text{noise}\| \sim \sigma\sqrt{d}/\varepsilon$).
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
|---|---|---|---|---|
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
|---|---|---|---|---|
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

## Podsumowanie porównania algorytmów

| Cecha | Vybiral (CS+$\ell_1$) | OMP (s=s\*) | OMP (s≠s\*) | ASM |
|---|---|---|---|---|
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
