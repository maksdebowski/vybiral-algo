# Opisy wykresów do pracy magisterskiej

Wszystkie wykresy zostały wygenerowane skryptem `generate_figures.py`.  
Format: **PDF** (grafika wektorowa, idealna do LaTeX-a).  
Aby wygenerować ponownie: `python generate_figures.py`

---

## Figura 1 — `fig1_alg1_convergence.pdf`

**Tytuł:** Algorytm 1: błąd rekonstrukcji wektora $a$ w zależności od $m_\Phi$

**Co przedstawia:**  
Wykres liniowy (z pasmem błędu ±std) pokazujący, jak maleje błąd rekonstrukcji
wektora kierunkowego $a$ wraz ze wzrostem liczby kierunków pomiarowych $m_\Phi$.
Trzy krzywe odpowiadają różnym wymiarom przestrzeni: $d = 50, 100, 200$.

**Parametry eksperymentu:**

- Funkcja testowa: $f(x) = \cos(\pi \cdot a^T x)$ z 3-rzadkim wektorem $a$
- $m_X = 30$ punktów próbkowania na sferze
- $\varepsilon = 0.1$ (krok różnicy skończonej)
- 10 prób losowych dla każdej konfiguracji
- Metryka: $\min_{\pm} \|a - (\pm\hat{a})\|_2$ (uwzgl. niejednoznaczność znaku)

**Oczekiwany wynik / interpretacja:**  
Błąd maleje ze wzrostem $m_\Phi$ (więcej pomiarów compressed sensing → lepsza
rekonstrukcja). Przy stałym $m_\Phi$ wyższy wymiar $d$ daje nieco lepsze wyniki,
bo macierz Bernoulliego ma lepsze własności RIP (Restricted Isometry Property)
gdy $m_\Phi/d$ jest mniejsze, a wektor $a$ ma stałą rzadkość $s=3$.

**Powiązanie z teorią:**  
Twierdzenie 3.1 (Vybiral et al.) — błąd $\nu_1$ maleje jak
$O\left(\frac{s^{1/q}}{(m_\Phi / \log(d/m_\Phi))^{1/2-1/q}}\right)$.

**Sugerowana sekcja w pracy:** 3.3 (Weryfikacja dokładności rekonstrukcji)

---

## Figura 2 — `fig2_alg1_vector.pdf`

**Tytuł:** Algorytm 1: porównanie prawdziwego i odzyskanego wektora $a$

**Co przedstawia:**  
Dwa panele:

1. **(góra)** Wykres słupkowy porównujący wartości współrzędnych prawdziwego
   wektora $a$ i estymaty $\hat{a}$ (pierwsze 15 współrzędnych). Pokazuje,
   że algorytm poprawnie identyfikuje aktywne współrzędne (indeksy 0, 1, 2)
   i przypisuje im bliskie prawdziwym wartości.
2. **(dół)** Błąd bezwzględny $|a_i - \hat{a}_i|$ na **wszystkich** $d=100$
   współrzędnych. Uwidacznia, że błąd jest skoncentrowany na współrzędnych
   nieaktywnych (małe wartości artefaktowe).

**Parametry eksperymentu:**

- $d = 100$, $m_\Phi = 50$, $m_X = 25$, $\varepsilon = 0.1$
- Prawdziwy wektor: $a = (0.864, 0.432, 0.259, 0, \ldots, 0)^T$ (znormalizowany)

**Interpretacja:**  
Wizualna walidacja algorytmu — potwierdza, że minimalizacja $\ell_1$ prawidłowo
promuje rzadkość rozwiązania i odzyskuje strukturę wektora $a$.

**Sugerowana sekcja w pracy:** 3.3

---

## Figura 3 — `fig3_alg2_convergence.pdf`

**Tytuł:** Algorytm 2: błąd rekonstrukcji podprzestrzeni w zależności od $m_\Phi$

**Co przedstawia:**  
Wykres liniowy (z pasmem błędu) pokazujący zbieżność błędu rekonstrukcji
podprzestrzeni $\|A^T A - \hat{A}^T \hat{A}\|_F$ dla Algorytmu 2 (przypadek ogólny $k \geq 1$).
Dwie krzywe: $k = 2$ i $k = 3$.

**Parametry eksperymentu:**

- $d = 80$, $m_X = 40$, $\varepsilon = 0.1$
- Funkcja testowa: $f(x) = \|Ax\|^2$ z rzadką macierzą $A$ ($k \times 3k$ aktywnych)
- 8 prób losowych, metryka: norma Frobeniusa rzutu ortogonalnego

**Oczekiwany wynik:**  
Błąd maleje ze wzrostem $m_\Phi$. Wyższe $k$ wymaga więcej pomiarów
(większa złożoność problemu). Odpowiada Twierdzeniu 4.1 — błąd
$\nu_2 \sim O(k^{1/q} \cdot m_\Phi^{1/2-1/q})$.

**Sugerowana sekcja w pracy:** 3.3

---

## Figura 4 — `fig4_alg2_singular_values.pdf`

**Tytuł:** Wartości osobliwe macierzy $\hat{X}^T$ — przerwa spektralna

**Co przedstawia:**  
Wykres słupkowy 15 największych wartości osobliwych macierzy $\hat{X}^T$
(odzyskanej w Algorytmie 2). Dwa pierwsze słupki ($k=2$) wyróżnione kolorem
odpowiadają aktywnym kierunkom. Przerwa spektralna (spectral gap) między
$\sigma_k$ a $\sigma_{k+1}$ zaznaczona linią przerywaną.

**Parametry eksperymentu:**

- $d = 80$, $k = 2$, $m_\Phi = 50$, $m_X = 40$

**Interpretacja:**  
Wyraźna przerwa spektralna potwierdza, że macierz $X$ ma efektywny rząd $k=2$,
co jest warunkiem poprawnego działania SVD w Algorytmie 2. Stabilność przerwy
gwarantowana jest twierdzeniem Wedina (Theorem 4.3 w pracy Vybirala).

**Sugerowana sekcja w pracy:** 3.3 (analiza struktury macierzy $\hat{X}$)

---

## Figura 5 — `fig5_epsilon_sensitivity.pdf`

**Tytuł:** Wpływ parametru $\varepsilon$ na jakość rekonstrukcji

**Co przedstawia:**  
Wykres log-log błędu rekonstrukcji w funkcji kroku różnicy skończonej $\varepsilon$.
Dwie krzywe:

1. **Bez szumu ($\sigma=0$):** błąd rośnie monotonicznie z $\varepsilon$ (dominuje reszta
   aproksymacji Taylora $O(\varepsilon)$ z równania (14)).
2. **Z szumem ($\sigma=0.01$):** dla małych $\varepsilon$ szum jest amplifikowany przez
   dzielenie $1/\varepsilon$ (zjawisko _noise folding_), co prowadzi do kompletnej utraty
   rekonstrukcji. Dopiero przy dużych $\varepsilon \geq 0.5$ szum jest wystarczająco
   tłumiony, by algorytm odzyskał sygnał.

**Parametry eksperymentu:**

- $d = 100$, $m_\Phi = 50$, $m_X = 25$
- $\varepsilon \in \{0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0\}$
- 10 prób losowych

**Interpretacja:**  
W przypadku bezszumowym mniejsze $\varepsilon$ jest zawsze lepsze (lepsze
przybliżenie pochodnej kierunkowej). W obecności szumu pojawia się kompromis
między dokładnością Taylora a amplifikacją szumu — zgodnie z Uwagą 4(iii)
pracy Vybirala. Optymalne $\varepsilon$ zależy od poziomu szumu $\sigma$.

**Sugerowana sekcja w pracy:** 3.3 (analiza wrażliwości hiperparametrów)

---

## Figura 6 — `fig6_alg1_noise.pdf`

**Tytuł:** Algorytm 1: odporność na szum Gaussowski

**Co przedstawia:**  
Błąd rekonstrukcji wektora $a$ w zależności od poziomu szumu $\sigma$
dodanego do pomiarów funkcji. Dwie krzywe z różnymi $\varepsilon$:

1. **$\varepsilon=0.1$:** niska tolerancja na szum — algorytm daje doskonałe wyniki
   przy $\sigma \leq 0.001$, ale gwałtownie traci dokładność powyżej $\sigma \approx 0.002$
   (przejście fazowe).
2. **$\varepsilon=0.5$:** wyższy błąd bazowy (reszta Taylora), ale znacznie większa
   odporność na szum — algorytm działa poprawnie aż do $\sigma \approx 0.01$.

**Parametry eksperymentu:**

- $d = 100$, $m_\Phi = 50$, $m_X = 25$
- $\sigma \in \{0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05\}$
- 8 prób losowych
- Dla $\sigma > 0$ stosowany jest relaxed basis pursuit (BPDN)

**Interpretacja:**  
Wykres demonstruje kompromis: mniejsze $\varepsilon$ daje lepszą dokładność
bezszumową, ale amplifikuje szum (efektywny szum $\propto \sigma/\varepsilon$).
Przejście fazowe odpowiada granicy, powyżej której compressed sensing traci
własność RIP. Odpowiada Sekcji 5.1 pracy Vybirala.

**Sugerowana sekcja w pracy:** 3.4 (Doświadczenia numeryczne w obecności szumu)

---

## Figura 7 — `fig7_dimension_scaling.pdf`

**Tytuł:** Skalowanie błędu Algorytmu 1 z wymiarem $d$

**Co przedstawia:**  
Dwa panele badające, jak zmienia się błąd rekonstrukcji ze wzrostem wymiaru $d$
przy stałej rzadkości $s = 3$:

1. **(a) Stałe $m_\Phi = 40$:** Błąd jako funkcja $d$ przy ustalonej
   liczbie pomiarów. Pokazuje, że nawet przy stałym $m_\Phi$ algorytm
   działa dobrze w wysokich wymiarach (bo $m_\Phi$ zależy logarytmicznie od $d$).

2. **(b) $m_\Phi = \lceil 10 \ln d \rceil$:** Błąd przy skalowaniu $m_\Phi$
   logarytmicznie z $d$. Adnotacje przy punktach pokazują użyte $m_\Phi$.
   Potwierdza teoretyczną zależność, że wystarczy $m_\Phi = O(\log d)$ pomiarów
   dla stałej rzadkości.

**Parametry eksperymentu:**

- $d \in \{30, 50, 80, 120, 200, 300\}$
- $m_X = 25$, $\varepsilon = 0.1$, 8 prób losowych

**Interpretacja:**  
Kluczowa obserwacja: złożoność problemu rośnie **logarytmicznie** (nie liniowo!)
z wymiarem $d$, co czyni metodę traktowalną nawet dla bardzo wysokich wymiarów.
Jest to zgodne z teorią compressed sensing (Twierdzenie 3.2) i wnioskami
o traktowalności z Sekcji 3.3 pracy Vybirala (Corollary 3.9).

**Sugerowana sekcja w pracy:** 3.3 (skalowanie z wymiarem)

---

## Jak zamieścić w LaTeX-u

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/fig1_alg1_convergence.pdf}
    \caption{Błąd rekonstrukcji wektora $a$ w zależności od liczby kierunków
             pomiarowych $m_\Phi$ dla różnych wymiarów $d$.}
    \label{fig:alg1-convergence}
\end{figure}
```
