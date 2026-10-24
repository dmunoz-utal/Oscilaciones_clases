# ⚙️ Simulador de Oscilaciones y Análisis de Fourier (1 y 2 GDL)

Herramienta interactiva y pedagógica desarrollada para la enseñanza de vibraciones mecánicas y oscilaciones para estudiantes de ingeniería y física. Enfocada en **vibración libre** (no amortiguada y amortiguada) para 1 y 2 grados de libertad, animación física en tiempo real y análisis de Fourier.

Repositorio: [https://github.com/dmunoz-utal/Oscilaciones_clases](https://github.com/dmunoz-utal/Oscilaciones_clases)

---

## 🌟 Características Principales

### 1. Sistema de 1 Grado de Libertad (1 GDL) — Vibración Libre y Amortiguada
- **Ecuación gobernante:**
  $$m \ddot{x}(t) + c \dot{x}(t) + k x(t) = 0$$
- **Configuración completa e interactiva:**
  - Masa $m$, Rigidez $k$, Coeficiente de amortiguamiento $c$.
  - Condiciones iniciales: posición $x_0$ y velocidad $v_0$.
- **Cálculo analítico y numérico en tiempo real:**
  - Frecuencia natural no amortiguada $\omega_n = \sqrt{k/m}$ [rad/s] y $f_n$ [Hz].
  - Razón de amortiguamiento $\zeta = \frac{c}{2\sqrt{km}}$ y amortiguamiento crítico $c_c = 2\sqrt{km}$.
  - Clasificación automática del régimen de vibración:
    - *No amortiguado* ($\zeta = 0$)
    - *Subamortiguado* ($0 < \zeta < 1$)
    - *Críticamente amortiguado* ($\zeta = 1$)
    - *Sobreamortiguado* ($\zeta > 1$)
  - Frecuencia amortiguada $\omega_d = \omega_n \sqrt{1 - \zeta^2}$.
- **Visualizaciones interactivas:**
  - Animación física con masa, resorte y amortiguador en lona Canvas interactiva (con controles de Iniciar, Pausar y Reiniciar).
  - Gráfico dinámico progresivo y gráfico completo de $x(t)$ vs $t$ con controles de Zoom In, Zoom Out y Reset.
  - Diagrama de fase (Espacio de Estados: $x$ vs $\dot{x}$).

---

### 2. Sistema de 2 Grados de Libertad (2 GDL) — Modos y Batimiento
- **Ecuación matricial:**
  $$\mathbf{M} \ddot{\mathbf{x}} + \mathbf{C} \dot{\mathbf{x}} + \mathbf{K} \mathbf{x} = \mathbf{0}$$
- **Topologías seleccionables:**
  - Dos masas acopladas entre dos paredes fijas: $\text{Pared} - (k_1, c_1) - m_1 - (k_2, c_2) - m_2 - (k_3, c_3) - \text{Pared}$.
  - Dos masas en voladizo / extremo libre ($k_3 = 0, c_3 = 0$).
- **Análisis modal completo:**
  - Cálculo de autovalores y autovectores con $\det(\mathbf{K} - \omega^2 \mathbf{M}) = 0$.
  - Frecuencias naturales de los dos modos: $\omega_{n1}$ y $\omega_{n2}$ ($f_{n1}, f_{n2}$).
  - Formas modales normalizadas (Modo 1 en fase y Modo 2 en contrafase).
- **Preajustes didácticos:**
  - *Modo 1 Puro* (en fase).
  - *Modo 2 Puro* (en contrafase).
  - *Batimiento / Transferencia periódica de energía* ($m_1=m_2, k_1=k_3, k_2 \ll k_1$).
  - *Libre Amortiguado acoplado* ($c > 0$).
- **Visualizaciones:**
  - Animación física sincronizada de ambas masas acopladas con resortes y amortiguadores.
  - Curvas temporales de $x_1(t)$, $x_2(t)$ y opcionalmente la curva suma $x_1(t) + x_2(t)$.
  - Gráficos con controles interactivos de Zoom In, Zoom Out y Reset.
  - Diagrama de órbita acoplada ($x_1$ vs $x_2$) y diagramas de fase individuales.

---

### 3. Transformada de Fourier y Ajuste Espectral
- **Descomposición por FFT:**
  - Gráfico interactivo del espectro de amplitud unilateral vs frecuencia (en [rad/s] o [Hz]).
  - Detección automática de los picos de mayor relevancia espectral.
- **Ajuste y Reconstrucción seleccionando el número de componentes ($N$):**
  - Slider intuitivo para escoger cuántos armónicos / componentes dominantes ($N$) utilizar.
  - Reconstrucción armónica analítica en fase exacta:
    $$\hat{x}(t) = \bar{x} + \sum_{k=1}^N A_k \cos(\omega_k t + \phi_k)$$
  - Comparación superpuesta: Señal Original vs Señal Reconstruida.
  - Gráfico de Error Residual: $e(t) = x(t) - \hat{x}(t)$.
  - Indicadores cuantitativos: Coeficiente de determinación $R^2$, RMSE y porcentaje de energía capturada.
  - Tabla de componentes con frecuencia, amplitud, ángulo de fase y potencia relativa.

---

## 🚀 Despliegue en Vercel (100% Automático)

Este repositorio está optimizado para **Vercel** como aplicación web cliente moderna basada en `index.html`:

1. Conecta tu repositorio en Vercel.
2. Vercel detecta la configuración estática optimizada en `vercel.json`.
3. Haz clic en **"Deploy"**.

En segundos tendrás la aplicación web publicada con URL pública de alta velocidad.

---

## 💻 Ejecución Local

### Opción A: Versión Web Directa (Sin instalar nada)
Simplemente abre el archivo `index.html` en cualquier navegador web moderno (Chrome, Edge, Firefox, Safari).

### Opción B: Versión Streamlit en Python
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
*(O doble clic en `run_app.bat` en Windows).*

---

## 📂 Estructura del Proyecto

```
Oscilaciones_clases/
│
├── index.html                 # Aplicación web completa para Vercel
├── vercel.json                # Configuración de despliegue estático en Vercel
├── streamlit_app.py           # Aplicación interactiva en Python/Streamlit
├── modules/
│   ├── __init__.py
│   ├── system_1dof.py         # Modelo físico y solución numérica de 1 GDL
│   ├── system_2dof.py         # Análisis modal y simulación temporal de 2 GDL
│   ├── fourier_analysis.py    # FFT, detección de picos y reconstrucción espectral
│   └── plots.py               # Gráficos interactivos en Plotly
│
├── requirements.txt           # Librerías requeridas (numpy, scipy, plotly, streamlit, pandas)
├── run_app.bat                # Lanzador directo para Windows
├── run_app.py                 # Lanzador multiplataforma en Python
├── test_app.py                # Script de pruebas automatizadas
└── README.md                  # Documentación del proyecto
```

---

## 👨‍🏫 Desarrollado para
Clases y laboratorios de Mecánica, Dinámica de Estructuras y Teoría de Vibraciones.
