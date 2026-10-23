# ⚙️ Simulador de Oscilaciones y Análisis de Fourier (1 y 2 GDL)

Herramienta interactiva y pedagógica desarrollada en Python con **Streamlit** y **Plotly** para la enseñanza de vibraciones mecánicas y oscilaciones para estudiantes de ingeniería y física.

Repositorio: [https://github.com/dmunoz-utal/Oscilaciones_clases.git](https://github.com/dmunoz-utal/Oscilaciones_clases.git)

---

## 🌟 Características Principales

### 1. Sistema de 1 Grado de Libertad (1 GDL)
- **Ecuación gobernante:**
  $$m \ddot{x}(t) + c \dot{x}(t) + k x(t) = F_0 \cos(\omega t)$$
- **Configuración completa e interactiva:**
  - Masa $m$, Rigidez $k$, Coeficiente de amortiguamiento $c$.
  - Fuerza externa armónica: amplitud $F_0$ y frecuencia $\omega$.
  - Condiciones iniciales: posición $x_0$ y velocidad $v_0$.
- **Cálculo analítico y numérico en tiempo real:**
  - Frecuencia natural no amortiguada $\omega_n = \sqrt{k/m}$ [rad/s] y $f_n$ [Hz].
  - Razón de amortiguamiento $\zeta = \frac{c}{2\sqrt{km}}$ y amortiguamiento crítico $c_c$.
  - Clasificación automática del régimen:
    - *No amortiguado* ($\zeta = 0$)
    - *Subamortiguado* ($0 < \zeta < 1$)
    - *Críticamente amortiguado* ($\zeta = 1$)
    - *Sobreamortiguado* ($\zeta > 1$)
  - Frecuencia amortiguada $\omega_d = \omega_n \sqrt{1 - \zeta^2}$ y decremento logarítmico $\delta$.
  - Factor de amplificación dinámica $M(r)$, amplitud estacionaria $X$ y desfasaje $\phi$.
- **Visualizaciones interactivas:**
  - Desplazamiento $x(t)$ con envolvente analítica exponencial y velocidad $v(t)$.
  - Diagrama de fase (Espacio de Estados: $x$ vs $\dot{x}$).
  - Curva de resonancia y ángulo de fase en función de $r = \omega/\omega_n$ con indicación del punto de operación actual.

---

### 2. Sistema de 2 Grados de Libertad (2 GDL)
- **Ecuación matricial:**
  $$\mathbf{M} \ddot{\mathbf{x}} + \mathbf{C} \dot{\mathbf{x}} + \mathbf{K} \mathbf{x} = \mathbf{F}(t)$$
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
  - *Resonancia en Modo 1 y Modo 2*.
- **Visualizaciones:**
  - Oscilación temporal simultánea $x_1(t)$ y $x_2(t)$.
  - Formas modales en barras comparativas.
  - Plano de configuración acoplada ($x_1$ vs $x_2$) y diagramas de fase individuales.
  - Curvas de respuesta en frecuencia con identificación de picos resonantes.

---

### 3. Transformada de Fourier y Ajuste Espectral (Sección Separada)
- **Descomposición por FFT:**
  - Gráfico interactivo del espectro de amplitud unilateral vs frecuencia (en [rad/s] o [Hz]).
  - Detección automática de los picos de mayor relevancia energética.
- **Ajuste y Reconstrucción seleccionando el número de componentes ($N$):**
  - Slider intuitivo para escoger cuántos armónicos / componentes dominantes ($N$) utilizar.
  - Reconstrucción armónica analítica:
    $$\hat{x}(t) = \bar{x} + \sum_{k=1}^N A_k \cos(\omega_k t + \phi_k)$$
  - Comparación superpuesta: Señal Original vs Señal Reconstruida.
  - Gráfico inferior de Error Residual: $e(t) = x(t) - \hat{x}(t)$.
  - Indicadores cuantitativos: Coeficiente de determinación $R^2$, RMSE y porcentaje de energía capturada.
  - Tabla de armónicos con frecuencia, amplitud, ángulo de fase y potencia relativa.

---

## 🚀 Despliegue en Vercel (100% Automático y Gratuito)

¡Este repositorio está optimizado para **Vercel** sin necesidad de configurar servidores!

1. Ve a [vercel.com](https://vercel.com) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **"Add New..."** -> **"Project"**.
3. Selecciona el repositorio: `Oscilaciones_clases`.
4. Deja la configuración por defecto y haz clic en **"Deploy"**.

En menos de 15 segundos tendrás tu aplicación web publicada con URL pública (por ejemplo: `https://oscilaciones-clases.vercel.app`) lista para compartir con tus estudiantes.

---

## 💻 Ejecución Local

### Opción A: Versión Web Directa (Sin instalar librerías)
Simplemente abre el archivo `index.html` en cualquier navegador web (Chrome, Edge, Firefox, Safari).

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
├── app.py                     # Aplicación principal interactiva en Streamlit
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
