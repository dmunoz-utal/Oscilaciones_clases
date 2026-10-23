import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paleta de colores educativa moderna
COLOR_M1 = "#1f77b4"     # Azul
COLOR_M2 = "#ff7f0e"     # Naranja
COLOR_FORCE = "#2ca02c"  # Verde
COLOR_FIT = "#d62728"    # Rojo
COLOR_ENV = "#9467bd"    # Morado
COLOR_RES = "#7f7f7f"    # Gris

def plot_1dof_time_response(sim_res: dict, sys_1dof, show_envelope: bool = True):
    """
    Gráfico interactivo de la respuesta en el tiempo x(t) para 1 GDL.
    Incluye envolvente analítica de decaimiento si es subamortiguado libre.
    """
    t = sim_res["t"]
    x = sim_res["x"]
    v = sim_res["v"]
    F = sim_res["F"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Desplazamiento x(t) y Fuerza aplicada F(t)/k", "Velocidad v(t) = dx/dt"),
        row_heights=[0.65, 0.35]
    )

    # Curva de desplazamiento
    fig.add_trace(
        go.Scatter(x=t, y=x, mode="lines", name="Desplazamiento x(t)",
                   line=dict(color=COLOR_M1, width=2.5)),
        row=1, col=1
    )

    # Curva de fuerza escalada (F(t)/k es la deformación estática equivalente)
    if sys_1dof.F0 > 0:
        F_scaled = F / sys_1dof.k
        fig.add_trace(
            go.Scatter(x=t, y=F_scaled, mode="lines", name="F(t)/k (Def. estática equivalente)",
                       line=dict(color=COLOR_FORCE, width=1.5, dash="dash")),
            row=1, col=1
        )

    # Envolvente exponencial si es vibración libre subamortiguada
    if show_envelope and sys_1dof.F0 == 0.0 and 0.0 < sys_1dof.zeta < 1.0:
        w_n = sys_1dof.omega_n
        z = sys_1dof.zeta
        w_d = sys_1dof.omega_d
        # Amplitud inicial equivalente
        C_amp = np.sqrt(sys_1dof.x0**2 + ((sys_1dof.v0 + z*w_n*sys_1dof.x0)/w_d)**2) if w_d > 0 else abs(sys_1dof.x0)
        env_pos = C_amp * np.exp(-z * w_n * t)
        env_neg = -env_pos

        fig.add_trace(
            go.Scatter(x=t, y=env_pos, mode="lines", name="Envolvente ±Ce^(-ζωnt)",
                       line=dict(color=COLOR_ENV, width=1.5, dash="dot")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=env_neg, mode="lines", name="Envolvente inferior",
                       showlegend=False, line=dict(color=COLOR_ENV, width=1.5, dash="dot")),
            row=1, col=1
        )

    # Velocidad
    fig.add_trace(
        go.Scatter(x=t, y=v, mode="lines", name="Velocidad v(t)",
                   line=dict(color="#17becf", width=2.0)),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=30, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Tiempo t [s]", row=2, col=1)
    fig.update_yaxes(title_text="Posición x [m]", row=1, col=1)
    fig.update_yaxes(title_text="Velocidad v [m/s]", row=2, col=1)

    return fig

def plot_phase_portrait(x: np.ndarray, v: np.ndarray, title: str = "Diagrama de Fase (Espacio de Estados)"):
    """
    Plano de fase: Velocidad v vs Posición x.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x, y=v, mode="lines",
            name="Trayectoria",
            line=dict(color=COLOR_M1, width=2.0)
        )
    )

    # Punto inicial y final
    fig.add_trace(
        go.Scatter(
            x=[x[0]], y=[v[0]], mode="markers+text",
            text=["Inicio"], textposition="top right",
            marker=dict(color="#2ca02c", size=10, symbol="circle"),
            name="Estado inicial"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x[-1]], y=[v[-1]], mode="markers+text",
            text=["Final"], textposition="bottom left",
            marker=dict(color="#d62728", size=10, symbol="x"),
            name="Estado final"
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=400,
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(title="Posición x [m]", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(title="Velocidad dx/dt [m/s]", zeroline=True, zerolinecolor="#ccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_resonance_curve(sys_1dof):
    """
    Curva teórica de Factor de Amplificación Dinámica M(r) y Ángulo de Fase phi(r).
    """
    r_vals, M_vals, phi_vals = sys_1dof.frequency_response_curve(r_max=3.0)
    current_params = sys_1dof.forced_response_params()
    r_curr = current_params["r"]
    M_curr = current_params["M"]
    phi_curr = current_params["phi_deg"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Factor de Amplificación Dinámica M = X / (F0/k)", "Ángulo de Desfasaje φ [grados]"),
        row_heights=[0.55, 0.45]
    )

    # Curva M(r)
    fig.add_trace(
        go.Scatter(x=r_vals, y=M_vals, mode="lines", name=f"M(r) para ζ = {sys_1dof.zeta:.3f}",
                   line=dict(color="#2ca02c", width=2.5)),
        row=1, col=1
    )

    # Punto actual en M(r)
    if not np.isinf(M_curr) and r_curr <= 3.0:
        fig.add_trace(
            go.Scatter(x=[r_curr], y=[M_curr], mode="markers+text",
                       text=[f"r={r_curr:.2f}, M={M_curr:.2f}"],
                       textposition="top right",
                       marker=dict(color="#d62728", size=10, symbol="diamond"),
                       name="Punto de Operación Actual"),
            row=1, col=1
        )

    # Línea vertical de resonancia r = 1.0
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Resonancia (r=1)", row=1, col=1)

    # Curva phi(r)
    fig.add_trace(
        go.Scatter(x=r_vals, y=phi_vals, mode="lines", name="φ(r)",
                   line=dict(color="#9467bd", width=2.5)),
        row=2, col=1
    )

    if r_curr <= 3.0:
        fig.add_trace(
            go.Scatter(x=[r_curr], y=[phi_curr], mode="markers+text",
                       text=[f"φ={phi_curr:.1f}°"],
                       textposition="top right",
                       marker=dict(color="#d62728", size=10, symbol="diamond"),
                       showlegend=False),
            row=2, col=1
        )

    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=90.0, line_dash="dot", line_color="gray",
                  annotation_text="90° (En resonancia)", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Razón de frecuencias r = ω / ωn", row=2, col=1)
    fig.update_yaxes(title_text="M(r) [adim]", row=1, col=1)
    fig.update_yaxes(title_text="Desfase φ [°]", row=2, col=1, range=[0, 185])

    return fig

def plot_2dof_time_response(sim_res: dict, F01: float, F02: float):
    """
    Gráfico interactivo de desplazamiento temporal para 2 GDL: x1(t) y x2(t).
    """
    t = sim_res["t"]
    x1 = sim_res["x1"]
    x2 = sim_res["x2"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=t, y=x1, mode="lines", name="Masa 1: x₁(t)",
                   line=dict(color=COLOR_M1, width=2.5))
    )
    fig.add_trace(
        go.Scatter(x=t, y=x2, mode="lines", name="Masa 2: x₂(t)",
                   line=dict(color=COLOR_M2, width=2.5))
    )

    fig.update_layout(
        template="plotly_white",
        title="Respuesta en el Tiempo de Masas Acopladas: x₁(t) y x₂(t)",
        height=450,
        margin=dict(l=40, r=30, t=50, b=40),
        hovermode="x unified",
        xaxis=dict(title="Tiempo t [s]"),
        yaxis=dict(title="Desplazamiento [m]", zeroline=True, zerolinecolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_2dof_orbit(x1: np.ndarray, x2: np.ndarray):
    """
    Plano de configuración x1 vs x2 (muestra acoplamiento y trayectoria en el espacio de coordenadas generalizadas).
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x1, y=x2, mode="lines", name="Trayectoria x₁ vs x₂",
                   line=dict(color="#8c564b", width=2.0))
    )
    fig.add_trace(
        go.Scatter(x=[x1[0]], y=[x2[0]], mode="markers+text",
                   text=["Inicio"], textposition="top right",
                   marker=dict(color="#2ca02c", size=10), name="Inicio")
    )
    fig.add_trace(
        go.Scatter(x=[x1[-1]], y=[x2[-1]], mode="markers+text",
                   text=["Fin"], textposition="bottom left",
                   marker=dict(color="#d62728", size=10, symbol="x"), name="Fin")
    )

    fig.update_layout(
        template="plotly_white",
        title="Plano de Configuración Acoplada (x₁ vs x₂)",
        height=400,
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(title="Desplazamiento Masa 1 x₁ [m]", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(title="Desplazamiento Masa 2 x₂ [m]", zeroline=True, zerolinecolor="#ccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_mode_shapes(modal_res: dict):
    """
    Gráfico de barras comparativo de las formas modales normalizadas (Modo 1 y Modo 2).
    """
    mode1 = modal_res["mode1"]
    mode2 = modal_res["mode2"]
    w1 = modal_res["omega_n1"]
    w2 = modal_res["omega_n2"]
    f1 = modal_res["f_n1"]
    f2 = modal_res["f_n2"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Modo 1: ωₙ₁ = {w1:.2f} rad/s ({f1:.2f} Hz)<br><sub>Relación x₂/x₁ = {mode1[1]:.2f} (En fase)</sub>",
            f"Modo 2: ωₙ₂ = {w2:.2f} rad/s ({f2:.2f} Hz)<br><sub>Relación x₂/x₁ = {mode2[1]:.2f} (En contrafase)</sub>"
        )
    )

    x_labels = ["Masa 1", "Masa 2"]

    # Modo 1
    fig.add_trace(
        go.Bar(x=x_labels, y=mode1, marker_color=COLOR_M1,
               text=[f"{mode1[0]:.2f}", f"{mode1[1]:.2f}"], textposition="outside",
               name="Modo 1"),
        row=1, col=1
    )

    # Modo 2
    fig.add_trace(
        go.Bar(x=x_labels, y=mode2, marker_color=COLOR_M2,
               text=[f"{mode2[0]:.2f}", f"{mode2[1]:.2f}"], textposition="outside",
               name="Modo 2"),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_white",
        title="Formas Modales Normalizadas (Autovectores)",
        height=380,
        showlegend=False,
        margin=dict(l=40, r=30, t=70, b=40)
    )
    fig.update_yaxes(title_text="Amplitud Normalizada", zeroline=True, zerolinecolor="#999")
    return fig

def plot_2dof_resonance_curves(sys_2dof):
    """
    Curvas de respuesta en frecuencia X1(w) y X2(w) para 2 GDL.
    """
    w_vals, X1, X2 = sys_2dof.frequency_response_sweep(w_max=15.0)
    modal = sys_2dof.modal_analysis()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=w_vals, y=X1, mode="lines", name="Amplitud Masa 1 |X₁(ω)|",
                   line=dict(color=COLOR_M1, width=2.5))
    )
    fig.add_trace(
        go.Scatter(x=w_vals, y=X2, mode="lines", name="Amplitud Masa 2 |X₂(ω)|",
                   line=dict(color=COLOR_M2, width=2.5))
    )

    # Líneas verticales en las frecuencias naturales
    fig.add_vline(x=modal["omega_n1"], line_dash="dash", line_color="#2ca02c",
                  annotation_text=f"ωₙ₁ = {modal['omega_n1']:.2f}")
    fig.add_vline(x=modal["omega_n2"], line_dash="dash", line_color="#d62728",
                  annotation_text=f"ωₙ₂ = {modal['omega_n2']:.2f}")

    if sys_2dof.omega > 0:
        fig.add_vline(x=sys_2dof.omega, line_dash="dot", line_color="#9467bd",
                      annotation_text=f"ω excitación = {sys_2dof.omega:.2f}")

    fig.update_layout(
        template="plotly_white",
        title="Respuesta en Frecuencia del Sistema de 2 GDL",
        xaxis=dict(title="Frecuencia de excitación ω [rad/s]"),
        yaxis=dict(title="Amplitud de régimen permanente |X| [m]"),
        height=420,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_fourier_spectrum(fft_res: dict, components_info: list, freq_unit: str = "rad"):
    """
    Espectro unilateral de Fourier mostrando las frecuencias presentes y los componentes seleccionados.
    """
    if freq_unit == "rad":
        freqs = fft_res["freq_rad"]
        x_label = "Frecuencia ω [rad/s]"
    else:
        freqs = fft_res["freq_hz"]
        x_label = "Frecuencia f [Hz]"

    amps = fft_res["amplitude"]

    # Limitamos la visualización a frecuencias de interés (hasta donde la amplitud sea relevante)
    max_idx = min(len(freqs), max(50, int(len(freqs) * 0.4)))
    freqs_plot = freqs[:max_idx]
    amps_plot = amps[:max_idx]

    fig = go.Figure()

    # Espectro completo continuo / línea base
    fig.add_trace(
        go.Scatter(
            x=freqs_plot, y=amps_plot,
            mode="lines",
            name="Espectro FFT Completo",
            line=dict(color="#4682b4", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(70, 130, 180, 0.15)"
        )
    )

    # Resaltar los N componentes seleccionados
    sel_f = [info["freq_rad" if freq_unit == "rad" else "freq_hz"] for info in components_info]
    sel_a = [info["amplitude"] for info in components_info]
    sel_text = [f"Comp {info['rank']}: {sel_f[i]:.2f} {x_label.split()[-1]}, Amp={sel_a[i]:.3f}"
                for i, info in enumerate(components_info)]

    fig.add_trace(
        go.Scatter(
            x=sel_f, y=sel_a,
            mode="markers+text",
            text=[f"#{info['rank']}" for info in components_info],
            textposition="top center",
            hovertext=sel_text,
            name="Componentes Seleccionados para Ajuste",
            marker=dict(color=COLOR_FIT, size=11, symbol="circle",
                        line=dict(color="black", width=1.5))
        )
    )

    # Dibujar líneas verticales desde 0 hasta el pico para los componentes seleccionados
    for sf, sa in zip(sel_f, sel_a):
        fig.add_shape(
            type="line",
            x0=sf, y0=0, x1=sf, y1=sa,
            line=dict(color=COLOR_FIT, width=2, dash="solid")
        )

    fig.update_layout(
        template="plotly_white",
        title="Espectro de Amplitud (Transformada Rápida de Fourier - FFT)",
        height=420,
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(title=x_label, zeroline=True),
        yaxis=dict(title="Amplitud Espectral [m]", zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_fourier_reconstruction(t: np.ndarray, x_orig: np.ndarray, x_rec: np.ndarray,
                               residual: np.ndarray, num_comp: int, r2: float, rmse: float):
    """
    Comparación temporal: Señal Original vs Señal Reconstruida con N componentes de Fourier,
    más el gráfico de error residual inferior.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            f"Comparación: Señal Original vs Reconstrucción ({num_comp} componentes) — R² = {r2:.4f}, RMSE = {rmse:.4e}",
            "Error Residual e(t) = x(t) - x̂(t)"
        ),
        row_heights=[0.70, 0.30]
    )

    # Señal Original
    fig.add_trace(
        go.Scatter(x=t, y=x_orig, mode="lines", name="Señal Original x(t)",
                   line=dict(color=COLOR_M1, width=2.5)),
        row=1, col=1
    )

    # Señal Reconstruida
    fig.add_trace(
        go.Scatter(x=t, y=x_rec, mode="lines", name=f"Ajuste Fourier ({num_comp} comp.)",
                   line=dict(color=COLOR_FIT, width=2.2, dash="dash")),
        row=1, col=1
    )

    # Error Residual
    fig.add_trace(
        go.Scatter(x=t, y=residual, mode="lines", name="Residuo e(t)",
                   line=dict(color=COLOR_RES, width=1.5)),
        row=2, col=1
    )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#bbb", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=30, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Tiempo t [s]", row=2, col=1)
    fig.update_yaxes(title_text="Posición [m]", row=1, col=1)
    fig.update_yaxes(title_text="Error [m]", row=2, col=1)

    return fig
