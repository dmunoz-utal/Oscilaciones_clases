import streamlit as np_st
import streamlit as st
import numpy as np
import pandas as pd

from modules.system_1dof import System1DOF
from modules.system_2dof import System2DOF
from modules.fourier_analysis import compute_fft, reconstruct_fourier_top_n, find_prominent_peaks
from modules.plots import (
    plot_1dof_time_response,
    plot_phase_portrait,
    plot_resonance_curve,
    plot_2dof_time_response,
    plot_2dof_orbit,
    plot_mode_shapes,
    plot_2dof_resonance_curves,
    plot_fourier_spectrum,
    plot_fourier_reconstruction,
)

# Configuración de página
st.set_page_config(
    page_title="Simulador de Oscilaciones y Fourier | 1 y 2 GDL",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración global de controles de interactividad y zoom para Plotly
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "scrollZoom": True
}

# Estilos CSS personalizados para mejorar el aspecto visual educativo
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #777;
    }
    .section-banner {
        background: linear-gradient(90deg, #1f77b4 0%, #2b5876 100%);
        color: white;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# BARRA LATERAL: NAVEGACIÓN Y SELECCIÓN DE MODO
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/physics.png", width=64)
st.sidebar.title("Oscilaciones y Vibraciones")
st.sidebar.markdown("**Herramienta docente interactiva**")

app_mode = st.sidebar.radio(
    "Selecciona el módulo:",
    [
        "🔵 1 Grado de Libertad (1 GDL)",
        "🟣 2 Grados de Libertad (2 GDL)",
        "📊 Análisis de Fourier y Ajuste Espectral",
        "📖 Guía Teórica y Fórmulas"
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("""
**Repositorio GitHub:**
[Oscilaciones_clases](https://github.com/dmunoz-utal/Oscilaciones_clases.git)
""")

# -------------------------------------------------------------
# MÓDULO 1: 1 GRADO DE LIBERTAD
# -------------------------------------------------------------
if app_mode == "🔵 1 Grado de Libertad (1 GDL)":
    st.markdown("""
    <div class="section-banner">
        <h2 style='margin:0; padding:0; color:white;'>⚙️ Sistema de 1 Grado de Libertad (1 GDL)</h2>
        <p style='margin:4px 0 0 0; color:#e0e0e0; font-size:0.95rem;'>
            Ecuación diferencial: <i>m x''(t) + c x'(t) + k x(t) = F₀ cos(ω t)</i>.
            Estudia vibraciones libres, amortiguadas, resonancia y desfasaje.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_side, col_main = st.columns([1, 2.5])

    with col_side:
        st.subheader("⚙️ Configuración del Sistema")

        preset_1dof = st.selectbox(
            "Preajustes didácticos:",
            [
                "Personalizado",
                "1. Libre No Amortiguada (c = 0)",
                "2. Libre Subamortiguada (0 < ζ < 1)",
                "3. Libre Críticamente Amortiguada (ζ = 1)",
                "4. Libre Sobreamortiguada (ζ > 1)",
                "5. Resonancia Armónica Pura (ω = ωn, c = 0)",
                "6. Resonancia Amortiguada (ω ≈ ωn, c > 0)",
                "7. Aislamiento de Vibraciones (ω >> ωn)"
            ],
            index=2
        )

        # Valores por defecto según preajuste
        if preset_1dof == "1. Libre No Amortiguada (c = 0)":
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 0.0, 0.0, 0.0, 0.15, 0.0
        elif preset_1dof == "2. Libre Subamortiguada (0 < ζ < 1)":
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 2.0, 0.0, 0.0, 0.20, 0.0
        elif preset_1dof == "3. Libre Críticamente Amortiguada (ζ = 1)":
            # c_c = 2*sqrt(k*m) = 2*sqrt(50*2) = 20
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 20.0, 0.0, 0.0, 0.20, 0.0
        elif preset_1dof == "4. Libre Sobreamortiguada (ζ > 1)":
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 35.0, 0.0, 0.0, 0.20, 0.5
        elif preset_1dof == "5. Resonancia Armónica Pura (ω = ωn, c = 0)":
            # wn = sqrt(50/2) = 5 rad/s
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 0.0, 5.0, 5.0, 0.0, 0.0
        elif preset_1dof == "6. Resonancia Amortiguada (ω ≈ ωn, c > 0)":
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 2.5, 8.0, 5.0, 0.0, 0.0
        elif preset_1dof == "7. Aislamiento de Vibraciones (ω >> ωn)":
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 3.0, 15.0, 14.0, 0.0, 0.0
        else:
            def_m, def_k, def_c, def_F0, def_w, def_x0, def_v0 = 2.0, 50.0, 1.5, 5.0, 4.0, 0.10, 0.0

        with st.expander("📦 Parámetros Estructurales (m, k, c)", expanded=True):
            m = st.number_input("Masa m [kg]:", min_value=0.05, max_value=500.0, value=def_m, step=0.5)
            k = st.number_input("Rigidez del resorte k [N/m]:", min_value=0.1, max_value=5000.0, value=def_k, step=5.0)
            c = st.number_input("Coeficiente de amortiguamiento c [N·s/m]:", min_value=0.0, max_value=200.0, value=def_c, step=0.5)

        with st.expander("⚡ Excitación (Fuerza Aplicada)", expanded=True):
            force_type = st.radio("Tipo de régimen:", ["Vibración Libre (F₀ = 0)", "Vibración Forzada Armónica F(t)"],
                                  index=0 if def_F0 == 0.0 else 1)
            if force_type == "Vibración Forzada Armónica F(t)":
                F0 = st.number_input("Amplitud de la fuerza F₀ [N]:", min_value=0.0, max_value=500.0, value=max(def_F0, 1.0), step=1.0)
                omega = st.number_input("Frecuencia de excitación ω [rad/s]:", min_value=0.05, max_value=50.0, value=max(def_w, 1.0), step=0.2)
            else:
                F0 = 0.0
                omega = 0.0

        with st.expander("🎯 Condiciones Iniciales y Tiempo", expanded=False):
            x0 = st.number_input("Posición inicial x₀ [m]:", min_value=-5.0, max_value=5.0, value=def_x0, step=0.02)
            v0 = st.number_input("Velocidad inicial v₀ [m/s]:", min_value=-20.0, max_value=20.0, value=def_v0, step=0.1)
            t_max = st.slider("Tiempo total de simulación [s]:", min_value=2.0, max_value=60.0, value=20.0, step=1.0)

    # Crear objeto de cálculo 1DOF
    sys_1dof = System1DOF(m=m, k=k, c=c, F0=F0, omega=omega, x0=x0, v0=v0)
    sim_res = sys_1dof.simulate(t_max=t_max, num_points=2500)
    forced_params = sys_1dof.forced_response_params()

    # Guardar en st.session_state para compartir con la pestaña de Fourier
    st.session_state["signal_1dof_t"] = sim_res["t"]
    st.session_state["signal_1dof_x"] = sim_res["x"]
    st.session_state["signal_1dof_name"] = f"1 GDL (m={m}kg, k={k}N/m, c={c}N·s/m)"

    with col_main:
        # Métricas principales
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Frecuencia Natural</div>
                <div class="metric-value">{sys_1dof.omega_n:.2f} <span style='font-size:0.8rem'>rad/s</span></div>
                <div class="metric-sub">{sys_1dof.f_n:.2f} Hz | T = {2*np.pi/sys_1dof.omega_n:.2f} s</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Razón Amortig. (ζ)</div>
                <div class="metric-value">{sys_1dof.zeta:.3f}</div>
                <div class="metric-sub">c_crít = {sys_1dof.c_crit:.2f} N·s/m</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Frec. Amortiguada</div>
                <div class="metric-value">{sys_1dof.omega_d:.2f} <span style='font-size:0.8rem'>rad/s</span></div>
                <div class="metric-sub">{sys_1dof.f_d:.2f} Hz</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            amp_display = f"{forced_params['X_steady']:.3f} m" if (F0 > 0 and not np.isinf(forced_params['X_steady'])) else f"{sim_res['X_max_observed']:.3f} m"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Amplitud Máx / Estac.</div>
                <div class="metric-value">{amp_display}</div>
                <div class="metric-sub">{'M = ' + f"{forced_params['M']:.2f}" if F0 > 0 else 'Obs. en tiempo'}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            phase_display = f"{forced_params['phi_deg']:.1f}°" if F0 > 0 else "0.0°"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Ángulo de Desfase</div>
                <div class="metric-value">{phase_display}</div>
                <div class="metric-sub">{'r = ' + f"{forced_params['r']:.2f}" if F0 > 0 else 'Vibración libre'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"**Régimen actual:** {sys_1dof.regime_label}")

        # Pestañas de gráficos
        tab_time, tab_phase, tab_bode, tab_table = st.tabs([
            "📈 Gráfico de Oscilación x(t)",
            "🌀 Diagrama de Fase (x vs v)",
            "🎯 Curva de Resonancia y Fase",
            "📋 Parámetros y Exportación"
        ])

        with tab_time:
            fig_time = plot_1dof_time_response(sim_res, sys_1dof, show_envelope=True)
            st.plotly_chart(fig_time, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_phase:
            fig_phase = plot_phase_portrait(sim_res["x"], sim_res["v"], title="Plano de Fase: Velocidad vs Desplazamiento")
            st.plotly_chart(fig_phase, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_bode:
            fig_bode = plot_resonance_curve(sys_1dof)
            st.plotly_chart(fig_bode, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_table:
            st.markdown("#### Resumen Analítico de Variables del Sistema")
            df_params = pd.DataFrame([
                {"Parámetro": "Masa (m)", "Valor": f"{m:.3f}", "Unidad": "kg", "Descripción": "Inercia del sistema"},
                {"Parámetro": "Rigidez (k)", "Valor": f"{k:.3f}", "Unidad": "N/m", "Descripción": "Constante del resorte"},
                {"Parámetro": "Amortiguamiento (c)", "Valor": f"{c:.3f}", "Unidad": "N·s/m", "Descripción": "Coeficiente disipativo"},
                {"Parámetro": "Frecuencia natural (ωn)", "Valor": f"{sys_1dof.omega_n:.3f}", "Unidad": "rad/s", "Descripción": "sqrt(k/m)"},
                {"Parámetro": "Frecuencia natural cíclica (fn)", "Valor": f"{sys_1dof.f_n:.3f}", "Unidad": "Hz", "Descripción": "ωn / (2π)"},
                {"Parámetro": "Amortiguamiento crítico (cc)", "Valor": f"{sys_1dof.c_crit:.3f}", "Unidad": "N·s/m", "Descripción": "2 * sqrt(k*m)"},
                {"Parámetro": "Razón de amortiguamiento (ζ)", "Valor": f"{sys_1dof.zeta:.4f}", "Unidad": "adimensional", "Descripción": "c / cc"},
                {"Parámetro": "Frecuencia amortiguada (ωd)", "Valor": f"{sys_1dof.omega_d:.3f}", "Unidad": "rad/s", "Descripción": "ωn * sqrt(1 - ζ²)"},
                {"Parámetro": "Decremento logarítmico (δ)", "Valor": f"{sys_1dof.log_decrement:.3f}", "Unidad": "adimensional", "Descripción": "2πζ / sqrt(1 - ζ²)"},
                {"Parámetro": "Amplitud fuerza (F₀)", "Valor": f"{F0:.3f}", "Unidad": "N", "Descripción": "Magnitud excitación"},
                {"Parámetro": "Frecuencia excitación (ω)", "Valor": f"{omega:.3f}", "Unidad": "rad/s", "Descripción": "Pulsación fuerza armónica"},
                {"Parámetro": "Razón de frecuencias (r)", "Valor": f"{forced_params['r']:.3f}", "Unidad": "adimensional", "Descripción": "ω / ωn"},
                {"Parámetro": "Factor amplificación dinámica (M)", "Valor": f"{forced_params['M']:.3f}", "Unidad": "adimensional", "Descripción": "X / (F0/k)"},
                {"Parámetro": "Desfasaje (φ)", "Valor": f"{forced_params['phi_deg']:.2f}", "Unidad": "grados", "Descripción": "Retardo respecto a la fuerza"},
            ])
            st.dataframe(df_params, use_container_width=True)

            # Exportar datos temporales
            df_export = pd.DataFrame({
                "tiempo_s": sim_res["t"],
                "posicion_m": sim_res["x"],
                "velocidad_m_s": sim_res["v"],
                "fuerza_N": sim_res["F"]
            })
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Serie de Tiempo en CSV",
                data=csv_data,
                file_name="simulacion_1gdl.csv",
                mime="text/csv"
            )


# -------------------------------------------------------------
# MÓDULO 2: 2 GRADOS DE LIBERTAD
# -------------------------------------------------------------
elif app_mode == "🟣 2 Grados de Libertad (2 GDL)":
    st.markdown("""
    <div class="section-banner" style="background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);">
        <h2 style='margin:0; padding:0; color:white;'>⚙️ Sistema de 2 Grados de Libertad (2 GDL)</h2>
        <p style='margin:4px 0 0 0; color:#e0e0e0; font-size:0.95rem;'>
            Ecuación matricial: <i><b>M</b> x''(t) + <b>C</b> x'(t) + <b>K</b> x(t) = <b>F</b>(t)</i>.
            Descubre modos normales de vibración, frecuencias naturales acopladas, batimientos y resonancia.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_side, col_main = st.columns([1, 2.5])

    with col_side:
        st.subheader("⚙️ Configuración del Sistema 2 GDL")

        config_type = st.selectbox(
            "Topología del sistema:",
            [
                "Pared — m₁ — m₂ — Pared (Extremos fijos)",
                "Pared — m₁ — m₂ — Libre (Extremo libre / Voladizo)"
            ]
        )
        cfg_key = "fixed-fixed" if "Extremos fijos" in config_type else "fixed-free"

        preset_2dof = st.selectbox(
            "Preajustes didácticos 2 GDL:",
            [
                "Personalizado",
                "1. Modo 1 Puro (Movimiento en fase)",
                "2. Modo 2 Puro (Movimiento en contrafase)",
                "3. Batimiento / Transferencia de Energía (m1=m2, k2 débil)",
                "4. Resonancia en Modo 1 (ω = ωn1)",
                "5. Resonancia en Modo 2 (ω = ωn2)",
                "6. Amortiguado acoplado (c1, c2 > 0)"
            ],
            index=3
        )

        # Definir configuraciones por defecto
        if preset_2dof == "1. Modo 1 Puro (Movimiento en fase)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 20.0, 40.0
            p_c1, p_c2, p_c3 = 0.0, 0.0, 0.0
            p_F01, p_F02, p_w = 0.0, 0.0, 0.0
            # Modo 1 para simetría: x1 = 0.15, x2 = 0.15
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.15, 0.15, 0.0, 0.0
        elif preset_2dof == "2. Modo 2 Puro (Movimiento en contrafase)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 20.0, 40.0
            p_c1, p_c2, p_c3 = 0.0, 0.0, 0.0
            p_F01, p_F02, p_w = 0.0, 0.0, 0.0
            # Modo 2 para simetría: x1 = 0.15, x2 = -0.15
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.15, -0.15, 0.0, 0.0
        elif preset_2dof == "3. Batimiento / Transferencia de Energía (m1=m2, k2 débil)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 50.0, 5.0, 50.0
            p_c1, p_c2, p_c3 = 0.0, 0.0, 0.0
            p_F01, p_F02, p_w = 0.0, 0.0, 0.0
            # Soltar m1 desplazada con m2 en reposo genera batimiento puro
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.20, 0.0, 0.0, 0.0
        elif preset_2dof == "4. Resonancia en Modo 1 (ω = ωn1)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 20.0, 40.0
            p_c1, p_c2, p_c3 = 0.5, 0.5, 0.5
            # wn1 = sqrt(40/2) = 4.47 rad/s
            p_F01, p_F02, p_w = 5.0, 0.0, 4.47
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.0, 0.0, 0.0, 0.0
        elif preset_2dof == "5. Resonancia en Modo 2 (ω = ωn2)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 20.0, 40.0
            p_c1, p_c2, p_c3 = 0.5, 0.5, 0.5
            # wn2 = sqrt((40+40)/2) = 6.32 rad/s
            p_F01, p_F02, p_w = 5.0, 0.0, 6.32
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.0, 0.0, 0.0, 0.0
        elif preset_2dof == "6. Amortiguado acoplado (c1, c2 > 0)":
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 15.0, 40.0
            p_c1, p_c2, p_c3 = 1.2, 0.8, 1.2
            p_F01, p_F02, p_w = 0.0, 0.0, 0.0
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.20, 0.0, 0.0, 0.0
        else:
            p_m1, p_m2, p_k1, p_k2, p_k3 = 2.0, 2.0, 40.0, 20.0, 40.0
            p_c1, p_c2, p_c3 = 0.2, 0.2, 0.2
            p_F01, p_F02, p_w = 0.0, 0.0, 0.0
            p_x1_0, p_x2_0, p_v1_0, p_v2_0 = 0.15, 0.0, 0.0, 0.0

        with st.expander("📦 Masas (m₁, m₂)", expanded=True):
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                m1 = st.number_input("Masa m₁ [kg]:", min_value=0.1, max_value=200.0, value=p_m1, step=0.5)
            with col_m2:
                m2 = st.number_input("Masa m₂ [kg]:", min_value=0.1, max_value=200.0, value=p_m2, step=0.5)

        with st.expander("🔗 Resortes (k₁, k₂, k₃)", expanded=True):
            k1 = st.number_input("Rigidez k₁ [N/m]:", min_value=0.1, max_value=2000.0, value=p_k1, step=5.0)
            k2 = st.number_input("Rigidez acople k₂ [N/m]:", min_value=0.1, max_value=2000.0, value=p_k2, step=5.0)
            if cfg_key == "fixed-fixed":
                k3 = st.number_input("Rigidez k₃ [N/m]:", min_value=0.1, max_value=2000.0, value=p_k3, step=5.0)
            else:
                k3 = 0.0
                st.caption("Extremo libre: k₃ = 0 N/m")

        with st.expander("💧 Amortiguadores (c₁, c₂, c₃)", expanded=False):
            c1 = st.number_input("Amortig. c₁ [N·s/m]:", min_value=0.0, max_value=50.0, value=p_c1, step=0.1)
            c2 = st.number_input("Amortig. acople c₂ [N·s/m]:", min_value=0.0, max_value=50.0, value=p_c2, step=0.1)
            if cfg_key == "fixed-fixed":
                c3 = st.number_input("Amortig. c₃ [N·s/m]:", min_value=0.0, max_value=50.0, value=p_c3, step=0.1)
            else:
                c3 = 0.0

        with st.expander("⚡ Fuerzas Aplicadas", expanded=False):
            f_active = st.checkbox("Activar fuerza armónica externa", value=(p_F01 > 0 or p_F02 > 0))
            if f_active:
                F01 = st.number_input("Amplitud F₀₁ sobre m₁ [N]:", min_value=0.0, max_value=200.0, value=max(p_F01, 2.0), step=1.0)
                F02 = st.number_input("Amplitud F₀₂ sobre m₂ [N]:", min_value=0.0, max_value=200.0, value=p_F02, step=1.0)
                omega_2dof = st.number_input("Frecuencia ω [rad/s]:", min_value=0.1, max_value=30.0, value=max(p_w, 2.0), step=0.2)
            else:
                F01, F02, omega_2dof = 0.0, 0.0, 0.0

        with st.expander("🎯 Condiciones Iniciales", expanded=False):
            col_ci1, col_ci2 = st.columns(2)
            with col_ci1:
                x1_0 = st.number_input("x₁(0) [m]:", min_value=-2.0, max_value=2.0, value=p_x1_0, step=0.02)
                v1_0 = st.number_input("v₁(0) [m/s]:", min_value=-10.0, max_value=10.0, value=p_v1_0, step=0.1)
            with col_ci2:
                x2_0 = st.number_input("x₂(0) [m]:", min_value=-2.0, max_value=2.0, value=p_x2_0, step=0.02)
                v2_0 = st.number_input("v₂(0) [m/s]:", min_value=-10.0, max_value=10.0, value=p_v2_0, step=0.1)
            t_max_2dof = st.slider("Tiempo total [s]:", min_value=5.0, max_value=80.0, value=25.0, step=1.0)

    # Crear objeto de cálculo 2DOF
    sys_2dof = System2DOF(
        m1=m1, m2=m2, k1=k1, k2=k2, k3=k3,
        c1=c1, c2=c2, c3=c3,
        config_type=cfg_key,
        F01=F01, F02=F02, omega=omega_2dof,
        x1_0=x1_0, v1_0=v1_0, x2_0=x2_0, v2_0=v2_0
    )
    modal = sys_2dof.modal_analysis()
    sim_2dof = sys_2dof.simulate(t_max=t_max_2dof, num_points=3000)
    forced_2dof = sys_2dof.forced_response_params()

    # Guardar en st.session_state para la pestaña de Fourier
    st.session_state["signal_2dof_t"] = sim_2dof["t"]
    st.session_state["signal_2dof_x1"] = sim_2dof["x1"]
    st.session_state["signal_2dof_x2"] = sim_2dof["x2"]
    st.session_state["signal_2dof_xsum"] = sim_2dof["x1"] + sim_2dof["x2"]
    st.session_state["signal_2dof_name1"] = f"2 GDL - Masa 1 (m1={m1}kg)"
    st.session_state["signal_2dof_name2"] = f"2 GDL - Masa 2 (m2={m2}kg)"
    st.session_state["signal_2dof_namesum"] = f"2 GDL - Suma x₁(t) + x₂(t)"

    with col_main:
        k1_c, k2_c, k3_c, k4_c = st.columns(4)
        with k1_c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Modo 1 (ωₙ₁)</div>
                <div class="metric-value">{modal['omega_n1']:.2f} <span style='font-size:0.8rem'>rad/s</span></div>
                <div class="metric-sub">{modal['f_n1']:.2f} Hz | En fase (x₂/x₁ = {modal['mode1'][1]:.2f})</div>
            </div>
            """, unsafe_allow_html=True)
        with k2_c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Modo 2 (ωₙ₂)</div>
                <div class="metric-value">{modal['omega_n2']:.2f} <span style='font-size:0.8rem'>rad/s</span></div>
                <div class="metric-sub">{modal['f_n2']:.2f} Hz | Contrafase (x₂/x₁ = {modal['mode2'][1]:.2f})</div>
            </div>
            """, unsafe_allow_html=True)
        with k3_c:
            x1_max = np.max(np.abs(sim_2dof["x1"]))
            x2_max = np.max(np.abs(sim_2dof["x2"]))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Amplitudes Máximas</div>
                <div class="metric-value">{x1_max:.3f} / {x2_max:.3f} <span style='font-size:0.8rem'>m</span></div>
                <div class="metric-sub">Max |x₁| y Max |x₂| observadas</div>
            </div>
            """, unsafe_allow_html=True)
        with k4_c:
            fase_info = f"Δφ = {forced_2dof['phase_diff_deg']:.1f}°" if (F01 > 0 or F02 > 0) else "Vibración Libre"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Desfase Estacionario</div>
                <div class="metric-value">{fase_info}</div>
                <div class="metric-sub">{'ω exc = ' + f'{omega_2dof:.2f} rad/s' if omega_2dof > 0 else 'Sin forzamiento'}</div>
            </div>
            """, unsafe_allow_html=True)

        tab_2dof_time, tab_2dof_modes, tab_2dof_orbit, tab_2dof_bode, tab_2dof_mat = st.tabs([
            "📈 Respuesta en el Tiempo x₁(t) y x₂(t)",
            "📊 Formas Modales (Autovectores)",
            "🌀 Plano de Fase y Acoplamiento (x₁ vs x₂)",
            "🎯 Barrido de Frecuencia y Resonancias",
            "📋 Matrices del Sistema [M], [K], [C]"
        ])

        with tab_2dof_time:
            show_sum_streamlit = st.checkbox("Mostrar curva de la suma x₁(t) + x₂(t)", value=True)
            fig_2t = plot_2dof_time_response(sim_2dof, F01, F02, show_sum=show_sum_streamlit)
            st.plotly_chart(fig_2t, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_2dof_modes:
            fig_modes = plot_mode_shapes(modal)
            st.plotly_chart(fig_modes, use_container_width=True, config=PLOTLY_CONFIG)
            st.markdown("""
            > **Interpretación de Modos de Vibración:**
            > - **Modo 1:** Ambas masas oscilan en la misma dirección al unísono con frecuencia $\\omega_{n1}$.
            > - **Modo 2:** Las masas oscilan en sentidos opuestos (contrafase) con frecuencia $\\omega_{n2}$, existiendo un nodo intermedio donde el desplazamiento neto es nulo.
            """)

        with tab_2dof_orbit:
            col_orb1, col_orb2 = st.columns(2)
            with col_orb1:
                fig_orb = plot_2dof_orbit(sim_2dof["x1"], sim_2dof["x2"])
                st.plotly_chart(fig_orb, use_container_width=True, config=PLOTLY_CONFIG)
            with col_orb2:
                fig_pf1 = plot_phase_portrait(sim_2dof["x1"], sim_2dof["v1"], title="Plano de Fase Masa 1 (x₁ vs v₁)")
                st.plotly_chart(fig_pf1, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_2dof_bode:
            fig_sw = plot_2dof_resonance_curves(sys_2dof)
            st.plotly_chart(fig_sw, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_2dof_mat:
            st.markdown("#### Formulación Matricial del Sistema")
            col_m_mat, col_k_mat, col_c_mat = st.columns(3)
            with col_m_mat:
                st.latex(r"\mathbf{M} = \begin{bmatrix}" + f"{m1:.2f} & 0.00 \\\\ 0.00 & {m2:.2f}" + r"\end{bmatrix} \text{ [kg]}")
            with col_k_mat:
                K = sys_2dof.K_matrix
                st.latex(r"\mathbf{K} = \begin{bmatrix}" + f"{K[0,0]:.2f} & {K[0,1]:.2f} \\\\ {K[1,0]:.2f} & {K[1,1]:.2f}" + r"\end{bmatrix} \text{ [N/m]}")
            with col_c_mat:
                C = sys_2dof.C_matrix
                st.latex(r"\mathbf{C} = \begin{bmatrix}" + f"{C[0,0]:.2f} & {C[0,1]:.2f} \\\\ {C[1,0]:.2f} & {C[1,1]:.2f}" + r"\end{bmatrix} \text{ [N·s/m]}")


# -------------------------------------------------------------
# MÓDULO 3: TRANSFORMADA DE FOURIER Y AJUSTE ESPECTRAL
# -------------------------------------------------------------
elif app_mode == "📊 Análisis de Fourier y Ajuste Espectral":
    st.markdown("""
    <div class="section-banner" style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);">
        <h2 style='margin:0; padding:0; color:white;'>📊 Análisis de Fourier y Reconstrucción por Componentes</h2>
        <p style='margin:4px 0 0 0; color:#e0e0e0; font-size:0.95rem;'>
            Descomposición espectral mediante la Transformada Rápida de Fourier (FFT).
            Escoge interactivamente el número de componentes (armónicos dominantes) para observar la reconstrucción de la señal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Selección de la fuente de señal
    col_ctrl, col_view = st.columns([1, 2.6])

    with col_ctrl:
        st.subheader("🎛️ Selección de Señal")

        available_sources = ["Señal Sintética de Prueba (Demostración)"]
        if "signal_1dof_t" in st.session_state:
            available_sources.append("Simulación 1 GDL: x(t)")
        if "signal_2dof_t" in st.session_state:
            available_sources.append("Simulación 2 GDL: Masa 1 x₁(t)")
            available_sources.append("Simulación 2 GDL: Masa 2 x₂(t)")
            available_sources.append("Simulación 2 GDL: Suma x₁(t) + x₂(t)")

        source_choice = st.selectbox(
            "Origen de la señal a analizar:",
            available_sources,
            index=1 if len(available_sources) > 1 else 0
        )

        # Cargar los datos correspondientes
        if source_choice == "Simulación 1 GDL: x(t)":
            t_signal = st.session_state["signal_1dof_t"]
            x_signal = st.session_state["signal_1dof_x"]
            signal_name = st.session_state["signal_1dof_name"]
        elif source_choice == "Simulación 2 GDL: Masa 1 x₁(t)":
            t_signal = st.session_state["signal_2dof_t"]
            x_signal = st.session_state["signal_2dof_x1"]
            signal_name = st.session_state["signal_2dof_name1"]
        elif source_choice == "Simulación 2 GDL: Masa 2 x₂(t)":
            t_signal = st.session_state["signal_2dof_t"]
            x_signal = st.session_state["signal_2dof_x2"]
            signal_name = st.session_state["signal_2dof_name2"]
        elif source_choice == "Simulación 2 GDL: Suma x₁(t) + x₂(t)":
            t_signal = st.session_state["signal_2dof_t"]
            x_signal = st.session_state["signal_2dof_xsum"]
            signal_name = st.session_state["signal_2dof_namesum"]
        else:
            # Señal sintética con 3 armónicos y decaimiento o ruido
            t_signal = np.linspace(0, 20, 2000)
            x_signal = (
                0.5 * np.cos(3.0 * t_signal) +
                0.25 * np.cos(7.5 * t_signal + np.pi/4) +
                0.12 * np.cos(12.0 * t_signal - np.pi/3)
            ) * np.exp(-0.03 * t_signal)
            signal_name = "Señal Sintética Educativa (3 frecuencias acopladas amortiguadas)"

        st.info(f"**Analizando:** {signal_name}")

        st.divider()
        st.subheader("⚙️ Parámetros del Ajuste")

        num_components = st.slider(
            "Número de componentes del ajuste (N):",
            min_value=1,
            max_value=25,
            value=3,
            step=1,
            help="Selecciona cuántos armónicos / picos de frecuencia con mayor energía usar para reconstruir la señal original."
        )

        freq_unit = st.radio(
            "Unidad en eje de frecuencias:",
            ["Radianes por segundo [rad/s]", "Hertz [Hz]"],
            index=0
        )
        unit_key = "rad" if "rad/s" in freq_unit else "hz"

    # Realizar el ajuste de Fourier
    recon_res = reconstruct_fourier_top_n(t_signal, x_signal, num_components=num_components)
    fft_res = recon_res["fft_res"]
    components_info = recon_res["components_info"]
    r2 = recon_res["r2"]
    rmse = recon_res["rmse"]
    energy_captured = recon_res["energy_captured_pct"]

    with col_view:
        # Tarjetas de rendimiento del ajuste
        kpi_f1, kpi_f2, kpi_f3, kpi_f4 = st.columns(4)
        with kpi_f1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Componentes Usados</div>
                <div class="metric-value">{num_components}</div>
                <div class="metric-sub">Picos dominantes seleccionados</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_f2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Bondad de Ajuste R²</div>
                <div class="metric-value">{r2:.4f}</div>
                <div class="metric-sub">{'Ajuste excelente' if r2 > 0.95 else 'Ajuste aproximado'}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_f3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Error RMSE</div>
                <div class="metric-value">{rmse:.4e}</div>
                <div class="metric-sub">Raíz error cuadrático medio</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_f4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Energía Capturada</div>
                <div class="metric-value">{energy_captured:.2f} %</div>
                <div class="metric-sub">Potencia espectral contenida</div>
            </div>
            """, unsafe_allow_html=True)

        tab_fft_spec, tab_fft_recon, tab_fft_table = st.tabs([
            "📈 Espectro de Frecuencia (FFT)",
            "🔄 Comparación y Error de Reconstrucción",
            "📋 Tabla de Componentes y Ecuación de Ajuste"
        ])

        with tab_fft_spec:
            fig_spec = plot_fourier_spectrum(fft_res, components_info, freq_unit=unit_key)
            st.plotly_chart(fig_spec, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("Los puntos rojos señalan los N picos de mayor amplitud seleccionados para reconstruir la señal.")

        with tab_fft_recon:
            fig_rec = plot_fourier_reconstruction(
                t_signal, x_signal, recon_res["x_rec"], recon_res["residual"],
                num_comp=num_components, r2=r2, rmse=rmse
            )
            st.plotly_chart(fig_rec, use_container_width=True, config=PLOTLY_CONFIG)

        with tab_fft_table:
            st.markdown(f"#### Ecuación Analítica del Ajuste con {num_components} Componentes:")
            # Generar fórmula KaTeX
            eq_terms = []
            if abs(recon_res["dc_val"]) > 1e-4:
                eq_terms.append(f"{recon_res['dc_val']:.4f}")
            for comp in components_info[:5]: # Mostrar los primeros 5 en la ecuación para no saturar
                A_k = comp["amplitude"]
                w_k = comp["freq_rad"]
                phi_k = comp["phase_rad"]
                sign_phi = "+" if phi_k >= 0 else "-"
                eq_terms.append(f"{A_k:.4f} \\cos({w_k:.2f} t {sign_phi} {abs(phi_k):.2f})")
            
            formula_str = " + ".join(eq_terms)
            if num_components > 5:
                formula_str += " + \\dots"
            st.latex(r"\hat{x}(t) = " + formula_str)

            st.markdown("#### Tabla de Componentes Espectrales Seleccionados")
            df_comps = pd.DataFrame([
                {
                    "Rango": f"#{c['rank']}",
                    "Frecuencia ω [rad/s]": f"{c['freq_rad']:.3f}",
                    "Frecuencia f [Hz]": f"{c['freq_hz']:.3f}",
                    "Amplitud Ak [m]": f"{c['amplitude']:.4f}",
                    "Fase φk [°]": f"{c['phase_deg']:.2f}",
                    "Fase φk [rad]": f"{c['phase_rad']:.3f}",
                    "Potencia Relativa [%]": f"{c['power_pct']:.2f}%"
                }
                for c in components_info
            ])
            st.dataframe(df_comps, use_container_width=True)


# -------------------------------------------------------------
# MÓDULO 4: GUÍA TEÓRICA Y FÓRMULAS
# -------------------------------------------------------------
elif app_mode == "📖 Guía Teórica y Fórmulas":
    st.markdown("""
    <div class="section-banner" style="background: linear-gradient(90deg, #302b63 0%, #24243e 100%);">
        <h2 style='margin:0; padding:0; color:white;'>📖 Guía Didáctica: Teoría de Vibraciones y Fourier</h2>
        <p style='margin:4px 0 0 0; color:#e0e0e0; font-size:0.95rem;'>
            Conceptos clave, deducciones y fórmulas de referencia para las clases de oscilaciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("1. Sistema de 1 Grado de Libertad (1 GDL)", expanded=True):
        st.markdown(r"""
        La ecuación diferencial fundamental según la 2ª Ley de Newton es:
        $$m \ddot{x}(t) + c \dot{x}(t) + k x(t) = F_0 \cos(\omega t)$$

        Dividiendo por la masa $m$:
        $$\ddot{x}(t) + 2 \zeta \omega_n \dot{x}(t) + \omega_n^2 x(t) = \frac{F_0}{m} \cos(\omega t)$$

        **Parámetros fundamentales:**
        - **Frecuencia natural no amortiguada:** $\omega_n = \sqrt{\frac{k}{m}}$ [rad/s], $f_n = \frac{\omega_n}{2\pi}$ [Hz].
        - **Amortiguamiento crítico:** $c_c = 2\sqrt{km} = 2 m \omega_n$.
        - **Razón de amortiguamiento:** $\zeta = \frac{c}{c_c} = \frac{c}{2\sqrt{km}}$.
        - **Frecuencia natural amortiguada (cuando $\zeta < 1$):** $\omega_d = \omega_n \sqrt{1 - \zeta^2}$.
        - **Factor de amplificación dinámica:**
          $$M(r) = \frac{X}{F_0/k} = \frac{1}{\sqrt{(1 - r^2)^2 + (2\zeta r)^2}}, \quad \text{donde } r = \frac{\omega}{\omega_n}$$
        - **Ángulo de desfasaje entre fuerza y desplazamiento:**
          $$\phi = \arctan\left( \frac{2\zeta r}{1 - r^2} \right)$$
        """)

    with st.expander("2. Sistema de 2 Grados de Libertad (2 GDL)", expanded=True):
        st.markdown(r"""
        La formulación matricial para dos masas acopladas es:
        $$\mathbf{M}\ddot{\mathbf{x}} + \mathbf{C}\dot{\mathbf{x}} + \mathbf{K}\mathbf{x} = \mathbf{F}(t)$$

        Donde para el caso clásico entre dos paredes fijas:
        $$\mathbf{M} = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix}, \quad
          \mathbf{K} = \begin{bmatrix} k_1 + k_2 & -k_2 \\ -k_2 & k_2 + k_3 \end{bmatrix}, \quad
          \mathbf{C} = \begin{bmatrix} c_1 + c_2 & -c_2 \\ -c_2 & c_2 + c_3 \end{bmatrix}$$

        **Problema de Autovalores y Modos Normales:**
        Para vibración libre no amortiguada ($\mathbf{C} = \mathbf{0}, \mathbf{F} = \mathbf{0}$), asumiendo $\mathbf{x}(t) = \boldsymbol{\phi} e^{i\omega t}$:
        $$\det(\mathbf{K} - \omega^2 \mathbf{M}) = 0$$

        Esto genera una ecuación bicuadrática cuyas dos raíces reales positivas $\omega_{n1}$ y $\omega_{n2}$ son las frecuencias naturales del sistema acoplado.
        Para cada frecuencia natural se obtiene un modo característico de vibración (autovector $\boldsymbol{\phi}_1, \boldsymbol{\phi}_2$).
        """)

    with st.expander("3. Transformada de Fourier y Reconstrucción Espectral", expanded=True):
        st.markdown(r"""
        Cualquier señal temporal de oscilación $x(t)$ puede descomponerse en el dominio de la frecuencia mediante la Transformada de Fourier:
        $$X(\omega) = \int_{-\infty}^{\infty} x(t) e^{-i \omega t} dt$$

        En su versión discreta (FFT), obtenemos coeficientes complejos $X_k$ para cada frecuencia discreta $\omega_k$:
        $$x(t) \approx \bar{x} + \sum_{k=1}^N A_k \cos(\omega_k t + \phi_k)$$

        Al ordenar los componentes por su contenido energético ($A_k^2$) y retener los $N$ componentes principales, podemos observar cómo la aproximación converge rápidamente a la señal original a medida que aumenta $N$.
        """)
