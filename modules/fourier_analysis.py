import numpy as np
from scipy.signal import find_peaks

def compute_fft(t: np.ndarray, x: np.ndarray):
    """
    Calcula la Transformada Rápida de Fourier (FFT) unilateral de la señal x(t).
    
    Retorna:
    - freq_hz: Frecuencias en Hertz [Hz]
    - freq_rad: Frecuencias en radianes por segundo [rad/s]
    - amplitude: Amplitud unilateral [m o unidad de x]
    - phase_rad: Fase en radianes [-pi, pi]
    - phase_deg: Fase en grados [-180, 180]
    - power_pct: Porcentaje de potencia por frecuencia
    - dt: Intervalo de muestreo
    - fs: Frecuencia de muestreo
    """
    N = len(x)
    dt = float(t[1] - t[0])
    fs = 1.0 / dt

    # FFT estándar
    X_raw = np.fft.rfft(x)
    freq_hz = np.fft.rfftfreq(N, d=dt)
    freq_rad = freq_hz * 2.0 * np.pi

    # Amplitud física unilateral
    amplitude = (2.0 / N) * np.abs(X_raw)
    amplitude[0] = amplitude[0] / 2.0 # Componente DC

    # Fase
    phase_rad = np.angle(X_raw)
    phase_deg = np.degrees(phase_rad)

    # Espectro de potencia y energía
    power = amplitude**2
    total_power = np.sum(power)
    power_pct = (power / total_power * 100.0) if total_power > 1e-15 else np.zeros_like(power)

    return {
        "freq_hz": freq_hz,
        "freq_rad": freq_rad,
        "amplitude": amplitude,
        "phase_rad": phase_rad,
        "phase_deg": phase_deg,
        "power_pct": power_pct,
        "dt": dt,
        "fs": fs,
        "N": N,
        "X_raw": X_raw
    }

def reconstruct_fourier_top_n(t: np.ndarray, x: np.ndarray, num_components: int = 5):
    """
    Reconstruye la señal usando los N componentes espectrales más dominantes (con mayor amplitud),
    más la componente continua DC (valor medio).
    
    Permite ver exactamente cómo converge la aproximación al aumentar el número de armónicos/componentes.
    """
    fft_res = compute_fft(t, x)
    freq_hz = fft_res["freq_hz"]
    freq_rad = fft_res["freq_rad"]
    amplitude = fft_res["amplitude"]
    phase_rad = fft_res["phase_rad"]
    X_raw = fft_res["X_raw"]
    N = fft_res["N"]

    # Excluir la componente DC (índice 0) al seleccionar los N picos de oscilación
    ac_indices = np.arange(1, len(amplitude))
    ac_amps = amplitude[ac_indices]

    num_components = min(max(1, num_components), len(ac_indices))

    # Ordenar índices AC de mayor a menor amplitud
    sorted_order = np.argsort(ac_amps)[::-1]
    top_ac_indices = ac_indices[sorted_order[:num_components]]

    # Reconstrucción analítica armónica:
    # x_rec(t) = DC + sum_k A_k * cos(w_k * t + phi_k)
    # donde phi_k es ajustada para cos(w_k * t + phi_k)
    # Nota: la FFT de numpy usa sum x[n] exp(-i 2pi k n / N), por lo que:
    # x[n] = DC + sum (2/N)*|X_k| * cos(2pi k n / N + angle(X_k))
    dc_val = float(amplitude[0])
    x_rec = np.full_like(t, dc_val, dtype=float)

    components_info = []
    # Vector binario para marcar componentes seleccionados en el espectro
    selected_mask = np.zeros(len(freq_hz), dtype=bool)

    for rank, idx in enumerate(top_ac_indices, start=1):
        w_k = freq_rad[idx]
        f_k = freq_hz[idx]
        A_k = amplitude[idx]
        phi_k = phase_rad[idx]
        selected_mask[idx] = True

        harmonic_k = A_k * np.cos(w_k * (t - t[0]) + phi_k)
        x_rec += harmonic_k

        components_info.append({
            "rank": rank,
            "index": int(idx),
            "freq_hz": float(f_k),
            "freq_rad": float(w_k),
            "amplitude": float(A_k),
            "phase_deg": float(np.degrees(phi_k)),
            "phase_rad": float(phi_k),
            "power_pct": float(fft_res["power_pct"][idx])
        })

    # Error y métricas de ajuste
    residual = x - x_rec
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    
    var_x = float(np.var(x))
    if var_x > 1e-12:
        r2 = 1.0 - (mse / var_x)
    else:
        r2 = 1.0

    # Porcentaje de energía capturada por los componentes seleccionados
    total_ac_energy = np.sum(amplitude[1:]**2)
    selected_energy = np.sum([info["amplitude"]**2 for info in components_info])
    energy_captured_pct = (selected_energy / total_ac_energy * 100.0) if total_ac_energy > 1e-12 else 100.0

    return {
        "x_rec": x_rec,
        "residual": residual,
        "rmse": rmse,
        "r2": r2,
        "energy_captured_pct": energy_captured_pct,
        "components_info": components_info,
        "selected_mask": selected_mask,
        "fft_res": fft_res,
        "dc_val": dc_val
    }

def find_prominent_peaks(freq_rad: np.ndarray, amplitude: np.ndarray, height_thresh: float = 0.05, max_peaks: int = 5):
    """
    Encuentra los picos más prominentes del espectro para asociarlos a frecuencias naturales y de excitación.
    """
    max_amp = np.max(amplitude[1:]) if len(amplitude) > 1 else 1.0
    thresh = height_thresh * max_amp

    peaks, properties = find_peaks(amplitude[1:], height=thresh, distance=5)
    peaks = peaks + 1 # compensar índice 0

    if len(peaks) == 0:
        return []

    # Ordenar por altura
    peak_amps = amplitude[peaks]
    sorted_order = np.argsort(peak_amps)[::-1][:max_peaks]
    top_peaks = peaks[sorted_order]

    results = []
    for p in top_peaks:
        results.append({
            "freq_rad": float(freq_rad[p]),
            "freq_hz": float(freq_rad[p] / (2.0 * np.pi)),
            "amplitude": float(amplitude[p])
        })
    return results
