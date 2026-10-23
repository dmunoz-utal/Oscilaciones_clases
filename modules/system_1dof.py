import numpy as np
from scipy.integrate import solve_ivp

class System1DOF:
    """
    Modelo físico para un sistema masa-resorte-amortiguador de 1 Grado de Libertad (1 GDL).
    Ecuación diferencial:
        m * x''(t) + c * x'(t) + k * x(t) = F(t)
    Donde:
        F(t) = F0 * cos(omega * t)
    """
    def __init__(self, m: float, k: float, c: float, F0: float = 0.0, omega: float = 0.0,
                 x0: float = 0.0, v0: float = 0.0):
        self.m = float(m)
        self.k = float(k)
        self.c = float(c)
        self.F0 = float(F0)
        self.omega = float(omega)
        self.x0 = float(x0)
        self.v0 = float(v0)

    @property
    def omega_n(self) -> float:
        """Frecuencia natural no amortiguada [rad/s]"""
        return np.sqrt(self.k / self.m)

    @property
    def f_n(self) -> float:
        """Frecuencia natural en Hertz [Hz]"""
        return self.omega_n / (2.0 * np.pi)

    @property
    def c_crit(self) -> float:
        """Coeficiente de amortiguamiento crítico [N*s/m]"""
        return 2.0 * np.sqrt(self.k * self.m)

    @property
    def zeta(self) -> float:
        """Razón de amortiguamiento adimensional"""
        return self.c / self.c_crit

    @property
    def omega_d(self) -> float:
        """Frecuencia natural amortiguada [rad/s]"""
        if self.zeta < 1.0:
            return self.omega_n * np.sqrt(1.0 - self.zeta**2)
        return 0.0

    @property
    def f_d(self) -> float:
        """Frecuencia natural amortiguada en Hertz [Hz]"""
        return self.omega_d / (2.0 * np.pi)

    @property
    def regime_label(self) -> str:
        """Nombre del régimen de amortiguamiento"""
        if np.isclose(self.zeta, 0.0, atol=1e-5):
            return "No amortiguado (ζ = 0)"
        elif self.zeta < 1.0:
            return f"Subamortiguado (0 < ζ < 1, ζ = {self.zeta:.3f})"
        elif np.isclose(self.zeta, 1.0, atol=1e-3):
            return f"Críticamente amortiguado (ζ ≈ 1.0)"
        else:
            return f"Sobreamortiguado (ζ > 1, ζ = {self.zeta:.3f})"

    @property
    def log_decrement(self) -> float:
        """Decremento logarítmico delta"""
        if 0.0 < self.zeta < 1.0:
            return (2.0 * np.pi * self.zeta) / np.sqrt(1.0 - self.zeta**2)
        return 0.0

    def forced_response_params(self):
        """
        Calcula parámetros teóricos del estado estacionario bajo excitación armónica:
        r = omega / omega_n
        Factor de amplificación dinámica M(r)
        Amplitud teórica permanente X
        Ángulo de desfase phi [rad] y [grados]
        """
        r = self.omega / self.omega_n if self.omega_n > 0 else 0.0
        denom = np.sqrt((1.0 - r**2)**2 + (2.0 * self.zeta * r)**2)
        if denom < 1e-12:
            M = float('inf')
            X = float('inf')
        else:
            M = 1.0 / denom
            delta_est = self.F0 / self.k if self.k > 0 else 0.0
            X = delta_est * M

        # Ángulo de desfasaje: phi = atan2(2*zeta*r, 1 - r^2)
        phi_rad = np.arctan2(2.0 * self.zeta * r, 1.0 - r**2)
        if phi_rad < 0:
            phi_rad += 2 * np.pi
        phi_deg = np.degrees(phi_rad)

        return {
            "r": r,
            "M": M,
            "X_steady": X,
            "phi_rad": phi_rad,
            "phi_deg": phi_deg
        }

    def simulate(self, t_max: float = 20.0, num_points: int = 2000):
        """
        Simula la respuesta temporal usando integración numérica precisa (RK45).
        """
        t_eval = np.linspace(0.0, t_max, num_points)

        def odes(t, y):
            x, v = y
            F_t = self.F0 * np.cos(self.omega * t)
            # x'' = (F(t) - c*v - k*x) / m
            a = (F_t - self.c * v - self.k * x) / self.m
            return [v, a]

        sol = solve_ivp(
            odes,
            t_span=(0.0, t_max),
            y0=[self.x0, self.v0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        t = sol.t
        x = sol.y[0]
        v = sol.y[1]
        F = self.F0 * np.cos(self.omega * t)
        a = (F - self.c * v - self.k * x) / self.m

        # Amplitud observada en régimen permanente (último 30% del tiempo)
        idx_perm = int(0.7 * len(t))
        x_tail = x[idx_perm:]
        X_max_observed = float(np.max(np.abs(x_tail))) if len(x_tail) > 0 else float(np.max(np.abs(x)))

        return {
            "t": t,
            "x": x,
            "v": v,
            "a": a,
            "F": F,
            "X_max_observed": X_max_observed
        }

    def frequency_response_curve(self, r_max: float = 3.0, num_r: int = 400):
        """
        Genera las curvas teóricas de magnitud y fase en función de r = omega / omega_n.
        """
        r_vals = np.linspace(0.01, r_max, num_r)
        z = self.zeta
        M_vals = 1.0 / np.sqrt((1.0 - r_vals**2)**2 + (2.0 * z * r_vals)**2)
        phi_vals = np.degrees(np.arctan2(2.0 * z * r_vals, 1.0 - r_vals**2))
        phi_vals = np.where(phi_vals < 0, phi_vals + 360.0, phi_vals)

        return r_vals, M_vals, phi_vals
