import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp

class System2DOF:
    """
    Modelo físico para un sistema de 2 Grados de Libertad (2 GDL).
    Ecuaciones de movimiento matriciales:
        M * x''(t) + C * x'(t) + K * x(t) = F(t)

    Configuraciones:
    - 'fixed-fixed': Pared - k1,c1 - m1 - k2,c2 - m2 - k3,c3 - Pared
    - 'fixed-free':  Pared - k1,c1 - m1 - k2,c2 - m2 - Libre (k3 = 0, c3 = 0)
    """
    def __init__(self, m1: float, m2: float, k1: float, k2: float, k3: float = 0.0,
                 c1: float = 0.0, c2: float = 0.0, c3: float = 0.0,
                 config_type: str = "fixed-fixed",
                 F01: float = 0.0, F02: float = 0.0, omega: float = 0.0,
                 x1_0: float = 0.0, v1_0: float = 0.0,
                 x2_0: float = 0.0, v2_0: float = 0.0):
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3) if config_type == "fixed-fixed" else 0.0
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3) if config_type == "fixed-fixed" else 0.0
        self.config_type = config_type

        self.F01 = float(F01)
        self.F02 = float(F02)
        self.omega = float(omega)

        self.x1_0 = float(x1_0)
        self.v1_0 = float(v1_0)
        self.x2_0 = float(x2_0)
        self.v2_0 = float(v2_0)

    @property
    def M_matrix(self) -> np.ndarray:
        """Matriz de Masa M (2x2)"""
        return np.array([
            [self.m1, 0.0],
            [0.0, self.m2]
        ], dtype=float)

    @property
    def K_matrix(self) -> np.ndarray:
        """Matriz de Rigidez K (2x2)"""
        return np.array([
            [self.k1 + self.k2, -self.k2],
            [-self.k2, self.k2 + self.k3]
        ], dtype=float)

    @property
    def C_matrix(self) -> np.ndarray:
        """Matriz de Amortiguamiento C (2x2)"""
        return np.array([
            [self.c1 + self.c2, -self.c2],
            [-self.c2, self.c2 + self.c3]
        ], dtype=float)

    def modal_analysis(self):
        """
        Calcula las frecuencias naturales y los modos de vibración (autovalores y autovectores)
        del sistema no amortiguado: K * phi = omega^2 * M * phi
        """
        eigenvals, eigenvecs = la.eigh(self.K_matrix, self.M_matrix)

        # Ordenar de menor a mayor frecuencia
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Frecuencias naturales [rad/s]
        omega_n = np.sqrt(np.maximum(eigenvals, 0.0))
        f_n = omega_n / (2.0 * np.pi)

        # Normalización de modos de vibración tal que el primer elemento sea 1.0
        phi_norm = np.copy(eigenvecs)
        for i in range(2):
            if np.abs(phi_norm[0, i]) > 1e-9:
                phi_norm[:, i] /= phi_norm[0, i]
            elif np.abs(phi_norm[1, i]) > 1e-9:
                phi_norm[:, i] /= phi_norm[1, i]

        return {
            "omega_n1": float(omega_n[0]),
            "omega_n2": float(omega_n[1]),
            "f_n1": float(f_n[0]),
            "f_n2": float(f_n[1]),
            "mode1": phi_norm[:, 0], # [1.0, ratio_1]
            "mode2": phi_norm[:, 1], # [1.0, ratio_2]
            "raw_eigenvecs": eigenvecs
        }

    def forced_response_params(self):
        """
        Calcula amplitudes teóricas de régimen permanente y ángulos de desfase
        mediante la matriz de impedancia dinámica Z(omega) = -omega^2 M + i*omega*C + K.
        """
        if self.omega <= 0.0 or (self.F01 == 0.0 and self.F02 == 0.0):
            return {
                "X1": 0.0, "phi1_deg": 0.0,
                "X2": 0.0, "phi2_deg": 0.0,
                "phase_diff_deg": 0.0
            }

        w = self.omega
        Z = - (w**2) * self.M_matrix + 1j * w * self.C_matrix + self.K_matrix
        F_vec = np.array([self.F01, self.F02], dtype=complex)

        try:
            X_complex = la.solve(Z, F_vec)
        except la.LinAlgError:
            # Resonancia exacta no amortiguada
            return {
                "X1": float('inf'), "phi1_deg": 0.0,
                "X2": float('inf'), "phi2_deg": 0.0,
                "phase_diff_deg": 0.0
            }

        X1 = float(np.abs(X_complex[0]))
        X2 = float(np.abs(X_complex[1]))
        phi1_rad = float(np.angle(X_complex[0]))
        phi2_rad = float(np.angle(X_complex[1]))

        phi1_deg = np.degrees(phi1_rad) % 360.0
        phi2_deg = np.degrees(phi2_rad) % 360.0
        diff_deg = (phi2_deg - phi1_deg) % 360.0

        return {
            "X1": X1,
            "phi1_deg": phi1_deg,
            "phi1_rad": phi1_rad,
            "X2": X2,
            "phi2_deg": phi2_deg,
            "phi2_rad": phi2_rad,
            "phase_diff_deg": diff_deg
        }

    def simulate(self, t_max: float = 20.0, num_points: int = 2500):
        """
        Simulación en el dominio del tiempo con RK45.
        y = [x1, x2, v1, v2]
        """
        t_eval = np.linspace(0.0, t_max, num_points)
        M_inv = la.inv(self.M_matrix)
        K = self.K_matrix
        C = self.C_matrix

        def odes(t, y):
            x1, x2, v1, v2 = y
            x_vec = np.array([x1, x2])
            v_vec = np.array([v1, v2])
            F_vec = np.array([
                self.F01 * np.cos(self.omega * t),
                self.F02 * np.cos(self.omega * t)
            ])
            # a_vec = M_inv * (F - C*v - K*x)
            a_vec = M_inv @ (F_vec - C @ v_vec - K @ x_vec)
            return [v1, v2, a_vec[0], a_vec[1]]

        y0 = [self.x1_0, self.x2_0, self.v1_0, self.v2_0]
        sol = solve_ivp(
            odes,
            t_span=(0.0, t_max),
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )

        t = sol.t
        x1 = sol.y[0]
        x2 = sol.y[1]
        v1 = sol.y[2]
        v2 = sol.y[3]

        F1 = self.F01 * np.cos(self.omega * t)
        F2 = self.F02 * np.cos(self.omega * t)

        return {
            "t": t,
            "x1": x1,
            "x2": x2,
            "v1": v1,
            "v2": v2,
            "F1": F1,
            "F2": F2
        }

    def frequency_response_sweep(self, w_max: float = 15.0, num_w: int = 500):
        """
        Barrido de frecuencia de excitación para graficar la respuesta en frecuencia de ambas masas.
        """
        w_vals = np.linspace(0.05, w_max, num_w)
        X1_vals = np.zeros(num_w)
        X2_vals = np.zeros(num_w)
        F_vec = np.array([self.F01 if self.F01 > 0 else 1.0, self.F02], dtype=complex)

        for i, w in enumerate(w_vals):
            Z = - (w**2) * self.M_matrix + 1j * w * self.C_matrix + self.K_matrix
            try:
                X_comp = la.solve(Z, F_vec)
                X1_vals[i] = np.abs(X_comp[0])
                X2_vals[i] = np.abs(X_comp[1])
            except la.LinAlgError:
                X1_vals[i] = np.nan
                X2_vals[i] = np.nan

        return w_vals, X1_vals, X2_vals
