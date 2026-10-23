import numpy as np
from modules.system_1dof import System1DOF
from modules.system_2dof import System2DOF
from modules.fourier_analysis import compute_fft, reconstruct_fourier_top_n, find_prominent_peaks

def test_1dof():
    print("Testing 1DOF...")
    sys = System1DOF(m=2.0, k=50.0, c=1.5, F0=5.0, omega=4.0, x0=0.1, v0=0.0)
    assert np.isclose(sys.omega_n, 5.0)
    assert sys.zeta < 1.0
    sim = sys.simulate(t_max=5.0, num_points=500)
    assert len(sim["t"]) == 500
    assert len(sim["x"]) == 500
    forced = sys.forced_response_params()
    assert "X_steady" in forced
    print("1DOF OK!")

def test_2dof():
    print("Testing 2DOF...")
    sys2 = System2DOF(
        m1=2.0, m2=2.0, k1=40.0, k2=20.0, k3=40.0,
        c1=0.2, c2=0.2, c3=0.2,
        config_type="fixed-fixed",
        F01=2.0, F02=0.0, omega=4.0,
        x1_0=0.1, v1_0=0.0, x2_0=0.0, v2_0=0.0
    )
    modal = sys2.modal_analysis()
    assert modal["omega_n1"] < modal["omega_n2"]
    sim = sys2.simulate(t_max=5.0, num_points=500)
    assert len(sim["t"]) == 500
    assert len(sim["x1"]) == 500
    forced = sys2.forced_response_params()
    assert "phase_diff_deg" in forced
    print("2DOF OK!")

def test_fourier():
    print("Testing Fourier...")
    t = np.linspace(0, 10, 1000)
    x = 1.2 * np.cos(5.0 * t) + 0.6 * np.cos(10.0 * t + 0.5)
    fft_res = compute_fft(t, x)
    assert len(fft_res["freq_rad"]) == len(fft_res["amplitude"])
    recon = reconstruct_fourier_top_n(t, x, num_components=2)
    assert recon["r2"] > 0.98
    print(f"Fourier R^2 with 2 components: {recon['r2']:.4f}")
    print("Fourier OK!")

if __name__ == "__main__":
    test_1dof()
    test_2dof()
    test_fourier()
    print("ALL TESTS PASSED SUCCESSFULLY!")
