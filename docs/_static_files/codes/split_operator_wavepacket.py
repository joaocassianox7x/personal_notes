import pathlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def make_grid(n: int = 1024, x_max: float = 12.0) -> tuple[np.ndarray, float, np.ndarray]:
    """Step 1: Build a 1D periodic grid x and its FFT-compatible k grid."""
    x = np.linspace(-x_max, x_max, n, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    return x, dx, k


def harmonic_potential(x: np.ndarray, m: float, omega: float) -> np.ndarray:
    """Step 2: Define the potential V(x)."""
    return 0.5 * m * omega**2 * x**2


def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    """Step 3: Initialize a normalized Gaussian wavepacket."""
    psi = (1.0 / (np.pi * sigma**2) ** 0.25) * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    return psi


def normalize_l2(psi: np.ndarray, dx: float) -> np.ndarray:
    """Step 4: Normalize so that sum(|psi|^2) dx = 1."""
    norm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))
    return psi / norm


def split_operator_step(psi: np.ndarray, v: np.ndarray, k: np.ndarray, dt: float, hbar: float, m: float) -> np.ndarray:
    """One Strang-splitting time step: V(dt/2) -> FFT -> T(dt) -> IFFT -> V(dt/2)."""
    phase_v_half = np.exp(-1j * v * (dt / 2.0) / hbar)
    psi = phase_v_half * psi

    psi_k = np.fft.fft(psi)
    phase_t = np.exp(-1j * (hbar * k**2) * dt / (2.0 * m))
    psi_k = phase_t * psi_k
    psi = np.fft.ifft(psi_k)

    psi = phase_v_half * psi
    return psi


def main() -> None:
    # Units: set ħ = m = ω = 1 for a clean demo
    hbar = 1.0
    m = 1.0
    omega = 1.0

    # Step 1: grid
    x, dx, k = make_grid(n=1024, x_max=12.0)

    # Step 2: potential
    v = harmonic_potential(x, m=m, omega=omega)

    # Step 3: initial state
    psi = gaussian_wavepacket(x, x0=4.0, sigma=1.0, k0=0.0)
    psi = normalize_l2(psi, dx=dx)

    # Time stepping
    dt = 0.02
    steps = 420
    stride = 3  # render every few steps

    # Precompute a scaled potential curve for plotting context
    v_plot = v / np.max(v)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    (line,) = ax.plot([], [], color="#4C72B0", lw=2, label=r"$|\psi(x,t)|^2$")
    (vline,) = ax.plot([], [], color="#888888", lw=1.5, alpha=0.8, label=r"scaled $V(x)$")

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title("Split-operator propagation (harmonic potential)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Initialize artists
    def init():
        line.set_data([], [])
        vline.set_data(x, 0.7 * v_plot)
        return (line, vline)

    # Evolve and store frames on the fly
    psi_state = psi.copy()

    def update(frame_index: int):
        nonlocal psi_state
        for _ in range(stride):
            psi_state = split_operator_step(psi_state, v, k, dt, hbar, m)

        density = np.abs(psi_state) ** 2
        line.set_data(x, density)
        ax.set_title(f"Split-operator propagation (t={frame_index * stride * dt:.2f})")
        return (line, vline)

    frames = steps // stride
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=30, blit=True)

    out_path = IMG_DIR / "split_operator_wavepacket.gif"
    try:
        writer = animation.PillowWriter(fps=30)
        anim.save(out_path, writer=writer, dpi=110)
    except Exception:
        # Fallback: if Pillow isn't available, try ImageMagick.
        # If this also fails, the exception will show what dependency is missing.
        writer = animation.ImageMagickWriter(fps=30)
        anim.save(out_path, writer=writer, dpi=110)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
