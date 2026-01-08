import pathlib
from typing import Callable, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Physical parameters
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 9.81

State = np.ndarray  # [theta1, theta2, omega1, omega2]


def deriv(state: State) -> State:
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2

    denom1 = l1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
    denom2 = l2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    ) / denom1

    domega2 = (
        2
        * np.sin(delta)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(delta)
        )
    ) / denom2

    return np.array([dtheta1, dtheta2, domega1, domega2])


def rk4_step(state: State, dt: float, f: Callable[[State], State]) -> State:
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(state0: State, t_final: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    steps = int(t_final / dt)
    states = np.zeros((steps, 4))
    times = np.linspace(0, t_final, steps)
    state = state0.copy()
    for i in range(steps):
        states[i] = state
        state = rk4_step(state, dt, deriv)
    return times, states


def angles_to_cartesian(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta1 = states[:, 0]
    theta2 = states[:, 1]
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def plot_geometry(theta1: float = np.pi / 3, theta2: float = np.pi / 3 + 0.5) -> None:
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")

    # Reference vertical
    ax.plot([0, 0], [0, -1.2], "k--", alpha=0.5, linewidth=1)

    # Rods and masses
    ax.plot([0, x1], [0, y1], "-o", color="#4C72B0", linewidth=2, markersize=8, label="Link 1")
    ax.plot([x1, x2], [y1, y2], "-o", color="#C44E52", linewidth=2, markersize=8, label="Link 2")

    # Angle arcs for theta1 and theta2 (from vertical)
    arc1 = patches.Arc((0, 0), 0.8, 0.8, angle=0, theta1=-90, theta2=np.degrees(theta1) - 90, color="#4C72B0")
    ax.add_patch(arc1)
    ax.text(0.35, -0.25, r"$\theta_1$", color="#4C72B0")

    arc2_center = (x1, y1)
    arc2 = patches.Arc(arc2_center, 0.8, 0.8, angle=0, theta1=-90, theta2=np.degrees(theta2) - 90, color="#C44E52")
    ax.add_patch(arc2)
    ax.text(x1 + 0.35, y1 - 0.25, r"$\theta_2$", color="#C44E52")

    # Length labels
    ax.text(x1 * 0.5 + 0.05, y1 * 0.5, r"$\ell_1, m_1$", color="#4C72B0")
    ax.text(x1 + (x2 - x1) * 0.5 + 0.05, y1 + (y2 - y1) * 0.5, r"$\ell_2, m_2$", color="#C44E52")

    ax.set_title("Double pendulum geometry and angles")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "double_pendulum_geometry.png", dpi=150)
    plt.close(fig)


def plot_divergence(times: np.ndarray, states_a: np.ndarray, states_b: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    dtheta1 = np.abs(states_a[:, 0] - states_b[:, 0])
    dtheta2 = np.abs(states_a[:, 1] - states_b[:, 1])
    plt.semilogy(times, dtheta1, label=r"|theta1_A - theta1_B|")
    plt.semilogy(times, dtheta2, label=r"|theta2_A - theta2_B|")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle difference (rad, log scale)")
    plt.title("Divergence of nearly identical initial conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "double_pendulum_divergence.png", dpi=150)
    plt.close()


def make_animation(
    times: np.ndarray,
    coords_a: Tuple[np.ndarray, ...],
    coords_b: Tuple[np.ndarray, ...],
    stride: int = 5,
    fps: int = 30,
) -> None:
    x1a, y1a, x2a, y2a = coords_a
    x1b, y1b, x2b, y2b = coords_b

    idx = np.arange(0, len(times), stride)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_title("Double pendulum: sensitivity to initial conditions")

    line_a, = ax.plot([], [], "o-", lw=2, color="#4C72B0", label="Pendulum A")
    line_b, = ax.plot([], [], "o-", lw=2, color="#C44E52", label="Pendulum B")
    trail_a, = ax.plot([], [], "-", lw=1, alpha=0.4, color="#4C72B0")
    trail_b, = ax.plot([], [], "-", lw=1, alpha=0.4, color="#C44E52")
    ax.legend(loc="upper right")

    trail_len = 120

    def init():
        line_a.set_data([], [])
        line_b.set_data([], [])
        trail_a.set_data([], [])
        trail_b.set_data([], [])
        return line_a, line_b, trail_a, trail_b

    def update(frame):
        f = idx[frame]
        xa = [0, x1a[f], x2a[f]]
        ya = [0, y1a[f], y2a[f]]
        xb = [0, x1b[f], x2b[f]]
        yb = [0, y1b[f], y2b[f]]

        line_a.set_data(xa, ya)
        line_b.set_data(xb, yb)

        start = max(0, f - trail_len)
        trail_a.set_data(x2a[start:f], y2a[start:f])
        trail_b.set_data(x2b[start:f], y2b[start:f])
        return line_a, line_b, trail_a, trail_b

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(idx), interval=1000 / fps, blit=True)

    gif_path = IMG_DIR / "double_pendulum.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    t_final = 20.0
    dt = 0.01

    state_a = np.array([np.pi / 2, np.pi / 2 + 0.01, 0.0, 0.0])
    state_b = np.array([np.pi / 2, np.pi / 2 + 0.011, 0.0, 0.0])

    plot_geometry()
    times, states_a = simulate(state_a, t_final, dt)
    _, states_b = simulate(state_b, t_final, dt)

    coords_a = angles_to_cartesian(states_a)
    coords_b = angles_to_cartesian(states_b)

    plot_divergence(times, states_a, states_b)
    make_animation(times, coords_a, coords_b, stride=5, fps=30)

    # Quick textual check
    print("Final angles A (rad):", states_a[-1, :2])
    print("Final angles B (rad):", states_b[-1, :2])
    print("Saved:", IMG_DIR / "double_pendulum.gif")


if __name__ == "__main__":
    main()
