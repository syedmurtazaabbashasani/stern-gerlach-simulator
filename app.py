"""
Interactive Sequential Stern–Gerlach Experiment
College physics project demonstration.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Model:
    P(s | r, n) = (1 + s r·n) / 2, s ∈ {-1, +1}
    After ideal projective measurement: r_new = s n

The default initial state is spin-up along the first analyzer axis.
All analyzers are ideal; detector imperfections and decoherence are omitted.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Plotly is used only for the optional interactive Bloch sphere.
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ------------------------------ Physics engine ------------------------------

def unit_vector(theta_deg: float, phi_deg: float = 0.0) -> np.ndarray:
    """Spherical angles in degrees: theta from +z; phi from +x in xy plane."""
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    v = np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ], dtype=float)
    return v / np.linalg.norm(v)


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError("Vectors must contain exactly three components.")
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("A zero vector is not a valid direction.")
    return v / norm


def probability(state: np.ndarray, axis: np.ndarray, outcome: int) -> float:
    """Born-rule probability for spin outcome +1 or -1 along axis."""
    if outcome not in (-1, +1):
        raise ValueError("Outcome must be +1 or -1.")
    p = (1.0 + outcome * float(np.dot(state, axis))) / 2.0
    return float(np.clip(p, 0.0, 1.0))


def sequence_theory(axes, initial_state):
    """Exact probability for each possible complete outcome sequence."""
    output = {}
    for sequence in itertools.product((+1, -1), repeat=len(axes)):
        state = initial_state.copy()
        p_sequence = 1.0
        for axis, result in zip(axes, sequence):
            p_sequence *= probability(state, axis, result)
            state = result * axis
        output[sequence] = p_sequence
    return output


def run_monte_carlo(axes, initial_state, n_particles, seed):
    """Sample N independent particles and return sequence counts/frequencies."""
    rng = np.random.default_rng(seed)
    all_sequences = list(itertools.product((+1, -1), repeat=len(axes)))
    counts = Counter({seq: 0 for seq in all_sequences})

    for _ in range(n_particles):
        state = initial_state.copy()
        sequence = []
        for axis in axes:
            p_plus = probability(state, axis, +1)
            result = +1 if rng.random() < p_plus else -1
            sequence.append(result)
            state = result * axis  # ideal projective state update
        counts[tuple(sequence)] += 1

    frequencies = {seq: counts[seq] / n_particles for seq in all_sequences}
    return dict(counts), frequencies


def transition_matrix(previous_axis, next_axis):
    """P(next outcome | previous outcome) for each pair of outcomes."""
    matrix = {}
    for previous in (+1, -1):
        state = previous * previous_axis
        matrix[previous] = {
            nxt: probability(state, next_axis, nxt)
            for nxt in (+1, -1)
        }
    return matrix


def label_sequence(seq):
    return "  ".join("↑" if x == 1 else "↓" for x in seq)


# ------------------------------ Visualizations ------------------------------

def draw_sequential_schematic(axes, angles, theory, counts, n_particles):
    """Illustrative apparatus schematic with branching paths at each analyzer."""
    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.set_xlim(-0.2, len(axes) * 3.0 + 0.4)
    ax.set_ylim(-2.0, 2.0)
    ax.axis("off")
    ax.set_title("SEQUENTIAL STERN–GERLACH APPARATUS",
                 fontsize=15, fontweight="bold", pad=12)

    ax.add_patch(FancyArrowPatch(
        (0.05, 0), (0.85, 0), arrowstyle="-|>", mutation_scale=15,
        linewidth=2.5, color="#263746"
    ))
    ax.text(0.02, 0.25, "Incoming\nspin beam", fontsize=9, ha="left", color="#263746")

    for i, (axis, angle) in enumerate(zip(axes, angles)):
        x = 0.95 + i * 3.0
        block = FancyBboxPatch(
            (x, -0.35), 0.92, 0.70,
            boxstyle="round,pad=0.07,rounding_size=0.08",
            facecolor="#EAF1FA", edgecolor="#315A8A", linewidth=1.8
        )
        ax.add_patch(block)
        ax.text(x + 0.46, 0.06, f"SG {i+1}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#17395E")
        ax.text(x + 0.46, -0.15, f"θ={angle:g}°", ha="center", va="center",
                fontsize=8, color="#315A8A")

        # For the first stage use the actual initial state; later stages show
        # the transition probabilities for a + outcome from the prior analyzer.
        if i == 0:
            p_up = probability(axes[0], axis, +1)
        else:
            p_up = probability(axes[i - 1], axis, +1)

        for outcome, y_end, p, color in [
            (+1, 1.05, p_up, "#167D9A"),
            (-1, -1.05, 1 - p_up, "#C65D3A"),
        ]:
            lw = 1.0 + 2.8 * math.sqrt(max(p, 0.0))
            ax.add_patch(FancyArrowPatch(
                (x + 0.92, 0), (x + 2.55, y_end),
                arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                color=color, alpha=0.92,
                connectionstyle="arc3,rad=0.0"
            ))
            arrow = "↑" if outcome == +1 else "↓"
            ax.text(x + 2.48, y_end + (0.16 if outcome == 1 else -0.18),
                    f"{arrow}  {p:.0%}", fontsize=9, fontweight="bold",
                    color=color, ha="right", va="center")

        if i < len(axes) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 2.55, 0), (x + 2.95, 0),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.4, color="#68737D"
            ))
    ax.text(0.01, -0.06,
            f"Conceptual branching schematic · N = {n_particles:,} simulated particles. "
            "Branch labels are illustrative adjacent-stage probabilities; full sequence probabilities are below.",
            transform=ax.transAxes, fontsize=8.5, color="#59636D", va="top")
    fig.tight_layout()
    return fig


def draw_probability_chart(theory, observed):
    sequences = list(theory.keys())
    labels = [label_sequence(s) for s in sequences]
    t = np.array([theory[s] for s in sequences])
    e = np.array([observed[s] for s in sequences])
    x = np.arange(len(sequences))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.1))
    ax.bar(x - width/2, t, width, label="Theoretical quantum probability",
           color="#277DA1", edgecolor="white")
    ax.bar(x + width/2, e, width, label="Monte Carlo frequency",
           color="#F8961E", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Complete measurement sequence (left to right = analyzer order)")
    ax.set_title("THEORETICAL vs MONTE CARLO PROBABILITY",
                 fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_ylim(0, max(0.05, float(max(t.max(initial=0), e.max(initial=0))) * 1.28))
    ax.legend(frameon=False, ncols=2)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1%%", padding=2, fontsize=7.5)
    fig.tight_layout()
    return fig


def draw_bloch_sphere(axes, initial_state):
    """Create an interactive Plotly Bloch sphere with analyzer and state vectors."""
    if not PLOTLY_AVAILABLE:
        return None

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 32)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z, opacity=0.13, showscale=False,
        colorscale=[[0, "#9EC5E8"], [1, "#9EC5E8"]],
        hoverinfo="skip", name="Bloch sphere"
    ))

    # Coordinate axes.
    for name, end, color in [
        ("x", (1, 0, 0), "#D1495B"),
        ("y", (0, 1, 0), "#278C6A"),
        ("z", (0, 0, 1), "#315A8A"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[-1, end[0]], y=[-1 if name == "y" else 0, end[1]],
            z=[-1 if name == "z" else 0, end[2]],
            mode="lines+text", line=dict(color=color, width=4),
            text=["", name], textposition="top center",
            name=f"{name}-axis"
        ))

    # Analyzer axes drawn as lines through the origin.
    for i, axis in enumerate(axes):
        fig.add_trace(go.Scatter3d(
            x=[-axis[0], axis[0]], y=[-axis[1], axis[1]],
            z=[-axis[2], axis[2]], mode="lines",
            line=dict(color="#7A5195", width=5, dash="dash"),
            name=f"Analyzer {i+1}"
        ))

    # Initial state Bloch vector.
    fig.add_trace(go.Scatter3d(
        x=[0, initial_state[0]], y=[0, initial_state[1]],
        z=[0, initial_state[2]], mode="lines+markers",
        line=dict(color="#F8961E", width=7),
        marker=dict(size=[2, 5], color="#F8961E"),
        name="Initial state"
    ))

    fig.update_layout(
        title="Bloch sphere: analyzer axes and initial spin state",
        height=520, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01),
        scene=dict(
            xaxis=dict(range=[-1.1, 1.1], title="x", backgroundcolor="white"),
            yaxis=dict(range=[-1.1, 1.1], title="y", backgroundcolor="white"),
            zaxis=dict(range=[-1.1, 1.1], title="z", backgroundcolor="white"),
            aspectmode="cube"
        )
    )
    return fig


# -------------------------------- Streamlit UI -------------------------------

st.set_page_config(
    page_title="Sequential Stern–Gerlach Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background-color: #F7F9FC;}
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102A43 0%, #244E73 58%, #347C91 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem; font-size: 2.1rem;}
    .hero p {color: #E3EEF7; margin-bottom: .1rem;}
    .small-note {color: #5B6770; font-size: .88rem;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E3EAF1;
        padding: 12px 14px; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚛️ Sequential Stern–Gerlach Lab</h1>
  <p>Interactive spin-½ measurement · Quantum probabilities · Monte Carlo experiment</p>
  <p>Explore how analyzer orientation changes measurement statistics in a sequence of ideal quantum measurements.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Experiment controls")
    n_particles = st.select_slider(
        "Number of particles (N)",
        options=[100, 500, 1000, 5000, 10000, 25000, 50000, 100000],
        value=10000,
        help="Larger samples usually give frequencies closer to theoretical probabilities."
    )
    seed = st.number_input("Random seed", min_value=0, max_value=2**31-1,
                           value=42, step=1)
    st.markdown("---")
    st.subheader("Analyzer orientations")
    n_analyzers = st.slider("Number of analyzers", min_value=2, max_value=5, value=3)
    st.caption("Angles are measured from the +z axis. Each analyzer can also have an azimuthal angle φ.")
    angle_rows = []
    default_thetas = [0, 60, 120, 45, 90]
    default_phis = [0, 0, 0, 45, 90]
    for i in range(n_analyzers):
        with st.expander(f"Analyzer {i+1}", expanded=(i < 3)):
            theta = st.slider(
                f"Polar angle θ{i+1} (°)", 0, 180,
                default_thetas[i], key=f"theta_{i+1}"
            )
            phi = st.slider(
                f"Azimuth φ{i+1} (°)", 0, 360,
                default_phis[i], key=f"phi_{i+1}"
            )
            angle_rows.append((theta, phi))
    st.markdown("---")
    initial_choice = st.radio(
        "Initial spin preparation",
        ["Spin-up along Analyzer 1", "Custom initial Bloch vector"],
        index=0
    )
    custom_state = None
    if initial_choice == "Custom initial Bloch vector":
        st.caption("Enter a vector; it will be normalized.")
        cx = st.number_input("Initial rₓ", value=0.0, min_value=-1.0, max_value=1.0, step=0.1)
        cy = st.number_input("Initial rᵧ", value=0.0, min_value=-1.0, max_value=1.0, step=0.1)
        cz = st.number_input("Initial r𝓏", value=1.0, min_value=-1.0, max_value=1.0, step=0.1)
        if abs(cx) + abs(cy) + abs(cz) > 0:
            custom_state = normalize([cx, cy, cz])
        else:
            st.warning("The zero vector is invalid. A +z initial state will be used.")
            custom_state = np.array([0.0, 0.0, 1.0])
    run = st.button("▶ Run Experiment", type="primary", use_container_width=True)

# Calculate orientations and initial state.
axes = [unit_vector(theta, phi) for theta, phi in angle_rows]
initial_state = axes[0].copy() if custom_state is None else custom_state

# Run initially once so the page has useful results; subsequent clicks rerun with controls.
if "last_run" not in st.session_state:
    st.session_state.last_run = None

if run or st.session_state.last_run is None:
    theory = sequence_theory(axes, initial_state)
    counts, observed = run_monte_carlo(axes, initial_state, int(n_particles), int(seed))
    st.session_state.last_run = {
        "angles": angle_rows,
        "axes": axes,
        "initial_state": initial_state,
        "N": int(n_particles),
        "seed": int(seed),
        "theory": theory,
        "counts": counts,
        "observed": observed,
    }

# If settings have changed since the last run, explain that the displayed results
# correspond to the last clicked experiment, avoiding accidental mismatch.
result = st.session_state.last_run
if result["angles"] != angle_rows or result["N"] != int(n_particles) or result["seed"] != int(seed) or not np.allclose(result["initial_state"], initial_state):
    st.info("Controls changed. Click **Run Experiment** to update the results.")

axes_r = result["axes"]
theory = result["theory"]
counts = result["counts"]
observed = result["observed"]
N = result["N"]
seqs = list(theory.keys())
labels = [label_sequence(s) for s in seqs]
errors = {s: abs(theory[s] - observed[s]) for s in seqs}
tv_distance = 0.5 * sum(errors.values())

# Summary metrics.
m1, m2, m3, m4 = st.columns(4)
m1.metric("Particles simulated", f"{N:,}")
m2.metric("Analyzers", str(len(axes_r)))
m3.metric("Possible sequences", str(2 ** len(axes_r)))
m4.metric("Total variation distance", f"{tv_distance:.4f}",
          help="Half the sum of absolute probability differences across all sequences.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Apparatus & probabilities", "📊 Probability analysis",
    "🧭 Bloch sphere", "📘 Physics model"
])

with tab1:
    st.subheader("Sequential measurement path")
    st.markdown(
        "Each analyzer has two possible outcomes, **spin-up (↑)** and **spin-down (↓)**. "
        "An ideal measurement updates the state before the next analyzer."
    )
    fig = draw_sequential_schematic(axes_r, [a[0] for a in result["angles"]],
                                    theory, counts, N)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Analyzer orientations")
    orientation_rows = []
    for i, (axis, (theta, phi)) in enumerate(zip(axes_r, result["angles"]), start=1):
        orientation_rows.append({
            "Analyzer": f"A{i}",
            "θ (°)": theta,
            "φ (°)": phi,
            "nₓ": axis[0],
            "nᵧ": axis[1],
            "n𝓏": axis[2],
        })
    st.dataframe(pd.DataFrame(orientation_rows).round(4), use_container_width=True, hide_index=True)

    st.subheader("Conditional transition probabilities")
    for i in range(len(axes_r) - 1):
        matrix = transition_matrix(axes_r[i], axes_r[i + 1])
        with st.expander(f"Analyzer {i+1} → Analyzer {i+2}", expanded=(i == 0)):
            rows = []
            for previous in (+1, -1):
                rows.append({
                    "Previous result": "↑" if previous == +1 else "↓",
                    "P(next ↑)": matrix[previous][+1],
                    "P(next ↓)": matrix[previous][-1],
                })
            st.dataframe(pd.DataFrame(rows).style.format({
                "P(next ↑)": "{:.4f}", "P(next ↓)": "{:.4f}"
            }), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Theoretical probability vs Monte Carlo frequency")
    fig = draw_probability_chart(theory, observed)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    table = pd.DataFrame([{
        "Sequence": label_sequence(s),
        "Count": counts[s],
        "Theoretical probability": theory[s],
        "Monte Carlo probability": observed[s],
        "Absolute error": errors[s],
    } for s in seqs])
    st.dataframe(table.style.format({
        "Theoretical probability": "{:.4%}",
        "Monte Carlo probability": "{:.4%}",
        "Absolute error": "{:.4%}",
    }), use_container_width=True, hide_index=True)

    csv_data = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download probability table (CSV)",
        data=csv_data,
        file_name="stern_gerlach_probabilities.csv",
        mime="text/csv"
    )

    # Empirical error overview.
    st.caption(
        "Monte Carlo frequencies fluctuate because outcomes are randomly sampled. "
        "For a fixed sequence with probability p, the standard deviation of its estimated "
        "frequency is approximately √[p(1−p)/N]."
    )

with tab3:
    st.subheader("Bloch sphere")
    st.markdown(
        "The Bloch vector represents a pure spin-½ state. Dashed purple lines show analyzer axes; "
        "the orange arrow shows the initial state."
    )
    if PLOTLY_AVAILABLE:
        fig_bloch = draw_bloch_sphere(axes_r, result["initial_state"])
        st.plotly_chart(fig_bloch, use_container_width=True)
    else:
        st.warning("Install Plotly to enable the interactive Bloch sphere: `pip install plotly`.")
    st.markdown("**Initial Bloch vector**")
    st.code("r = (" + ", ".join(f"{v:.4f}" for v in result["initial_state"]) + ")")

with tab4:
    st.subheader("Mathematical model")
    st.latex(r"P(s\\mid\\vec r,\\hat n)=\\frac{1+s(\\vec r\\cdot\\hat n)}{2},\\qquad s\\in\\{+1,-1\\}")
    st.write(
        "Here, r is the current Bloch vector and n is the unit vector along the analyzer axis. "
        "The outcome s=+1 denotes spin-up; s=−1 denotes spin-down."
    )
    st.latex(r"\\vec r_{\\mathrm{new}}=s\\hat n")
    st.write(
        "After an ideal projective measurement, the Bloch vector is updated to the eigenstate "
        "corresponding to the measured outcome. The next analyzer's probabilities are computed "
        "from this updated state."
    )
    st.latex(r"P(s_1,\\ldots,s_k)=P(s_1)\\prod_{j=2}^{k}P(s_j\\mid s_{j-1})")
    st.markdown("**Assumptions**")
    st.markdown("""
    - Spin-½ particles and two possible outcomes at every analyzer.
    - Ideal projective measurements and state update after each measurement.
    - No detector inefficiency, environmental decoherence, or magnetic-field imperfections.
    - The Monte Carlo trials are independent, conditional on the prepared initial state.
    - Analyzer axes are 3D unit vectors; θ is measured from +z and φ around the z-axis.
    """)
    st.success(
        f"Probability normalization check: theoretical sum = {sum(theory.values()):.8f}; "
        f"simulated sum = {sum(observed.values()):.8f}."
    )

st.markdown("---")
st.markdown(
    '<div class="small-note">Educational simulation · Idealized quantum model · '
    'Designed for a college physics project demonstration</div>',
    unsafe_allow_html=True
)
