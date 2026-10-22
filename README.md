# Sequential Stern–Gerlach Lab

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS, if activation is blocked by your shell, you can also install with
`python3 -m pip install -r requirements.txt` and run `python3 -m streamlit run app.py`.

## Features
- Particle-count and random-seed controls
- 2–5 analyzer orientations with polar angle θ and azimuth φ
- Optional custom initial Bloch vector
- Run Experiment button
- Theoretical and Monte Carlo probabilities for every full outcome sequence
- Sequential analyzer branching schematic
- Conditional transition-probability tables
- Interactive Bloch sphere (Plotly)
- CSV export of the probability table

## Physics assumptions
Spin-1/2 particles, ideal projective measurements, Born-rule probabilities,
state update after each measurement, no detector imperfections or decoherence.
