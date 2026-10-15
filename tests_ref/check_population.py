"""An independent numpy implementation of the engine's population simulation (extras: `Pop::run`), and how the
settled averages depend on the time step.

    python3 tests_ref/check_population.py [output_dir]

For both hazards and the two gains above the threshold, it simulates 50,000 agents for 60 time units after a
common displacement of 0.05, as the engine does, at time steps 0.01 (the engine's), 0.005 and 0.0025, and
compares the average over the last 20 time units with the engine's (same step, other random numbers) and with
the stable stationary state of the continuous-time model. About eight minutes on two cores.
"""
import csv, json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
NU, ZETA, ELL, C, LAM = 0.5, 3.0, 0.7, 1.5, 1.0     # pipeline::figures::GHF_* and the pull
META = json.load(open(os.path.join(OUT, "data", "extras", "population_meta.json")))
REC = json.load(open(os.path.join(OUT, "data", "records.json")))["bifurcation"]


def hazard(name, u):
    e = np.exp(-u * u / (2 * ELL * ELL))
    return NU + ZETA * (1 - e) if name == "up" else NU + ZETA * e


def column(path, name):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r[name]) for r in rows])


def stable_level(name, gs):
    path = os.path.join(OUT, "data", f"tip_branch_{name}.csv")
    g, m, sl = column(path, "gS"), column(path, "m"), column(path, "slope")
    keep = (sl < 1) & (m > (REC["down"]["fold_m"] if name == "down" else 0.0))
    o = np.argsort(g[keep])
    return float(np.interp(gs, g[keep][o], m[keep][o]))


def settled(name, gs, dt, n=50_000, t_end=60.0, m0=0.05, seed=1):
    rng = np.random.default_rng(seed)
    ea, drift = np.exp(-LAM * dt), -np.expm1(-LAM * dt) / LAM
    sd = np.sqrt(-np.expm1(-2 * LAM * dt) / (2 * LAM))
    u = np.zeros(n)
    for _ in range(int(round(20.0 / dt))):                 # burn-in without feedback
        reset = rng.random(n) < -np.expm1(-hazard(name, u) * dt)
        u = np.where(reset, 0.0, u * ea + sd * rng.standard_normal(n))
    u += m0
    g, m, ms = gs / META[name]["S"], u.mean(), []
    for _ in range(int(round(t_end / dt))):
        f = g * C * np.tanh(m / C)
        reset = rng.random(n) < -np.expm1(-hazard(name, u) * dt)
        u = np.where(reset, 0.0, u * ea + f * drift + sd * rng.standard_normal(n))
        m = u.mean()
        ms.append(m)
    t = dt * np.arange(1, len(ms) + 1)
    return float(np.mean(np.array(ms)[t >= t_end - 20.0]))


worst_engine, worst_fine = 0.0, 0.0
for name in ("up", "down"):
    pop = os.path.join(OUT, "data", "extras", f"population_{name}.csv")
    t = column(pop, "t")
    for gs in (1.05, 1.2):
        level = stable_level(name, gs)
        engine = float(np.mean(column(pop, f"gS{gs}")[t >= 40.0]))
        here = {dt: settled(name, gs, dt) for dt in (0.01, 0.005, 0.0025)}
        worst_engine = max(worst_engine, abs(engine - level) / level)
        worst_fine = max(worst_fine, abs(here[0.0025] - level) / level)
        print(f"{name:4s} gS={gs}: stationary state {level:.4f}; engine (dt 0.01) {engine:.4f}; here "
              + ", ".join(f"dt {dt}: {v:.4f}" for dt, v in here.items()), flush=True)
print(f"largest relative gap to the stationary state: engine {100 * worst_engine:.1f}%, "
      f"here at dt 0.0025 {100 * worst_fine:.1f}%")
