"""Compare the outputs of the Rust engine (output/) with those of the reference pipeline (reference/).

    python3 tests_ref/compare_outputs.py [output_dir] [reference_dir]

Deterministic quantities must agree to a relative tolerance (default 1e-8); Monte Carlo estimates must
agree within four combined standard errors; macros are listed where their printed values differ.
Exit status 1 if any check fails.
"""
import sys, json, glob, os, re, math

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
REF = sys.argv[2] if len(sys.argv) > 2 else "reference"
RTOL = 1e-8
fails, notes = [], []


def load(path):
    with open(path) as fh:
        return json.loads(fh.read().replace("NaN", "null"))


def jsonl(path):
    with open(path) as fh:
        return [json.loads(l.replace("NaN", "null")) for l in fh if l.strip()]


def close(a, b, rtol=RTOL, atol=0.0):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def compare_rows(name, ra, rb, keyf, fields, rtol=RTOL, atol=None):
    """Compare two lists of records field by field. `atol` maps a field to an absolute tolerance, for quantities
    that vanish in exact arithmetic at some points (such as the gap S_B - S_P without pull, which is round-off);
    the relative differences reported leave out values within a thousand such tolerances of zero."""
    atol = atol or {}
    da = {keyf(r): r for r in ra}
    db = {keyf(r): r for r in rb}
    if set(da) != set(db):
        fails.append(f"{name}: different points ({len(da)} vs {len(db)})")
        return
    worst, worst_abs = 0.0, {}
    for k in da:
        for f in fields:
            a, b = da[k].get(f), db[k].get(f)
            if isinstance(a, str) or isinstance(b, str):
                if a != b:
                    fails.append(f"{name} {k} {f}: {a} vs {b}")
                continue
            at = atol.get(f, 0.0)
            if not close(a, b, rtol, at):
                fails.append(f"{name} {k} {f}: {a} vs {b}")
                continue
            if a is None or b is None:
                continue
            if at > 0:
                worst_abs[f] = max(worst_abs.get(f, 0.0), abs(a - b))
            if max(abs(a), abs(b)) > 1e3 * at and max(abs(a), abs(b)) > 0:
                worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
    extra = "".join(f"; {f} within {v:.1e} absolute" for f, v in worst_abs.items())
    notes.append(f"{name}: {len(da)} points, largest relative difference {worst:.1e}{extra}")


G = lambda root: os.path.join(root, "data", "grid")
V = lambda root: os.path.join(root, "data", "validation")
gkey = lambda r: (r["lam"], r.get("theta"), r["eta"], round(r["nu"], 10))
GF = ["SB_fd", "SB_formula", "SP_fd", "SP_formula", "gap_fd", "VarU", "F", "Kurt", "ALO", "Etau", "EI2_B", "EI2_P"]

for fn in sorted(glob.glob(os.path.join(G(REF), "grid_lam*.json")) + glob.glob(os.path.join(G(REF), "variant_*.json"))):
    base = os.path.basename(fn)
    mine = os.path.join(G(OUT), base)
    if not os.path.exists(mine):
        fails.append(f"missing {base}")
        continue
    compare_rows(base, load(mine), load(fn), gkey, GF, rtol=1e-7, atol={"gap_fd": 1e-9})

sp = os.path.join(G(OUT), "speed_grid.json")
compare_rows("speed_grid.json", load(sp), load(os.path.join(G(REF), "speed_grid.json")), gkey,
             ["SB", "SP", "M1B", "M1P", "LB", "LP"], rtol=1e-7)

skey = lambda r: (r["theta"], r["eta"], r["nu"], r["lam"], r["reset"], r["gam"])
for ph in ("main", "extreme"):
    compare_rows(f"scan_{ph}.jsonl", jsonl(os.path.join(G(OUT), f"scan_{ph}.jsonl")), jsonl(os.path.join(G(REF), f"scan_{ph}.jsonl")),
                 skey, ["S", "var", "F", "kurt6F", "fc_type", "g_first", "min_re_khat"], rtol=1e-8)
    a = {skey(r): r for r in jsonl(os.path.join(G(OUT), f"scan_{ph}.jsonl"))}
    b = {skey(r): r for r in jsonl(os.path.join(G(REF), f"scan_{ph}.jsonl"))}
    bad = [k for k in a if (a[k]["kmin"] < -1e-10) != (b[k]["kmin"] < -1e-10)]
    if bad:
        fails.append(f"scan_{ph}: kernel sign differs at {bad[:3]}")

ua = {skey(dict(r, gam=1.0)): r for r in load(os.path.join(G(OUT), "unimodality.json"))}
ub = {skey(dict(r, gam=1.0)): r for r in load(os.path.join(G(REF), "unimodality.json"))}
diff = [k for k in ub if (ua[k]["worst_rel_increase"] > 1e-10) != (ub[k]["worst_rel_increase"] > 1e-10)]
(fails if diff else notes).append(f"unimodality.json: classification differs at {len(diff)} of {len(ub)} points")

ca, cb = load(os.path.join(V(OUT), "episode_solver_controls.json")), load(os.path.join(V(REF), "episode_solver_controls.json"))
for x, y in zip(ca, cb):
    for k in ("SB_fd", "SB_formula", "SP_fd", "VarU", "F", "Kurt", "ALO"):
        if not close(x["num"][k], y["num"][k], 1e-7, 1e-12):
            fails.append(f"controls lam={x['lam']} eta={x['eta']} {k}: {x['num'][k]} vs {y['num'][k]}")
    if not close(x["exact"]["S"], y["exact"]["S"], 1e-9):
        fails.append(f"controls exact S lam={x['lam']} eta={x['eta']}")
notes.append(f"episode_solver_controls.json: {len(ca)} controls compared")

for fn in ("stationary_solver_renewal_limit.json", "stationary_solver_benchmark.json"):
    for x, y in zip(load(os.path.join(V(OUT), fn)), load(os.path.join(V(REF), fn))):
        for q in ("S_res", "S_fd", "var", "F", "kurt_over_6F"):
            if not close(x["extrap"][q]["extrap"], y["extrap"][q]["extrap"], 1e-8):
                fails.append(f"{fn} {x['name']} {q}: {x['extrap'][q]['extrap']} vs {y['extrap'][q]['extrap']}")
    notes.append(f"{fn}: extrapolated values compared")

ra, rb = load(os.path.join(V(OUT), "running_refinement.json")), load(os.path.join(V(REF), "running_refinement.json"))
if not close(ra["S_extrap_Na"], rb["S_extrap_Na"], 1e-9):
    fails.append(f"running_refinement: {ra['S_extrap_Na']} vs {rb['S_extrap_Na']}")

ia = load(os.path.join(V(OUT), "record_identity.json"))
imax = max(max(max(x.values()) for x in e["identity"].values()) for e in ia)
(notes if imax < 1e-12 else fails).append(f"record_identity.json: largest identity residual {imax:.1e}")


def mc_close(name, a, sa, b, sb, k=4.0):
    z = abs(a - b) / math.hypot(sa, sb)
    (notes if z <= k else fails).append(f"{name}: {a:.5f} +- {sa:.5f} vs reference {b:.5f} +- {sb:.5f} (z = {z:.1f})")


ma, mb = load(os.path.join(V(OUT), "mc_benchmark.json")), load(os.path.join(V(REF), "mc_benchmark.json"))
for k in ("SB_formula", "SP_formula", "gap_formula"):
    if not close(ma["pde"][k], mb["pde"][k], 1e-7):
        fails.append(f"mc_benchmark pde {k}")
r1, r2 = ma["runs"][0], mb["runs"][0]
for k in ("S_B", "S_P", "S_fd_B", "S_fd_P", "gap_fd"):
    se = "se_" + k if "se_" + k in r1 else "se_" + k.replace("S_fd", "S_fd")
    mc_close("mc_benchmark " + k, r1[k], r1[se], r2[k], r2[se])

ka = load(os.path.join(V(OUT), "kernel_mc_comparison.json"))
for name, d in ka.items():
    (notes if d["rms_z"] < 3 else fails).append(f"kernel_mc {name}: rms z = {d['rms_z']:.2f}")

ia, ib = load(os.path.join(V(OUT), "identity_checks.json")), load(os.path.join(V(REF), "identity_checks.json"))
fa, fb = ia["fluct_response_calvo"], ib["fluct_response_calvo"]
mc_close("fluctuation-response (Calvo)", fa["mc"], fa["se"], fb["mc"], fb["se"])
mc_close("fluctuation-response vs exact", fa["mc"], fa["se"], fa["exact"], 0.0)
pa, pb = load(os.path.join(V(OUT), "perturbation.json")), load(os.path.join(V(REF), "perturbation.json"))
if not close(pa["prediction"], pb["prediction"], 1e-6):
    fails.append(f"perturbation prediction {pa['prediction']} vs {pb['prediction']}")
mc_close("perturbation derivative", pa["mc_derivative"], pa["mc_se"], pb["mc_derivative"], pb["mc_se"])


def macros(path):
    out = {}
    for l in open(path):
        m = re.match(r"\\def\\(\w+)\{(.*)\}\s*$", l.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


for fn in ("values.tex", "tipping_values.tex", "validation_values.tex"):
    a, b = macros(os.path.join(OUT, "tex", fn)), macros(os.path.join(REF, "tex", fn))
    d = [k for k in b if a.get(k) != b[k]]
    notes.append(f"{fn}: {len(b) - len(d)} of {len(b)} macros identical" + (": differ " + ", ".join(f"{k} ({a.get(k)} vs {b[k]})" for k in d) if d else ""))

print("\n".join(notes))
if fails:
    print("\nFAILURES:\n" + "\n".join(fails))
    sys.exit(1)
print("\nall checks passed")
