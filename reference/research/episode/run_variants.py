"""Stronger history dependence than the pausing clock, to see how large S_B - S_P can be.
Variant 'dist':  processing accumulates with the distance from the reference, dA = |U| dt, and the
                 completion hazard depends on processing only: H = nu + eta (1 - exp(-gam A)) (chi = 1).
Variant 'sq':    dA = U^2 dt, same hazard.
Variant 'distthr_K': dA = |U| dt and H = nu + eta 1{A > K}: completion soon after accumulated distance passes K.
Completion therefore depends on the path only through its history (no direct state dependence).
Implementation: subclass of c9core.Model with alpha replaced and chi = 1, injected into c9run.
Usage: python3 run_variants.py VARIANT LAM ETA [ETA ...]   -> variants/<variant>_lam<LAM>.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c9core, c9run

VARIANT = sys.argv[1]
KAPPA = float(VARIANT.split('_')[1]) if VARIANT.startswith('distthr') else None   # e.g. distthr_1.0

class VModel(c9core.Model):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        u = self.u
        if VARIANT == 'dist':
            self.alpha = np.abs(u).copy()
        elif VARIANT == 'sq':
            self.alpha = (u * u).copy()
        elif VARIANT.startswith('distthr'):
            self.alpha = np.abs(u).copy()      # dA = |U| dt; hazard jumps when A exceeds KAPPA
        else:
            raise ValueError(VARIANT)
        self.chi = np.ones_like(u)
        if VARIANT.startswith('distthr'):
            bk = 1.0 - np.exp(-self.gam * KAPPA)
            self.H = np.ascontiguousarray(self.nu + self.eta * (self.b[:, None] > bk) * self.chi[None, :])
        else:
            self.H = np.ascontiguousarray(self.nu + self.eta * self.b[:, None] * self.chi[None, :])
        Nb = self.Nb
        vb = self.gam * (1.0 - (np.arange(Nb) + 1.0) * self.db)
        vb[-1] = 0.0
        self.R = np.ascontiguousarray(vb[:, None] * self.alpha[None, :] / self.db)
        self.Hmax = self.nu + self.eta * self.chi

c9run.Model = VModel

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x

if __name__ == "__main__":
    lam = float(sys.argv[2]); etas = [float(e) for e in sys.argv[3:]]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants", f"{VARIANT}_lam{lam}.json")
    done = json.load(open(out)) if os.path.exists(out) else []
    for eta in etas:
        if any(d['eta'] == eta for d in done):
            continue
        t0 = time.time()
        o = c9run.full_point(lam, 0.0, eta, 1.0)       # theta unused by the variant (alpha, chi replaced)
        o['variant'] = VARIANT
        done.append(clean(o)); json.dump(done, open(out, "w"), indent=1)
        print(f"{VARIANT} lam={lam} eta={eta}: S_B={o['SB_fd']:.6f}/{o['SB_formula']:.6f} S_P={o['SP_fd']:.6f}/{o['SP_formula']:.6f} "
              f"gap={o['gap_fd']:+.6f} rel={o['gap_fd']/o['SB_fd']:+.4%} VarU={o['VarU']:.6f} F={o['F']:.6f} ({time.time()-t0:.0f}s)", flush=True)
