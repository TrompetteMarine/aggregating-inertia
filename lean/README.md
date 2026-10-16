# Lean formalisation of Section 3

This folder formalises in [Lean 4](https://lean-lang.org) with [Mathlib](https://github.com/leanprover-community/mathlib4)
the analytic core of Section 3 of *Aggregating Inertia: Tipping, Slow Shifts and What Revision Records Reveal*:
the results about the roots of the characteristic equation

```
1 = g · k̂(z),     k̂(z) = ∫₀^∞ e^{-zt} k(t) dt,     k̂(0) = S,
```

for a given response kernel `k`. Lean's kernel checks every proof; there is no `sorry` and no axiom beyond Lean's
standard three (`propext`, `Classical.choice`, `Quot.sound`).

## What is checked

| Paper | Lean (`AggregatingInertia/…`) | Statement |
|---|---|---|
| Proposition 1(a) | `prop1a_real_root`, `prop1a_root_zero` (`Threshold.lean`) | for an integrable kernel of any sign and `g > 0`: if `gS > 1` there is a real root `x* > 0`; if `gS = 1`, `z = 0` is a root |
| Theorem 1(a) | `thm1a_stable` | `k ≥ 0`, not a.e. zero, with an exponential moment: if `gS < 1`, every root has `Re z < 0` |
| Theorem 1(b) | `thm1b_only_root`, `thm1b_simple` | if `gS = 1`, the only root with `Re z ≥ 0` is `0`, and it is simple: the derivative of `1 - g k̂` at `0` is `g ∫ t k(t) dt > 0` |
| Theorem 1(c) | `thm1c_unstable` | if `gS > 1`, there is a real root `x* > 0`, unique on `[0, ∞)`; every root has `Re z ≤ x*`, with equality only at `z = x*` |
| Example after Proposition 1, at `g = 7/2` | `example_unstable_below_threshold`, `laplace_exampleKernel`, `cumResp_exampleKernel` (`Example.lean`), `counterexample_root` (`Threshold.lean`) | for `k(t) = (3/2)e^{-2t} - (1/2)e^{-t}`: `k̂(z) = 3/(2(z+2)) - 1/(2(z+1))` for `Re z > -1`, `S = 1/4`, and at `g = 7/2`, where `gS = 7/8 < 1`, the root `(1 + i√3)/4` has positive real part |
| (hypotheses are satisfiable) | `kernelHyp_exp` | `k(t) = e^{-t}` meets the hypotheses of Theorem 1 with `β = 1/2` |
| Theorem 2(a), last step | `thm2a_lag` (`Speed.lean`) | the algebra from `k̂_g = k̂/(1 - g k̂)` to (lagagg): for any `K` with `K(0) = S`, `K'(0) = -S L̄` and `gS ≠ 1`, `K_g = K/(1 - gK)` has `K_g(0) = S/(1 - gS)` and `-K_g'(0)/K_g(0) = L̄/(1 - gS)` |
| Theorem 2(b), roots | `thm2b_unique`, `thm2b_simple`, `thm2b_simple_complex`, `thm2b_dominant`, `thm2b_exists_of_ge`, `thm2b_exists_near` | the real root `x_g > -β` is unique and simple (the complex derivative of `k̂` at `x_g` is `-∫ t e^{-x_g t} k(t) dt < 0`), every other root has `Re z < x_g`, and the root exists when `gS ≥ 1` and when `1/g ∈ (S, k̂(-β/2)]`, a non-empty range of `gS < 1` |
| Theorem 2(c), first order | `thm2c_slowing` (`SlowingDown.lean`) | the root is a function of the gain near `g = 1/S`, vanishing there, with derivative `S/L̄`: `x_g = (gS - 1)/L̄ + o(gS - 1)` |

The facts about the Laplace transform that these proofs use are in `Laplace.lean`: `‖k̂(x + iy)‖ ≤ k̂(x)` for
`k ≥ 0`, strictly when `y ≠ 0` (the set where `e^{-iyt}` takes a given value is countable); continuity, positivity
and strict decrease of `k̂` on the real axis, and its limit at infinity; and complex differentiability to the right
of an abscissa where `k` has an exponential moment.

## What is not checked

* The model behind the kernel: Lemma 1 (linear response), Proposition 2 (nonnegative kernels), and all of Sections 4
  and 6 (Theorems 3–4, Propositions 4–9) rest on stochastic calculus, point processes and Fokker–Planck equations,
  which Mathlib does not yet provide.
* The Paley–Wiener step from the roots of the characteristic equation to linear stability, and the resolvent
  equation that gives `k̂_g = k̂/(1 - g k̂)` in Theorem 2(a), are taken as given.
* Proposition 1(b) (the social multiplier), the resolvent and moment identities for `k_g` in Theorem 2(a), the
  integrability of `e^{-xt}Φ` in Theorem 2(b), the `O((gS - 1)²)` remainder in Theorem 2(c), Proposition 3 (slow
  shift or tipping), the part of Lemma 4 beyond analyticity of `k̂` (`hasDerivAt_laplace` gives that part), and, in
  the example, the crossing of the imaginary axis at `g = 3` (only `g = 7/2` is checked).

## Build

```sh
cd lean
lake exe cache get   # prebuilt Mathlib
lake build
```

Lean 4.35.0 and Mathlib v4.35.0 (`lean-toolchain`, `lakefile.toml`, `lake-manifest.json`). The GitHub Actions
workflow builds this folder on every push and checks that every theorem listed in `AggregatingInertia/Axioms.lean` uses only the three standard axioms.
