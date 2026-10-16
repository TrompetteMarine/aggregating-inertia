/-
Copyright (c) 2026 Gabriel L. S. Bontemps. Released under the MIT licence.

# Location of the tipping point: Proposition 1(a) and Theorem 1

The characteristic equation of the paper is `1 = g * k̂ z` (equation (char)), with
`k̂ z = ∫₀^∞ exp (-z t) k t dt` and `k̂ 0 = S`. Linear stability of consensus means that no root
has `Re z ≥ 0` (Paley–Wiener); that step, like the derivation of the renewal equation from the
model (Lemma 1), is not formalised here. What is formalised are the statements about the roots.

In Lean the Bochner integral of a non-integrable function is `0`, so `laplace k z` is defined for
every `z`; where the paper speaks of the roots of (char), the statements below quantify over all
`z : ℂ` and remain true, because `g * 0 ≠ 1`.
-/
import AggregatingInertia.Laplace

open MeasureTheory Set Filter Topology Function

noncomputable section

namespace AggregatingInertia

variable {k : ℝ → ℝ}

/-- The hypotheses of Theorem 1 on the kernel: `k ≥ 0` on `(0, ∞)`, not almost everywhere zero,
with an exponential moment `∫₀^∞ exp (β t) k t dt < ∞` for some `β > 0`. -/
structure KernelHyp (k : ℝ → ℝ) (β : ℝ) : Prop where
  beta_pos : 0 < β
  nonneg : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t
  not_ae_zero : 0 < volume (support k ∩ Ioi (0 : ℝ))
  exp_moment : IntegrableOn (fun t => Real.exp (β * t) * k t) (Ioi 0)

namespace KernelHyp

variable {β : ℝ}

/-- The exponential moment in the weighted form used in `Laplace.lean`, at abscissa `-β`. -/
lemma weighted (h : KernelHyp k β) :
    IntegrableOn (fun t => Real.exp (-((-β) * t)) * k t) (Ioi 0) := by
  simpa using h.exp_moment

/-- Integrability of `exp (-a t) k t` for every `a ≥ -β`. -/
lemma weighted_of_le (h : KernelHyp k β) {a : ℝ} (ha : -β ≤ a) :
    IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0) :=
  integrableOn_weighted_of_le ha h.weighted

lemma weighted_zero (h : KernelHyp k β) :
    IntegrableOn (fun t => Real.exp (-((0 : ℝ) * t)) * k t) (Ioi 0) :=
  h.weighted_of_le (by linarith [h.beta_pos])

lemma integrableOn (h : KernelHyp k β) : IntegrableOn k (Ioi 0) := by
  simpa using h.weighted_zero

lemma cumResp_pos (h : KernelHyp k β) : 0 < cumResp k := by
  rw [← laplaceR_zero]
  exact laplaceR_pos le_rfl h.weighted_zero h.nonneg h.not_ae_zero

lemma strictAntiOn (h : KernelHyp k β) : StrictAntiOn (laplaceR k) (Ici (-β)) :=
  laplaceR_strictAntiOn h.weighted h.nonneg h.not_ae_zero

/-- `t k t` is integrable: `t ≤ exp (β t) / β`. -/
lemma integrableOn_mul_id (h : KernelHyp k β) : IntegrableOn (fun t => t * k t) (Ioi 0) := by
  have hb : IntegrableOn
      (fun t : ℝ => (t * Real.exp (-(β * t))) * (Real.exp (β * t) * k t)) (Ioi 0) := by
    refine Integrable.bdd_mul (c := β⁻¹) h.exp_moment ?_ ?_
    · exact (continuous_id.mul (Real.continuous_exp.comp
        ((continuous_const.mul continuous_id).neg))).aestronglyMeasurable
    · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
      have ht' : (0 : ℝ) < t := ht
      have hβ := h.beta_pos
      rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ht'.le (Real.exp_pos _).le)]
      have h1 : β * t ≤ Real.exp (β * t) := le_exp_of_nonneg
      have h2 : Real.exp (β * t) * Real.exp (-(β * t)) = 1 := by
        rw [← Real.exp_add]; simp
      rw [inv_eq_one_div, le_div_iff₀ hβ]
      calc t * Real.exp (-(β * t)) * β = (β * t) * Real.exp (-(β * t)) := by ring
        _ ≤ Real.exp (β * t) * Real.exp (-(β * t)) := by gcongr
        _ = 1 := h2
  refine hb.congr (Eventually.of_forall fun t => ?_)
  simp only
  rw [show t * Real.exp (-(β * t)) * (Real.exp (β * t) * k t) =
      t * k t * (Real.exp (-(β * t)) * Real.exp (β * t)) by ring, ← Real.exp_add]
  simp

lemma firstMoment_pos (h : KernelHyp k β) : 0 < firstMoment k :=
  setIntegral_pos_of_weight (w := id) (N := ∅) h.nonneg h.not_ae_zero
    (fun _ ht => le_of_lt ht) (fun _ ht _ => ht) measure_empty h.integrableOn_mul_id

end KernelHyp

/-! ### Proposition 1(a): a real root when `gS > 1`, for kernels of any sign -/

/-- **Proposition 1(a)**, first claim. For an integrable kernel of any sign and `g > 0`, if
`gS > 1` then the characteristic equation has a real root `x* > 0`. -/
theorem prop1a_real_root (hint : IntegrableOn k (Ioi 0)) {g : ℝ} (hg : 0 < g)
    (hgS : 1 < g * cumResp k) : ∃ x : ℝ, 0 < x ∧ g * laplace k x = 1 := by
  have ha : IntegrableOn (fun t => Real.exp (-((0 : ℝ) * t)) * k t) (Ioi 0) := by
    simpa using hint
  have hcont := continuousOn_laplaceR ha
  have hlim := tendsto_laplaceR_atTop ha
  have hginv : 0 < 1 / g := by positivity
  obtain ⟨X, hX, hX0⟩ :=
    ((hlim.eventually (gt_mem_nhds hginv)).and (eventually_ge_atTop (0 : ℝ))).exists
  have hS : 1 / g < laplaceR k 0 := by
    rw [laplaceR_zero, div_lt_iff₀ hg]
    linarith
  obtain ⟨x, hx, hfx⟩ :=
    intermediate_value_Icc' hX0 (hcont.mono Icc_subset_Ici_self) ⟨hX.le, hS.le⟩
  refine ⟨x, ?_, ?_⟩
  · rcases eq_or_lt_of_le hx.1 with h0 | h0
    · rw [← h0] at hfx
      linarith
    · exact h0
  · rw [laplace_ofReal, hfx]
    push_cast
    field_simp [(show (g : ℂ) ≠ 0 by exact_mod_cast hg.ne')]

/-- **Proposition 1(a)**, second claim: if `gS = 1`, then `z = 0` is a root. -/
theorem prop1a_root_zero {g : ℝ} (hgS : g * cumResp k = 1) : g * laplace k 0 = 1 := by
  rw [laplace_zero]
  exact_mod_cast hgS

/-! ### Theorem 1: the static threshold -/

variable {β : ℝ}

/-- For `Re z ≥ 0`, `‖k̂ z‖ ≤ S` (`k ≥ 0`). -/
lemma norm_laplace_le_cumResp (h : KernelHyp k β) {z : ℂ} (hz : 0 ≤ z.re) :
    ‖laplace k z‖ ≤ cumResp k := by
  refine (norm_laplace_le h.nonneg z).trans ?_
  rw [← laplaceR_zero]
  exact h.strictAntiOn.antitoneOn (by simp [h.beta_pos.le])
    (by simp only [mem_Ici]; linarith [h.beta_pos]) hz

/-- A root `z` of the characteristic equation satisfies `1 = g ‖k̂ z‖`. -/
lemma norm_eq_of_root {g : ℝ} (hg : 0 < g) {z : ℂ} (hz : g * laplace k z = 1) :
    g * ‖laplace k z‖ = 1 := by
  have := congrArg (‖·‖) hz
  simpa [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hg] using this

/-- **Theorem 1(a)**. If `gS < 1`, every root of the characteristic equation has `Re z < 0`. -/
theorem thm1a_stable (h : KernelHyp k β) {g : ℝ} (hg : 0 < g) (hgS : g * cumResp k < 1) :
    ∀ z : ℂ, g * laplace k z = 1 → z.re < 0 := by
  intro z hz
  by_contra hre
  push Not at hre
  have h1 := norm_laplace_le_cumResp h hre
  have h2 := norm_eq_of_root hg hz
  have : g * ‖laplace k z‖ ≤ g * cumResp k := mul_le_mul_of_nonneg_left h1 hg.le
  linarith

/-- **Theorem 1(b)**, location. If `gS = 1`, the only root with `Re z ≥ 0` is `z = 0`. -/
theorem thm1b_only_root (h : KernelHyp k β) {g : ℝ} (hg : 0 < g) (hgS : g * cumResp k = 1) :
    ∀ z : ℂ, 0 ≤ z.re → g * laplace k z = 1 → z = 0 := by
  intro z hre hz
  have h2 := norm_eq_of_root hg hz
  have hβ := h.beta_pos
  have hmem0 : (0 : ℝ) ∈ Ici (-β) := by simp only [mem_Ici]; linarith
  have hmemz : z.re ∈ Ici (-β) := by simp only [mem_Ici]; linarith
  -- the imaginary part vanishes
  have him : z.im = 0 := by
    by_contra hy
    have h1 := norm_laplace_lt (a := -β) (by linarith) h.weighted h.nonneg h.not_ae_zero hy
    have h3 : laplaceR k z.re ≤ laplaceR k 0 := h.strictAntiOn.antitoneOn hmem0 hmemz hre
    rw [laplaceR_zero] at h3
    have : g * ‖laplace k z‖ < g * cumResp k := mul_lt_mul_of_pos_left (h1.trans_le h3) hg
    linarith
  -- and so does the real part
  have hre0 : z.re = 0 := by
    by_contra hx
    have hpos : 0 < z.re := lt_of_le_of_ne hre (Ne.symm hx)
    have h1 := norm_laplace_le h.nonneg z
    have h3 : laplaceR k z.re < laplaceR k 0 := h.strictAntiOn hmem0 hmemz hpos
    rw [laplaceR_zero] at h3
    have : g * ‖laplace k z‖ < g * cumResp k := mul_lt_mul_of_pos_left (h1.trans_lt h3) hg
    linarith
  exact Complex.ext hre0 him

/-- `k̂' 0 = -∫₀^∞ t k t dt`. -/
lemma hasDerivAt_laplace_zero (h : KernelHyp k β) :
    HasDerivAt (laplace k) (-(firstMoment k : ℂ)) 0 := by
  have hd := hasDerivAt_laplace (a := -β) (z₀ := 0) (by simp [h.beta_pos]) h.weighted
  convert hd using 1
  rw [integral_neg, firstMoment, ← integral_complex_ofReal]
  congr 1
  refine integral_congr_ae (Eventually.of_forall fun t => ?_)
  simp

/-- **Theorem 1(b)**, simplicity. The root `z = 0` is simple: the derivative of
`z ↦ 1 - g k̂ z` at `0` is `g ∫₀^∞ t k t dt`, which is positive. -/
theorem thm1b_simple (h : KernelHyp k β) {g : ℝ} (hg : 0 < g) :
    HasDerivAt (fun z => 1 - g * laplace k z) ((g * firstMoment k : ℝ) : ℂ) 0 ∧
      0 < g * firstMoment k := by
  refine ⟨?_, mul_pos hg h.firstMoment_pos⟩
  have hd := ((hasDerivAt_laplace_zero h).const_mul (g : ℂ)).const_sub 1
  convert hd using 1
  push_cast
  ring

/-- **Theorem 1(c)**. If `gS > 1`, there is a real root `x* > 0`; it is the only real root in
`[0, ∞)`, every root satisfies `Re z ≤ x*`, and `Re z = x*` only at `z = x*`. -/
theorem thm1c_unstable (h : KernelHyp k β) {g : ℝ} (hg : 0 < g) (hgS : 1 < g * cumResp k) :
    ∃ xs : ℝ, 0 < xs ∧ g * laplace k xs = 1 ∧
      (∀ x : ℝ, 0 ≤ x → g * laplace k x = 1 → x = xs) ∧
      ∀ z : ℂ, g * laplace k z = 1 → z.re ≤ xs ∧ (z.re = xs → z = xs) := by
  obtain ⟨xs, hxs, hroot⟩ := prop1a_real_root h.integrableOn hg hgS
  have hβ := h.beta_pos
  have hreal : g * laplaceR k xs = 1 := by
    rw [laplace_ofReal] at hroot
    exact_mod_cast hroot
  have hmem : ∀ x : ℝ, 0 ≤ x → x ∈ Ici (-β) := fun x hx => by
    simp only [mem_Ici]; linarith
  refine ⟨xs, hxs, hroot, ?_, ?_⟩
  · intro x hx hx1
    have hx1' : g * laplaceR k x = 1 := by
      rw [laplace_ofReal] at hx1
      exact_mod_cast hx1
    by_contra hne
    rcases lt_or_gt_of_ne hne with hlt | hlt
    · have := h.strictAntiOn (hmem x hx) (hmem xs hxs.le) hlt
      nlinarith
    · have := h.strictAntiOn (hmem xs hxs.le) (hmem x hx) hlt
      nlinarith
  · intro z hz
    have h2 := norm_eq_of_root hg hz
    by_cases hre : z.re < 0
    · refine ⟨by linarith, fun heq => ?_⟩
      linarith
    push Not at hre
    have h1 := norm_laplace_le h.nonneg z
    have hle : z.re ≤ xs := by
      by_contra hgt
      push Not at hgt
      have h3 := h.strictAntiOn (hmem xs hxs.le) (hmem z.re hre) hgt
      have : g * ‖laplace k z‖ < g * laplaceR k xs :=
        mul_lt_mul_of_pos_left (h1.trans_lt h3) hg
      linarith
    refine ⟨hle, fun heq => ?_⟩
    have him : z.im = 0 := by
      by_contra hy
      have h4 := norm_laplace_lt (a := -β) (by linarith) h.weighted h.nonneg h.not_ae_zero hy
      rw [heq] at h4
      have : g * ‖laplace k z‖ < g * laplaceR k xs := mul_lt_mul_of_pos_left h4 hg
      linarith
    exact Complex.ext (by simpa using heq) (by simpa using him)

/-! ### Why nonnegativity matters

The paper's example `k t = (3/2) exp (-2 t) - (1/2) exp (-t)` has `S = 1/4`, yet its
characteristic equation `z² + (3 - g) z + 2 - g/2 = 0` has roots with positive real part for
`3 < g < 4`, where `gS < 1`. The algebra behind that example: -/

/-- The root `(1 + i√3)/4` of the example's characteristic equation at `g = 7/2`. -/
def exampleRoot : ℂ := (1 + Complex.I * Real.sqrt 3) / 4

/-- For the example kernel, `k̂ z = 3 / (2 (z + 2)) - 1 / (2 (z + 1))`, and `1 = g k̂ z` is
equivalent (for `z ≠ -1, -2`) to `z² + (3 - g) z + 2 - g/2 = 0`. At `g = 7/2`, where `gS = 7/8 < 1`,
the root `z = (1 + i√3)/4` has positive real part. -/
theorem counterexample_root :
    0 < exampleRoot.re ∧
      exampleRoot ^ 2 + (3 - 7 / 2) * exampleRoot + (2 - 7 / 2 / 2) = 0 ∧
      exampleRoot + 1 ≠ 0 ∧ exampleRoot + 2 ≠ 0 ∧
      (7 / 2 : ℂ) * (3 / (2 * (exampleRoot + 2)) - 1 / (2 * (exampleRoot + 1))) = 1 := by
  have hs : ((Real.sqrt 3 : ℝ) : ℂ) ^ 2 = 3 := by
    rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num)]
    norm_num
  have hre : exampleRoot.re = 1 / 4 := by
    simp [exampleRoot]
  have hpoly : exampleRoot ^ 2 + (3 - 7 / 2) * exampleRoot + (2 - 7 / 2 / 2) = 0 := by
    unfold exampleRoot
    linear_combination ((Real.sqrt 3 : ℂ) ^ 2 / 16) * Complex.I_sq - (1 / 16 : ℂ) * hs
  have h1 : exampleRoot + 1 ≠ 0 := by
    intro h0
    have := congrArg Complex.re h0
    rw [Complex.add_re, hre] at this
    norm_num at this
  have h2 : exampleRoot + 2 ≠ 0 := by
    intro h0
    have := congrArg Complex.re h0
    rw [Complex.add_re, hre] at this
    norm_num at this
  refine ⟨by rw [hre]; norm_num, hpoly, h1, h2, ?_⟩
  field_simp
  linear_combination (-4 : ℂ) * hpoly

end AggregatingInertia
