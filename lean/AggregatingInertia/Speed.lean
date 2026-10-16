/-
Copyright (c) 2026 Gabriel L. S. Bontemps. Released under the MIT licence.

# Speed of the transition: Theorem 2

* Theorem 2(a): if the Laplace transform of the closed-loop kernel `k_g = k + Φ * k` is
  `k̂ / (1 - g k̂)` (which the paper derives from the resolvent equation; not formalised here),
  then its integral is `S / (1 - gS)` and its mean lag is `L̄ / (1 - gS)`.
* Theorem 2(b): for `k ≥ 0`, a root `x_g > -β` of `g k̂ x = 1` is unique and simple, every other
  root has `Re z < x_g`, and such a root exists when `gS ≥ 1` and when `gS < 1` is close to one.
* Theorem 2(c): the root moves with the gain at rate `S / L̄` at the critical gain `g = 1/S`,
  that is `x_g = (gS - 1) / L̄ + o(gS - 1)`. The paper's `O((gS - 1)²)` remainder, which follows
  from analyticity, is not formalised.
-/
import AggregatingInertia.Threshold

open MeasureTheory Set Filter Topology Function

noncomputable section

namespace AggregatingInertia

/-! ### Theorem 2(a): size and lag of the closed-loop response -/

/-- **Theorem 2(a)**, Laplace-domain form. Let `K` be the Laplace transform of `k` on the real
axis, with `K 0 = S` and `K' 0 = -∫ t k = -S L̄`. The closed-loop transform `K / (1 - g K)` has
value `S / (1 - gS)` at `0`, and its mean lag `-(K_g)' 0 / K_g 0` is `L̄ / (1 - gS)`. -/
theorem thm2a_lag {K : ℝ → ℝ} {S L g : ℝ} (hK0 : K 0 = S) (hK : HasDerivAt K (-(S * L)) 0)
    (hS : S ≠ 0) (hgS : g * S ≠ 1) :
    (fun x => K x / (1 - g * K x)) 0 = S / (1 - g * S) ∧
      HasDerivAt (fun x => K x / (1 - g * K x)) (-(S * L) / (1 - g * S) ^ 2) 0 ∧
      (S * L / (1 - g * S) ^ 2) / (S / (1 - g * S)) = L / (1 - g * S) := by
  have hden : 1 - g * S ≠ 0 := sub_ne_zero.2 (Ne.symm hgS)
  refine ⟨by simp [hK0], ?_, ?_⟩
  · have hD : HasDerivAt (fun x => 1 - g * K x) (-(g * -(S * L))) 0 := by
      simpa using (hK.const_mul g).const_sub 1
    have hq := hK.div hD (by simpa [hK0] using hden)
    convert hq using 1
    rw [hK0]
    field_simp
    ring
  · field_simp

/-! ### Theorem 2(b): the dominant real root -/

variable {k : ℝ → ℝ} {β : ℝ}

lemma laplaceR_eq_of_root {g : ℝ} {x : ℝ} (hx : g * laplace k x = 1) : g * laplaceR k x = 1 := by
  rw [laplace_ofReal] at hx
  exact_mod_cast hx

/-- **Theorem 2(b)**, uniqueness: a real root `x_g > -β` of `g k̂ x = 1` is unique. -/
theorem thm2b_unique (h : KernelHyp k β) {g x y : ℝ} (hx : -β < x) (hy : -β < y)
    (hgx : g * laplaceR k x = 1) (hgy : g * laplaceR k y = 1) : x = y := by
  by_contra hne
  have hg : g ≠ 0 := by rintro rfl; simp at hgx
  have hfx : laplaceR k x = laplaceR k y := by
    have : g * laplaceR k x = g * laplaceR k y := by rw [hgx, hgy]
    exact mul_left_cancel₀ hg this
  exact hne (h.strictAntiOn.injOn (mem_Ici.2 hx.le) (mem_Ici.2 hy.le) hfx)

/-- The derivative of `k̂` on the real axis: `k̂' x = -∫₀^∞ t exp (-x t) k t dt`, for `x > -β`. -/
lemma hasDerivAt_laplaceR (h : KernelHyp k β) {x : ℝ} (hx : -β < x) :
    HasDerivAt (laplaceR k) (-∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t) x := by
  have hd := hasDerivAt_laplace (a := -β) (z₀ := (x : ℂ)) (by simpa using hx) h.weighted
  have hr := hd.real_of_complex
  have hfun : (fun y : ℝ => (laplace k (y : ℂ)).re) = laplaceR k := by
    funext y
    rw [laplace_ofReal]
    simp
  rw [hfun] at hr
  have hc : (∫ t in Ioi (0 : ℝ), (t : ℂ) * Complex.exp (-((x : ℂ) * t)) * (k t : ℂ)) =
      ((∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t : ℝ) : ℂ) := by
    rw [← integral_complex_ofReal]
    refine integral_congr_ae (Eventually.of_forall fun t => ?_)
    push_cast
    ring_nf
  convert hr using 1
  rw [integral_neg, hc, Complex.neg_re, Complex.ofReal_re]

/-- **Theorem 2(b)**, simplicity: at a real root `x_g > -β`, the derivative of `g k̂` is
`-g ∫₀^∞ t exp (-x_g t) k t dt < 0`. -/
theorem thm2b_simple (h : KernelHyp k β) {g x : ℝ} (hg : 0 < g) (hx : -β < x) :
    HasDerivAt (fun y => g * laplaceR k y)
      (g * -∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t) x ∧
      g * -∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t < 0 := by
  refine ⟨(hasDerivAt_laplaceR h hx).const_mul g, ?_⟩
  have hint : IntegrableOn (fun t => t * (Real.exp (-(x * t)) * k t)) (Ioi 0) := by
    -- `exp (-x t) k` has an exponential moment at rate `x + β > 0`
    set c := (x + β) / 2 with hc
    have hcpos : 0 < c := by rw [hc]; linarith
    have hw : IntegrableOn (fun t => Real.exp (-((x - c) * t)) * k t) (Ioi 0) :=
      h.weighted_of_le (by rw [hc]; linarith)
    have hb : IntegrableOn
        (fun t : ℝ => (t * Real.exp (-(c * t))) * (Real.exp (-((x - c) * t)) * k t)) (Ioi 0) := by
      refine Integrable.bdd_mul (c := c⁻¹) hw ?_ ?_
      · exact (continuous_id.mul (Real.continuous_exp.comp
          ((continuous_const.mul continuous_id).neg))).aestronglyMeasurable
      · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
        have ht' : (0 : ℝ) < t := ht
        rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg ht'.le (Real.exp_pos _).le),
          inv_eq_one_div, le_div_iff₀ hcpos]
        have h1 : c * t ≤ Real.exp (c * t) := le_exp_of_nonneg
        have h2 : Real.exp (c * t) * Real.exp (-(c * t)) = 1 := by
          rw [← Real.exp_add]; simp
        calc t * Real.exp (-(c * t)) * c = (c * t) * Real.exp (-(c * t)) := by ring
          _ ≤ Real.exp (c * t) * Real.exp (-(c * t)) := by gcongr
          _ = 1 := h2
    refine hb.congr (Eventually.of_forall fun t => ?_)
    simp only
    rw [show t * Real.exp (-(c * t)) * (Real.exp (-((x - c) * t)) * k t) =
        t * k t * (Real.exp (-(c * t)) * Real.exp (-((x - c) * t))) by ring, ← Real.exp_add,
      show -(c * t) + -((x - c) * t) = -(x * t) by ring]
    ring
  have hpos : 0 < ∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t := by
    have := setIntegral_pos_of_weight (w := fun t => t * Real.exp (-(x * t))) (N := ∅)
      h.nonneg h.not_ae_zero
      (fun t ht => mul_nonneg (le_of_lt ht) (Real.exp_pos _).le)
      (fun t ht _ => mul_pos ht (Real.exp_pos _)) measure_empty
      (hint.congr (Eventually.of_forall fun t => by simp only; ring))
    simpa using this
  nlinarith

/-- **Theorem 2(b)**, simplicity in the complex sense: `k̂` is complex differentiable at a real
`x > -β`, with derivative the real number `-∫₀^∞ t exp (-x t) k t dt`, so a root `x_g > -β` of
`g k̂ = 1` is a simple zero of the analytic function `1 - g k̂`. -/
theorem thm2b_simple_complex (h : KernelHyp k β) {x : ℝ} (hx : -β < x) :
    HasDerivAt (laplace k) ((-∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t : ℝ) : ℂ) x := by
  have hd := hasDerivAt_laplace (a := -β) (z₀ := (x : ℂ)) (by simpa using hx) h.weighted
  have hc : (∫ t in Ioi (0 : ℝ), (t : ℂ) * Complex.exp (-((x : ℂ) * t)) * (k t : ℂ)) =
      ((∫ t in Ioi (0 : ℝ), t * Real.exp (-(x * t)) * k t : ℝ) : ℂ) := by
    rw [← integral_complex_ofReal]
    refine integral_congr_ae (Eventually.of_forall fun t => ?_)
    push_cast
    ring_nf
  convert hd using 1
  rw [integral_neg, hc, Complex.ofReal_neg]

/-- **Theorem 2(b)**, dominance: every root other than `x_g` has `Re z < x_g`. -/
theorem thm2b_dominant (h : KernelHyp k β) {g x : ℝ} (hg : 0 < g) (hx : -β < x)
    (hroot : g * laplaceR k x = 1) :
    ∀ z : ℂ, g * laplace k z = 1 → z ≠ x → z.re < x := by
  intro z hzroot hne
  by_cases hz : -β < z.re
  swap
  · push Not at hz
    linarith
  have h2 := norm_eq_of_root hg hzroot
  by_contra hge
  push Not at hge
  rcases eq_or_lt_of_le hge with heq | hlt
  · -- `Re z = x_g`, so `Im z ≠ 0`, and the strict bound gives a contradiction
    have him : z.im ≠ 0 := by
      intro him
      exact hne (Complex.ext (by simpa using heq.symm) (by simpa using him))
    have h4 := norm_laplace_lt (a := -β) hz.le h.weighted h.nonneg h.not_ae_zero him
    rw [← heq] at h4
    have : g * ‖laplace k z‖ < g * laplaceR k x := mul_lt_mul_of_pos_left h4 hg
    linarith
  · have h1 := norm_laplace_le h.nonneg z
    have h3 := h.strictAntiOn (mem_Ici.2 hx.le) (mem_Ici.2 hz.le) hlt
    have : g * ‖laplace k z‖ < g * laplaceR k x := mul_lt_mul_of_pos_left (h1.trans_lt h3) hg
    linarith

/-- **Theorem 2(b)**, existence when `gS ≥ 1`: a root `x_g ≥ 0`. -/
theorem thm2b_exists_of_ge (h : KernelHyp k β) {g : ℝ} (hg : 0 < g) (hgS : 1 ≤ g * cumResp k) :
    ∃ x : ℝ, 0 ≤ x ∧ g * laplaceR k x = 1 := by
  rcases eq_or_lt_of_le hgS with heq | hlt
  · exact ⟨0, le_rfl, by rw [laplaceR_zero]; exact heq.symm⟩
  · obtain ⟨x, hx, hroot⟩ := prop1a_real_root h.integrableOn hg hlt
    exact ⟨x, hx.le, laplaceR_eq_of_root hroot⟩

/-- **Theorem 2(b)**, existence when `gS < 1` is close to one: if `1/g ∈ (S, k̂ (-β/2)]`, there is
a root `x_g ∈ [-β/2, 0)`; and `k̂ (-β/2) > S`, so this interval is not empty. -/
theorem thm2b_exists_near (h : KernelHyp k β) {g : ℝ} (hg : 0 < g)
    (hlo : cumResp k < 1 / g) (hhi : 1 / g ≤ laplaceR k (-β / 2)) :
    (∃ x : ℝ, -β / 2 ≤ x ∧ x < 0 ∧ g * laplaceR k x = 1) ∧ cumResp k < laplaceR k (-β / 2) := by
  have hβ := h.beta_pos
  have hmem : ∀ x : ℝ, -β / 2 ≤ x → x ∈ Ici (-β) := fun x hx => by
    simp only [mem_Ici]; linarith
  have hcont : ContinuousOn (laplaceR k) (Icc (-β / 2) 0) :=
    (continuousOn_laplaceR h.weighted).mono fun x hx => hmem x hx.1
  obtain ⟨x, hx, hfx⟩ := intermediate_value_Icc' (by linarith) hcont
    ⟨by rw [laplaceR_zero]; exact hlo.le, hhi⟩
  have hgt : cumResp k < laplaceR k (-β / 2) := by
    rw [← laplaceR_zero]
    exact h.strictAntiOn (hmem _ le_rfl) (hmem 0 (by linarith)) (by linarith)
  refine ⟨⟨x, hx.1, ?_, ?_⟩, hgt⟩
  · rcases eq_or_lt_of_le hx.2 with h0 | h0
    · rw [h0, laplaceR_zero] at hfx
      linarith
    · exact h0
  · rw [hfx]
    field_simp

end AggregatingInertia
