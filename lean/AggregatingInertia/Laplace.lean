/-
Copyright (c) 2026 Gabriel L. S. Bontemps. Released under the MIT licence.

# The Laplace transform of a response kernel

Section 3 of *Aggregating Inertia: Tipping, Slow Shifts and What Revision Records Reveal* studies
the characteristic equation `1 = g * k̂ z`, where `k̂ z = ∫₀^∞ exp (-z t) k t dt` is the Laplace
transform of the response kernel `k`. This file proves the facts about `k̂` that the proofs of
Proposition 1, Theorem 1 and Theorem 2 use:

* `k̂` is real on the real axis (`laplace_ofReal`);
* for a nonnegative kernel, `‖k̂ (x + iy)‖ ≤ k̂ x`, strictly when `y ≠ 0`
  (`norm_laplace_le`, `norm_laplace_lt`);
* on the real axis `k̂` is continuous, tends to zero at infinity, and, for a nonnegative kernel
  that is not almost everywhere zero, is positive and strictly decreasing
  (`continuousOn_laplaceR`, `tendsto_laplaceR_atTop`, `laplaceR_pos`, `laplaceR_strictAntiOn`);
* `k̂` is complex differentiable, with `k̂' z = -∫₀^∞ t exp (-z t) k t dt`, to the right of an
  abscissa where the kernel has an exponential moment (`hasDerivAt_laplace`).

Integrability is always assumed in the form "`t ↦ exp (-(a t)) k t` is integrable on `(0, ∞)`"
for some real `a`; `a = 0` is plain integrability and `a = -β < 0` an exponential moment.
-/
import AggregatingInertia.Basic

open MeasureTheory Set Filter Topology Function
open scoped Real

noncomputable section

namespace AggregatingInertia

/-- The Laplace transform `k̂ z = ∫₀^∞ exp (-z t) k t dt` of a kernel on `(0, ∞)`. -/
def laplace (k : ℝ → ℝ) (z : ℂ) : ℂ :=
  ∫ t in Ioi (0 : ℝ), Complex.exp (-(z * t)) * (k t : ℂ)

/-- The Laplace transform on the real axis, as a real function. -/
def laplaceR (k : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), Real.exp (-(x * t)) * k t

/-- The cumulative response `S = ∫₀^∞ k`. -/
def cumResp (k : ℝ → ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), k t

/-- The first moment `∫₀^∞ t k t dt`; the mean lag is this divided by `S`. -/
def firstMoment (k : ℝ → ℝ) : ℝ :=
  ∫ t in Ioi (0 : ℝ), t * k t

variable {k : ℝ → ℝ}

lemma laplaceR_zero : laplaceR k 0 = cumResp k := by
  simp [laplaceR, cumResp]

lemma laplace_ofReal (x : ℝ) : laplace k x = (laplaceR k x : ℂ) := by
  unfold laplace laplaceR
  rw [← integral_complex_ofReal]
  refine integral_congr_ae (Eventually.of_forall fun t => ?_)
  push_cast
  ring_nf

lemma laplace_zero : laplace k 0 = (cumResp k : ℂ) := by
  have := laplace_ofReal (k := k) 0
  simpa [laplaceR_zero] using this

/-! ### Measurability and integrability -/

lemma aestronglyMeasurable_of_weighted {a : ℝ}
    (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    AEStronglyMeasurable k (volume.restrict (Ioi (0 : ℝ))) := by
  have h1 := ha.aestronglyMeasurable
  have h2 : AEStronglyMeasurable (fun t : ℝ => Real.exp (a * t))
      (volume.restrict (Ioi (0 : ℝ))) :=
    (Real.continuous_exp.comp (continuous_const.mul continuous_id)).aestronglyMeasurable
  refine (h2.mul h1).congr (Eventually.of_forall fun t => ?_)
  simp only [Pi.mul_apply]
  rw [← mul_assoc, ← Real.exp_add]
  simp

lemma integrableOn_weighted_of_le {a x : ℝ} (hax : a ≤ x)
    (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) : IntegrableOn (fun t => Real.exp (-(x * t)) * k t) (Ioi 0) := by
  have hb : IntegrableOn
      (fun t : ℝ => Real.exp (-((x - a) * t)) * (Real.exp (-(a * t)) * k t)) (Ioi 0) := by
    refine Integrable.bdd_mul (c := 1) ha ?_ ?_
    · exact (Real.continuous_exp.comp
        ((continuous_const.mul continuous_id).neg)).aestronglyMeasurable
    · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
      have ht' : (0 : ℝ) < t := ht
      rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
      have : 0 ≤ (x - a) * t := mul_nonneg (sub_nonneg.2 hax) ht'.le
      linarith
  refine hb.congr (Eventually.of_forall fun t => ?_)
  simp only
  rw [← mul_assoc, ← Real.exp_add]
  ring_nf

lemma integrableOn_abs_weighted {a : ℝ} (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    IntegrableOn (fun t : ℝ => Real.exp (-(a * t)) * |k t|) (Ioi 0) := by
  refine ha.norm.congr (Eventually.of_forall fun t => ?_)
  simp [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]

lemma norm_cexp_mul_ofReal (z : ℂ) (t r : ℝ) :
    ‖Complex.exp (-(z * t)) * (r : ℂ)‖ = Real.exp (-(z.re * t)) * |r| := by
  rw [norm_mul, Complex.norm_exp, Complex.norm_real, Real.norm_eq_abs]
  congr 2
  simp [Complex.mul_re]

lemma integrableOn_cexp {a : ℝ} {z : ℂ} (haz : a ≤ z.re)
    (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    IntegrableOn (fun t : ℝ => Complex.exp (-(z * t)) * (k t : ℂ)) (Ioi 0) := by
  have hx := integrableOn_abs_weighted (integrableOn_weighted_of_le haz ha)
  have hk := aestronglyMeasurable_of_weighted ha
  refine hx.mono' ?_ (Eventually.of_forall fun t => ?_)
  · have hc : Continuous fun t : ℝ => Complex.exp (-(z * t)) :=
      Complex.continuous_exp.comp ((continuous_const.mul Complex.continuous_ofReal).neg)
    exact hc.aestronglyMeasurable.mul (Complex.continuous_ofReal.comp_aestronglyMeasurable hk)
  · exact (norm_cexp_mul_ofReal z t (k t)).le

/-! ### Bounds off the real axis -/

lemma norm_laplace_le (hk : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t) (z : ℂ) :
    ‖laplace k z‖ ≤ laplaceR k z.re := by
  unfold laplace laplaceR
  refine (norm_integral_le_integral_norm _).trans (le_of_eq ?_)
  refine setIntegral_congr_fun measurableSet_Ioi (fun t ht => ?_)
  rw [norm_cexp_mul_ofReal, abs_of_nonneg (hk t ht)]

/-- Rotating `k̂ z` by a phase `φ` and taking the real part gives a real Laplace-type integral. -/
lemma re_rotate_laplace {a : ℝ} {z : ℂ} (haz : a ≤ z.re) (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0))
    (φ : ℝ) :
    (Complex.exp (φ * Complex.I) * laplace k z).re =
      ∫ t in Ioi (0 : ℝ), Real.exp (-(z.re * t)) * k t * Real.cos (φ - z.im * t) := by
  have hint := integrableOn_cexp haz ha
  have hre := integral_re (𝕜 := ℂ) (hint.const_mul (Complex.exp (φ * Complex.I)))
  simp only [RCLike.re_to_complex] at hre
  unfold laplace
  rw [← integral_const_mul, ← hre]
  refine setIntegral_congr_fun measurableSet_Ioi (fun t _ => ?_)
  rw [← mul_assoc, ← Complex.exp_add]
  have h1 : ((φ : ℂ) * Complex.I + -(z * t)).re = -(z.re * t) := by
    simp [Complex.mul_re]
  have h2 : ((φ : ℂ) * Complex.I + -(z * t)).im = φ - z.im * t := by
    simp [Complex.mul_im]
    ring
  rw [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, mul_zero, sub_zero, Complex.exp_re,
    h1, h2]
  ring

/-- The set of `t` where `cos (φ - y t) = 1` is countable when `y ≠ 0`. -/
lemma countable_cos_eq_one {y : ℝ} (hy : y ≠ 0) (φ : ℝ) :
    {t : ℝ | Real.cos (φ - y * t) = 1}.Countable := by
  refine (Set.countable_range (fun n : ℤ => (φ - n * (2 * π)) / y)).mono ?_
  intro t ht
  obtain ⟨n, hn⟩ := (Real.cos_eq_one_iff _).1 ht
  refine ⟨n, ?_⟩
  simp only
  field_simp
  linarith

/-- Positivity of a weighted integral of `k` with a weight positive on `(0, ∞)` off a null set. -/
lemma setIntegral_pos_of_weight {k : ℝ → ℝ} {w : ℝ → ℝ} {N : Set ℝ}
    (hk : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t) (hpos : 0 < volume (support k ∩ Ioi (0 : ℝ)))
    (hw : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ w t) (hwpos : ∀ t ∈ Ioi (0 : ℝ), t ∉ N → 0 < w t)
    (hN : volume N = 0) (hint : IntegrableOn (fun t => w t * k t) (Ioi 0)) :
    0 < ∫ t in Ioi (0 : ℝ), w t * k t := by
  rw [setIntegral_pos_iff_support_of_nonneg_ae _ hint]
  · refine lt_of_lt_of_le ?_ (measure_mono (s := (support k ∩ Ioi 0) \ N) ?_)
    · rwa [measure_sdiff_null hN]
    · rintro t ⟨⟨hkt, ht⟩, htN⟩
      refine ⟨?_, ht⟩
      exact mul_ne_zero (hwpos t ht htN).ne' hkt
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
    exact mul_nonneg (hw t ht) (hk t ht)

lemma laplaceR_pos {a x : ℝ} (hax : a ≤ x) (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0))
    (hk : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t) (hpos : 0 < volume (support k ∩ Ioi (0 : ℝ))) :
    0 < laplaceR k x :=
  setIntegral_pos_of_weight (N := ∅) hk hpos (fun _ _ => (Real.exp_pos _).le)
    (fun _ _ _ => Real.exp_pos _) measure_empty (integrableOn_weighted_of_le hax ha)

/-- The key strict inequality: off the real axis, `‖k̂ (x + iy)‖ < k̂ x` for a nonnegative kernel
that is not almost everywhere zero. -/
lemma norm_laplace_lt {a : ℝ} {z : ℂ} (haz : a ≤ z.re) (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0))
    (hk : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t) (hpos : 0 < volume (support k ∩ Ioi (0 : ℝ)))
    (hz : z.im ≠ 0) : ‖laplace k z‖ < laplaceR k z.re := by
  set w := laplace k z with hw
  have hx := integrableOn_weighted_of_le haz ha
  by_cases hw0 : w = 0
  · rw [hw0, norm_zero]
    exact laplaceR_pos haz ha hk hpos
  -- rotate `w` onto the positive real axis
  set θ := Complex.arg w with hθ
  have hw' : w = (‖w‖ : ℂ) * Complex.exp (θ * Complex.I) := by
    rw [hθ]
    exact (Complex.norm_mul_exp_arg_mul_I w).symm
  have hrot : (Complex.exp ((-θ : ℝ) * Complex.I) * w).re = ‖w‖ := by
    have e1 : Complex.exp ((-θ : ℝ) * Complex.I) * w = (‖w‖ : ℂ) := by
      conv_lhs => rw [hw']
      rw [mul_left_comm, ← Complex.exp_add]
      push_cast
      rw [neg_mul, neg_add_cancel, Complex.exp_zero, mul_one]
    rw [e1, Complex.ofReal_re]
  have hre := re_rotate_laplace haz ha (-θ)
  rw [← hw, hrot] at hre
  -- `k̂ x - ‖w‖` is the integral of `exp (-x t) k t (1 - cos (-θ - y t))`
  have hcos_int : IntegrableOn
      (fun t => Real.exp (-(z.re * t)) * k t * Real.cos (-θ - z.im * t)) (Ioi 0) := by
    refine Integrable.bdd_mul (f := fun t => Real.cos (-θ - z.im * t)) (c := 1) hx ?_ ?_
      |>.congr ?_
    · exact (Real.continuous_cos.comp
        (continuous_const.sub (continuous_const.mul continuous_id))).aestronglyMeasurable
    · exact Eventually.of_forall fun t => by
        simpa [Real.norm_eq_abs] using Real.abs_cos_le_one _
    · exact Eventually.of_forall fun t => by simp only; ring
  have hdiff : laplaceR k z.re - ‖w‖ =
      ∫ t in Ioi (0 : ℝ), (Real.exp (-(z.re * t)) * (1 - Real.cos (-θ - z.im * t))) * k t := by
    rw [hre, laplaceR, ← integral_sub hx hcos_int]
    refine integral_congr_ae (Eventually.of_forall fun t => ?_)
    simp only
    ring
  have hpos' : 0 < ∫ t in Ioi (0 : ℝ),
      (Real.exp (-(z.re * t)) * (1 - Real.cos (-θ - z.im * t))) * k t := by
    refine setIntegral_pos_of_weight (N := {t | Real.cos (-θ - z.im * t) = 1}) hk hpos
      (fun t _ => mul_nonneg (Real.exp_pos _).le (sub_nonneg.2 (Real.cos_le_one _)))
      (fun t _ htN => mul_pos (Real.exp_pos _)
        (sub_pos.2 (lt_of_le_of_ne (Real.cos_le_one _) htN)))
      ((countable_cos_eq_one hz (-θ)).measure_zero _) ?_
    have h1 : IntegrableOn (fun t => Real.exp (-(z.re * t)) * k t -
        Real.exp (-(z.re * t)) * k t * Real.cos (-θ - z.im * t)) (Ioi 0) := hx.sub hcos_int
    refine h1.congr (Eventually.of_forall fun t => ?_)
    simp only
    ring
  linarith

/-! ### The real axis -/

lemma continuousOn_laplaceR {a : ℝ} (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    ContinuousOn (laplaceR k) (Ici a) := by
  have hk := aestronglyMeasurable_of_weighted ha
  refine continuousOn_of_dominated (bound := fun t => Real.exp (-(a * t)) * |k t|)
    (fun x _ => ?_) (fun x hx => ?_) (integrableOn_abs_weighted ha) ?_
  · exact (Real.continuous_exp.comp
      ((continuous_const.mul continuous_id).neg)).aestronglyMeasurable.mul hk
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
    have ht' : (0 : ℝ) < t := ht
    have hxa : a ≤ x := hx
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
    gcongr
  · exact Eventually.of_forall fun t =>
      ((Real.continuous_exp.comp ((continuous_id.mul continuous_const).neg)).mul
        continuous_const).continuousOn

lemma tendsto_laplaceR_atTop {a : ℝ} (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    Tendsto (laplaceR k) atTop (𝓝 0) := by
  have hk := aestronglyMeasurable_of_weighted ha
  have h := tendsto_integral_filter_of_dominated_convergence
    (μ := volume.restrict (Ioi (0 : ℝ))) (l := atTop)
    (F := fun x t => Real.exp (-(x * t)) * k t) (f := fun _ => 0)
    (fun t => Real.exp (-(a * t)) * |k t|)
    (Eventually.of_forall fun x => (Real.continuous_exp.comp
      ((continuous_const.mul continuous_id).neg)).aestronglyMeasurable.mul hk)
    ?_ (integrableOn_abs_weighted ha) ?_
  · show Tendsto (fun x => ∫ t in Ioi (0 : ℝ), Real.exp (-(x * t)) * k t) atTop (𝓝 0)
    simpa using h
  · filter_upwards [eventually_ge_atTop a] with x hx
    refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
    have ht' : (0 : ℝ) < t := ht
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (Real.exp_pos _)]
    gcongr
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht => ?_)
    have ht' : (0 : ℝ) < t := ht
    have h1 : Tendsto (fun x : ℝ => -(x * t)) atTop atBot :=
      tendsto_neg_atTop_atBot.comp (tendsto_id.atTop_mul_const ht')
    simpa using (Real.tendsto_exp_atBot.comp h1).mul_const (k t)

lemma laplaceR_strictAntiOn {a : ℝ} (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0))
    (hk : ∀ t ∈ Ioi (0 : ℝ), 0 ≤ k t) (hpos : 0 < volume (support k ∩ Ioi (0 : ℝ))) :
    StrictAntiOn (laplaceR k) (Ici a) := by
  intro x hx y hy hxy
  have hx' : a ≤ x := hx
  have hint_x := integrableOn_weighted_of_le hx' ha
  have hint_y := integrableOn_weighted_of_le (hx'.trans hxy.le) ha
  have hd : laplaceR k x - laplaceR k y =
      ∫ t in Ioi (0 : ℝ), (Real.exp (-(x * t)) - Real.exp (-(y * t))) * k t := by
    rw [laplaceR, laplaceR, ← integral_sub hint_x hint_y]
    refine integral_congr_ae (Eventually.of_forall fun t => ?_)
    simp only
    ring
  have hp : 0 < ∫ t in Ioi (0 : ℝ), (Real.exp (-(x * t)) - Real.exp (-(y * t))) * k t := by
    refine setIntegral_pos_of_weight (N := ∅) hk hpos (fun t ht => ?_) (fun t ht _ => ?_)
      measure_empty ?_
    · have ht' : (0 : ℝ) < t := ht
      exact sub_nonneg.2 (Real.exp_le_exp.2 (by nlinarith))
    · have ht' : (0 : ℝ) < t := ht
      exact sub_pos.2 (Real.exp_lt_exp.2 (by nlinarith))
    · exact (hint_x.sub hint_y).congr (Eventually.of_forall fun t => by
        simp only [Pi.sub_apply]; ring)
  linarith

/-! ### Differentiability -/

lemma le_exp_of_nonneg {u : ℝ} : u ≤ Real.exp u := by
  linarith [Real.add_one_le_exp u]

/-- To the right of an abscissa `a` where `exp (-a t) k t` is integrable, `k̂` is complex
differentiable with `k̂' z = -∫₀^∞ t exp (-z t) k t dt`. -/
lemma hasDerivAt_laplace {a : ℝ} {z₀ : ℂ} (hz₀ : a < z₀.re) (ha : IntegrableOn (fun t => Real.exp (-(a * t)) * k t) (Ioi 0)) :
    HasDerivAt (laplace k)
      (∫ t in Ioi (0 : ℝ), -((t : ℂ) * Complex.exp (-(z₀ * t)) * (k t : ℂ))) z₀ := by
  set ε := (z₀.re - a) / 2 with hε
  have hεpos : 0 < ε := by rw [hε]; linarith
  have hk := aestronglyMeasurable_of_weighted ha
  have hmeas : ∀ z : ℂ, AEStronglyMeasurable
      (fun t : ℝ => Complex.exp (-(z * t)) * (k t : ℂ)) (volume.restrict (Ioi (0 : ℝ))) :=
    fun z => (Complex.continuous_exp.comp ((continuous_const.mul
      Complex.continuous_ofReal).neg)).aestronglyMeasurable.mul
      (Complex.continuous_ofReal.comp_aestronglyMeasurable hk)
  have hbound_int : IntegrableOn (fun t : ℝ => ε⁻¹ * (Real.exp (-(a * t)) * |k t|)) (Ioi 0) :=
    (integrableOn_abs_weighted ha).const_mul _
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := volume.restrict (Ioi (0 : ℝ))) (x₀ := z₀) (s := Metric.ball z₀ ε)
    (F := fun z t => Complex.exp (-(z * t)) * (k t : ℂ))
    (F' := fun z t => -((t : ℂ) * Complex.exp (-(z * t)) * (k t : ℂ)))
    (bound := fun t => ε⁻¹ * (Real.exp (-(a * t)) * |k t|))
    (Metric.ball_mem_nhds z₀ hεpos) (Eventually.of_forall fun z => hmeas z)
    (integrableOn_cexp (by linarith) ha) ?_ ?_ hbound_int ?_
  · exact key.2
  · -- measurability of the derivative at z₀
    exact ((Complex.continuous_ofReal.mul (Complex.continuous_exp.comp ((continuous_const.mul
      Complex.continuous_ofReal).neg))).aestronglyMeasurable.mul
      (Complex.continuous_ofReal.comp_aestronglyMeasurable hk)).neg
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (Eventually.of_forall fun t ht z hz => ?_)
    have ht' : (0 : ℝ) < t := ht
    have hzre : a + ε ≤ z.re := by
      have h1 : |z.re - z₀.re| ≤ ‖z - z₀‖ := by
        simpa using Complex.abs_re_le_norm (z - z₀)
      have h2 : ‖z - z₀‖ < ε := by simpa [dist_eq_norm] using hz
      have h3 := abs_lt.1 (lt_of_le_of_lt h1 h2)
      rw [hε] at h3 ⊢
      linarith [h3.1]
    rw [norm_neg, norm_mul, norm_mul, Complex.norm_real, Complex.norm_exp, Complex.norm_real,
      Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos ht']
    have hre : (-(z * (t : ℂ))).re = -(z.re * t) := by simp [Complex.mul_re]
    rw [hre]
    -- t * exp (-(re z) t) ≤ ε⁻¹ * exp (-a t)
    have h4 : t * Real.exp (-(z.re * t)) ≤ ε⁻¹ * Real.exp (-(a * t)) := by
      have h5 : ε * t ≤ Real.exp (ε * t) := le_exp_of_nonneg
      have h6 : Real.exp (-(z.re * t)) ≤ Real.exp (-((a + ε) * t)) :=
        Real.exp_le_exp.2 (by nlinarith)
      have h7 : Real.exp (-((a + ε) * t)) * Real.exp (ε * t) = Real.exp (-(a * t)) := by
        rw [← Real.exp_add]; ring_nf
      calc t * Real.exp (-(z.re * t)) ≤ t * Real.exp (-((a + ε) * t)) :=
            mul_le_mul_of_nonneg_left h6 ht'.le
        _ = ε⁻¹ * (ε * t) * Real.exp (-((a + ε) * t)) := by field_simp
        _ ≤ ε⁻¹ * Real.exp (ε * t) * Real.exp (-((a + ε) * t)) := by gcongr
        _ = ε⁻¹ * Real.exp (-(a * t)) := by rw [mul_assoc, mul_comm (Real.exp _), h7]
    calc t * Real.exp (-(z.re * t)) * |k t| ≤ ε⁻¹ * Real.exp (-(a * t)) * |k t| := by
          gcongr
      _ = ε⁻¹ * (Real.exp (-(a * t)) * |k t|) := by ring
  · refine Eventually.of_forall fun t z _ => ?_
    have h1 : HasDerivAt (fun z : ℂ => -(z * t)) (-(t : ℂ)) z :=
      (hasDerivAt_mul_const (t : ℂ)).neg
    have h2 := (h1.cexp).mul_const (k t : ℂ)
    convert h2 using 1
    ring

end AggregatingInertia
