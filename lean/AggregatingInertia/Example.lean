/-
Copyright (c) 2026 Gabriel L. S. Bontemps. Released under the MIT licence.

# Two kernels: the hypotheses are not vacuous, and nonnegativity matters

* `kernelHyp_exp`: the kernel `exp (-t)` satisfies the hypotheses of Theorem 1, so the theorems
  above are not vacuous.
* `example_unstable_below_threshold`: the sign-changing kernel `(3/2) exp (-2t) - (1/2) exp (-t)`
  of the example after Proposition 1 has `S = 1/4`, and at `g = 7/2`, where `gS = 7/8 < 1`, its
  characteristic equation has the root `(1 + i√3)/4` with positive real part. So `gS < 1` does
  not exclude roots in the right half-plane without nonnegativity (compare Theorem 1(a)).
-/
import AggregatingInertia.Threshold

open MeasureTheory Set Filter Topology Function

noncomputable section

namespace AggregatingInertia

/-- The kernel `exp (-t)` satisfies the hypotheses of Theorem 1 with `β = 1/2`. -/
lemma kernelHyp_exp : KernelHyp (fun t => Real.exp (-t)) (1 / 2) where
  beta_pos := by norm_num
  nonneg := fun _ _ => (Real.exp_pos _).le
  not_ae_zero := by
    have : support (fun t : ℝ => Real.exp (-t)) ∩ Ioi 0 = Ioi 0 := by
      ext t
      simp [(Real.exp_pos _).ne']
    rw [this, Real.volume_Ioi]
    simp
  exp_moment := by
    have := integrableOn_exp_mul_Ioi (a := -(1 / 2)) (by norm_num) 0
    refine this.congr_fun (fun t _ => ?_) measurableSet_Ioi
    simp only
    rw [← Real.exp_add]
    ring_nf

/-- The sign-changing kernel of the example after Proposition 1. -/
def exampleKernel (t : ℝ) : ℝ := 3 / 2 * Real.exp (-(2 * t)) - 1 / 2 * Real.exp (-(1 * t))

lemma cexp_mul_exp (c t : ℝ) (z : ℂ) :
    Complex.exp (-(z * t)) * (Real.exp (-(c * t)) : ℂ) = Complex.exp (-(z + c) * t) := by
  rw [Complex.ofReal_exp, ← Complex.exp_add]
  congr 1
  push_cast
  ring

lemma integrableOn_cexp_exp {c : ℝ} {z : ℂ} (h : -c < z.re) :
    IntegrableOn (fun t : ℝ => Complex.exp (-(z * t)) * (Real.exp (-(c * t)) : ℂ)) (Ioi 0) := by
  have ha : (-(z + c)).re < 0 := by simp; linarith
  refine (integrableOn_exp_mul_complex_Ioi ha 0).congr_fun (fun t _ => ?_) measurableSet_Ioi
  exact (cexp_mul_exp c t z).symm

lemma integral_cexp_exp {c : ℝ} {z : ℂ} (h : -c < z.re) :
    ∫ t in Ioi (0 : ℝ), Complex.exp (-(z * t)) * (Real.exp (-(c * t)) : ℂ) = 1 / (z + c) := by
  have ha : (-(z + c)).re < 0 := by simp; linarith
  have h1 : (∫ t in Ioi (0 : ℝ), Complex.exp (-(z * t)) * (Real.exp (-(c * t)) : ℂ)) =
      ∫ t in Ioi (0 : ℝ), Complex.exp (-(z + c) * t) :=
    setIntegral_congr_fun measurableSet_Ioi (fun t _ => cexp_mul_exp c t z)
  rw [h1, integral_exp_mul_complex_Ioi ha 0]
  simp only [Complex.ofReal_zero, mul_zero, Complex.exp_zero]
  rw [neg_div_neg_eq]

/-- `k̂ z = 3 / (2 (z + 2)) - 1 / (2 (z + 1))` for `Re z > -1`. -/
lemma laplace_exampleKernel {z : ℂ} (hz : -1 < z.re) :
    laplace exampleKernel z = 3 / (2 * (z + 2)) - 1 / (2 * (z + 1)) := by
  have h2 : -2 < z.re := by linarith
  have i2 := integrableOn_cexp_exp (c := 2) h2
  have i1 := integrableOn_cexp_exp (c := 1) hz
  have hsplit : laplace exampleKernel z =
      ∫ t in Ioi (0 : ℝ), ((3 / 2 : ℂ) * (Complex.exp (-(z * t)) * (Real.exp (-(2 * t)) : ℂ)) -
        (1 / 2 : ℂ) * (Complex.exp (-(z * t)) * (Real.exp (-(1 * t)) : ℂ))) := by
    unfold laplace exampleKernel
    refine setIntegral_congr_fun measurableSet_Ioi (fun t _ => ?_)
    push_cast
    ring
  have hz1 : z + 1 ≠ 0 := by
    intro h0
    have := congrArg Complex.re h0
    simp at this
    linarith
  have hz2 : z + 2 ≠ 0 := by
    intro h0
    have := congrArg Complex.re h0
    simp at this
    linarith
  rw [hsplit, integral_sub (i2.const_mul _) (i1.const_mul _), integral_const_mul,
    integral_const_mul, integral_cexp_exp h2, integral_cexp_exp hz]
  push_cast
  field_simp

lemma cumResp_exampleKernel : cumResp exampleKernel = 1 / 4 := by
  have h := laplace_exampleKernel (z := 0) (by simp)
  rw [laplace_zero] at h
  have h' : (cumResp exampleKernel : ℂ) = ((1 / 4 : ℝ) : ℂ) := by
    rw [h]
    norm_num
  exact_mod_cast h'

/-- The example after Proposition 1: `gS = 7/8 < 1`, yet `1 = g k̂ z` at `z = (1 + i√3)/4`,
which has positive real part. -/
theorem example_unstable_below_threshold :
    (7 / 2 : ℝ) * cumResp exampleKernel < 1 ∧ 0 < exampleRoot.re ∧
      ((7 / 2 : ℝ) : ℂ) * laplace exampleKernel exampleRoot = 1 := by
  obtain ⟨hre, -, -, -, hroot⟩ := counterexample_root
  refine ⟨by rw [cumResp_exampleKernel]; norm_num, hre, ?_⟩
  rw [laplace_exampleKernel (by linarith)]
  push_cast
  exact hroot

end AggregatingInertia
