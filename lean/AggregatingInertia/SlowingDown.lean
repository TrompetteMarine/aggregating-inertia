/-
Copyright (c) 2026 Gabriel L. S. Bontemps. Released under the MIT licence.

# Critical slowing down: Theorem 2(c), to first order

Near the critical gain `g = 1/S`, the real root `x_g` of `g k̂ x = 1` is a differentiable function
of the gain with `d x_g / d g = S / L̄ = S² / ∫ t k` at `g = 1/S`. Equivalently
`x_g = (gS - 1) / L̄ + o(gS - 1)`, the first-order part of equation (slowing) in Theorem 2(c).
The paper's `O((gS - 1)²)` bound on the remainder, which follows from analyticity in the
implicit function theorem, is not formalised.
-/
import AggregatingInertia.Speed

open MeasureTheory Set Filter Topology Function

noncomputable section

namespace AggregatingInertia

variable {k : ℝ → ℝ} {β : ℝ}

lemma hasDerivAt_laplaceR_zero (h : KernelHyp k β) :
    HasDerivAt (laplaceR k) (-firstMoment k) 0 := by
  have hd := hasDerivAt_laplaceR h (x := 0) (by linarith [h.beta_pos])
  convert hd using 2
  simp [firstMoment]

/-- **Theorem 2(c)**, first order. There is a function `r` of the gain, defined near `g = 1/S`,
whose value is the unique real root `x_g > -β/2` of `g k̂ x = 1`; it vanishes at `g = 1/S` and has
derivative `S² / ∫₀^∞ t k t dt = S / L̄` there. -/
theorem thm2c_slowing (h : KernelHyp k β) :
    ∃ r : ℝ → ℝ, r (1 / cumResp k) = 0 ∧
      (∀ᶠ g in 𝓝 (1 / cumResp k), -β / 2 < r g ∧ g * laplaceR k (r g) = 1 ∧
        ∀ x : ℝ, -β < x → g * laplaceR k x = 1 → x = r g) ∧
      HasDerivAt r (cumResp k ^ 2 / firstMoment k) (1 / cumResp k) := by
  classical
  set S := cumResp k with hSdef
  set M := firstMoment k with hMdef
  set F := laplaceR k with hFdef
  have hβ := h.beta_pos
  have hS : 0 < S := h.cumResp_pos
  have hM : 0 < M := h.firstMoment_pos
  have hF0 : F 0 = S := laplaceR_zero
  set g₀ := 1 / S with hg₀
  have hg₀pos : 0 < g₀ := by positivity
  have hmem : ∀ x : ℝ, -β / 2 ≤ x → x ∈ Ici (-β) := fun x hx => by
    simp only [mem_Ici]; linarith
  have hanti := h.strictAntiOn
  have hcont : ContinuousOn F (Ici (-β)) := continuousOn_laplaceR h.weighted
  -- the root map
  let r : ℝ → ℝ := fun g =>
    if hex : ∃ x, -β / 2 < x ∧ g * F x = 1 then hex.choose else 0
  have hr_spec : ∀ g, (∃ x, -β / 2 < x ∧ g * F x = 1) → -β / 2 < r g ∧ g * F (r g) = 1 := by
    intro g hex
    have hrg : r g = hex.choose := by simp [r, hex]
    rw [hrg]
    exact hex.choose_spec
  have huniq : ∀ g x y : ℝ, -β < x → -β < y → g * F x = 1 → g * F y = 1 → x = y :=
    fun g x y hx hy hgx hgy => thm2b_unique h hx hy hgx hgy
  -- for `ε ∈ (0, β/2)`, gains with `1/g ∈ (F ε, F (-ε))` have their root in `[-ε, ε]`
  have hloc : ∀ ε, 0 < ε → ε < β / 2 → ∀ g, 0 < g → F ε < 1 / g → 1 / g < F (-ε) →
      (-β / 2 < r g ∧ g * F (r g) = 1) ∧ -ε ≤ r g ∧ r g ≤ ε := by
    intro ε hε hεβ g hg hlo hhi
    have hc : ContinuousOn F (Icc (-ε) ε) := hcont.mono fun x hx => hmem x (by linarith [hx.1])
    obtain ⟨x, hx, hfx⟩ := intermediate_value_Icc' (by linarith) hc ⟨hlo.le, hhi.le⟩
    have hgx : g * F x = 1 := by rw [hfx]; field_simp
    have hex : ∃ x, -β / 2 < x ∧ g * F x = 1 := ⟨x, by linarith [hx.1], hgx⟩
    have hsp := hr_spec g hex
    have hxr : x = r g := huniq g x (r g) (by linarith [hx.1]) (by linarith [hsp.1]) hgx hsp.2
    exact ⟨hsp, hxr ▸ hx.1, hxr ▸ hx.2⟩
  -- `F ε < S < F (-ε)` for `ε > 0`
  have hbr : ∀ ε, 0 < ε → ε < β / 2 → F ε < S ∧ S < F (-ε) := by
    intro ε hε hεβ
    constructor
    · rw [← hF0]; exact hanti (hmem 0 (by linarith)) (hmem ε (by linarith)) hε
    · rw [← hF0]; exact hanti (hmem (-ε) (by linarith)) (hmem 0 (by linarith)) (by linarith)
  have hFpos : ∀ x, -β / 2 ≤ x → 0 < F x := fun x hx =>
    laplaceR_pos (hmem x hx) h.weighted h.nonneg h.not_ae_zero
  -- the gains with `1/g ∈ (F ε, F (-ε))` form a neighbourhood of `g₀`
  have hnhds : ∀ ε, 0 < ε → ε < β / 2 →
      ∀ᶠ g in 𝓝 g₀, 0 < g ∧ F ε < 1 / g ∧ 1 / g < F (-ε) := by
    intro ε hε hεβ
    obtain ⟨h1, h2⟩ := hbr ε hε hεβ
    have hinv : Tendsto (fun g : ℝ => 1 / g) (𝓝 g₀) (𝓝 S) := by
      have h1 : Tendsto (fun g : ℝ => g⁻¹) (𝓝 g₀) (𝓝 g₀⁻¹) := tendsto_inv₀ hg₀pos.ne'
      have h2 : g₀⁻¹ = S := by rw [hg₀]; simp
      rw [h2] at h1
      simpa only [one_div] using h1
    filter_upwards [(isOpen_Ioi.mem_nhds hg₀pos), hinv.eventually (isOpen_Ioi.mem_nhds h1),
      hinv.eventually (isOpen_Iio.mem_nhds h2)] with g hg hlo hhi
    exact ⟨hg, hlo, hhi⟩
  have hβ4 : (0 : ℝ) < β / 4 := by linarith
  have hβ4' : β / 4 < β / 2 := by linarith
  -- value at `g₀`
  have hr0 : r g₀ = 0 := by
    have hex : ∃ x, -β / 2 < x ∧ g₀ * F x = 1 := ⟨0, by linarith, by rw [hF0, hg₀]; field_simp⟩
    have hsp := hr_spec g₀ hex
    exact (huniq g₀ 0 (r g₀) (by linarith) (by linarith [hsp.1])
      (by rw [hF0, hg₀]; field_simp) hsp.2).symm
  refine ⟨r, hr0, ?_, ?_⟩
  · filter_upwards [hnhds (β / 4) hβ4 hβ4'] with g ⟨hg, hlo, hhi⟩
    obtain ⟨hsp, -, -⟩ := hloc (β / 4) hβ4 hβ4' g hg hlo hhi
    exact ⟨hsp.1, hsp.2, fun x hx hgx => huniq g x (r g) hx (by linarith [hsp.1]) hgx hsp.2⟩
  -- continuity of `r` at `g₀`
  have hcontr : Tendsto r (𝓝 g₀) (𝓝 0) := by
    rw [Metric.tendsto_nhds]
    intro e he
    set ε := min (e / 2) (β / 4) with hεdef
    have hε : 0 < ε := lt_min (by linarith) hβ4
    have hεβ : ε < β / 2 := lt_of_le_of_lt (min_le_right _ _) hβ4'
    filter_upwards [hnhds ε hε hεβ] with g ⟨hg, hlo, hhi⟩
    obtain ⟨-, h1, h2⟩ := hloc ε hε hεβ g hg hlo hhi
    rw [Real.dist_eq, sub_zero, abs_lt]
    have : ε ≤ e / 2 := min_le_left _ _
    constructor <;> linarith
  -- off `g₀`, the root is not zero
  have hne : ∀ᶠ g in 𝓝[≠] g₀, r g ≠ 0 ∧ 0 < g ∧ g * F (r g) = 1 := by
    have h1 : ∀ᶠ g in 𝓝[≠] g₀, 0 < g ∧ F (β / 4) < 1 / g ∧ 1 / g < F (-(β / 4)) :=
      nhdsWithin_le_nhds (hnhds (β / 4) hβ4 hβ4')
    filter_upwards [h1, self_mem_nhdsWithin] with g ⟨hg, hlo, hhi⟩ hgne
    obtain ⟨hsp, -, -⟩ := hloc (β / 4) hβ4 hβ4' g hg hlo hhi
    refine ⟨fun h0 => hgne ?_, hg, hsp.2⟩
    have := hsp.2
    rw [h0, hF0] at this
    rw [Set.mem_singleton_iff, hg₀]
    field_simp
    linarith
  have hrw : Tendsto r (𝓝[≠] g₀) (𝓝[≠] 0) := by
    refine tendsto_nhdsWithin_iff.2 ⟨tendsto_nhdsWithin_of_tendsto_nhds (hr0 ▸ hcontr), ?_⟩
    filter_upwards [hne] with g hg
    exact hg.1
  -- slopes
  have hslopeF : Tendsto (slope F 0) (𝓝[≠] 0) (𝓝 (-M)) :=
    (hasDerivAt_iff_tendsto_slope.1 (hasDerivAt_laplaceR_zero h))
  have hslopeInv : Tendsto (slope (fun g : ℝ => g⁻¹) g₀) (𝓝[≠] g₀) (𝓝 (-(g₀ ^ 2)⁻¹)) :=
    hasDerivAt_iff_tendsto_slope.1 (hasDerivAt_inv hg₀pos.ne')
  have hlim : Tendsto (fun g => (slope F 0 (r g))⁻¹ * slope (fun g : ℝ => g⁻¹) g₀ g)
      (𝓝[≠] g₀) (𝓝 ((-M)⁻¹ * (-(g₀ ^ 2)⁻¹))) :=
    ((hslopeF.comp hrw).inv₀ (by linarith)).mul hslopeInv
  have hval : (-M)⁻¹ * (-(g₀ ^ 2)⁻¹) = S ^ 2 / M := by
    rw [hg₀]
    field_simp
  rw [hval] at hlim
  rw [hasDerivAt_iff_tendsto_slope]
  refine hlim.congr' ?_
  filter_upwards [hne, self_mem_nhdsWithin] with g ⟨hr, hg, hroot⟩ hgne
  have hFr : F (r g) = g⁻¹ := (inv_eq_of_mul_eq_one_right hroot).symm
  have hg₀inv : g₀⁻¹ = S := by rw [hg₀]; simp
  have hdiff : g⁻¹ - S ≠ 0 := by
    intro h0
    apply hgne
    rw [Set.mem_singleton_iff]
    have hinv : g⁻¹ = g₀⁻¹ := by rw [hg₀inv]; exact sub_eq_zero.1 h0
    exact inv_inj.1 hinv
  have e1 : slope r g₀ g = r g / (g - g₀) := by rw [slope_def_field, hr0, sub_zero]
  have e2 : slope F 0 (r g) = (g⁻¹ - S) / r g := by rw [slope_def_field, hFr, hF0, sub_zero]
  have e3 : slope (fun y : ℝ => y⁻¹) g₀ g = (g⁻¹ - S) / (g - g₀) := by
    rw [slope_def_field, hg₀inv]
  rw [e1, e2, e3, inv_div, div_mul_div_comm, mul_comm (r g) (g⁻¹ - S),
    mul_div_mul_left _ _ hdiff]

end AggregatingInertia
