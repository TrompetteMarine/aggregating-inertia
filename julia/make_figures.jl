# Figures for "Aggregating Inertia" from the outputs of the Rust engine.
#
#   julia --project=julia julia/make_figures.jl [output_dir] [figures_dir] [prefix ...]
#
# Reads output_dir/data (written by `inertia all`) and writes every figure in a light and a dark
# theme to figures_dir/{light,dark}/ as PDF and PNG. Prefixes after the two folders restrict the drawing to
# the figures whose names start with one of them (`... output figures 04 00`).

include(joinpath(@__DIR__, "src", "InertiaPlots.jl"))
using .InertiaPlots
using CairoMakie

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output")
const FIGDIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "figures")
D(parts...) = joinpath(OUT, "data", parts...)

const REC = readjson(D("records.json"))
"Agents in the population simulations (the same for both hazards)."
const AGENTS = thousands(Int(only(unique(v["agents"] for v in values(readjson(D("extras", "population_meta.json")))))))
const THETA = 0.7
const ETAS = ["0", "1", "5", "20"]
const ETALAB = ["η = 0 (Calvo)", "η = 1", "η = 5", "η = 20"]

# --------------------------------------------------------------------------- helpers
"Consecutive runs of indices where `mask` is constant: [(range, value), ...]."
function runs(mask)
    out = Tuple{UnitRange{Int},Bool}[]
    i = 1
    while i <= length(mask)
        j = i
        while j < length(mask) && mask[j+1] == mask[i]
            j += 1
        end
        push!(out, (i:j, mask[i]))
        i = j + 1
    end
    out
end

"Centred moving average (window w, shrinking at the ends)."
function smooth(y, w)
    h = w ÷ 2
    [mean_(y[max(1, i - h):min(end, i + h)]) for i in eachindex(y)]
end
mean_(v) = sum(v) / length(v)

"Where the simulated ramp jumps up (first time m exceeds `up`) and drops (first time on the way down m falls below `dn`)."
function ramp_events(h, up, dn)
    half = h["t"] .<= 300.0
    iu = findfirst(>(up), h["m"][half])
    id = findfirst(<(dn), h["m"][.!half])
    gu = iu === nothing ? NaN : h["gS"][half][iu]
    gd = id === nothing ? NaN : h["gS"][.!half][id]
    gu, gd
end

"Branches of stationary states with m > 0 (those with m < 0 are their mirror images): solid where stable,
dashed where unstable."
function branches!(ax, b, p; color, lw = 2.0)
    gs, m, sl = b["gS"], b["m"], b["slope"]
    stable = sl .< 1
    for (r, st) in runs(stable)
        rr = first(r) > 1 ? (first(r)-1:last(r)) : r
        lines!(ax, gs[rr], m[rr]; color = color, linewidth = st ? lw : 1.2, linestyle = st ? :solid : (:dash, :dense))
    end
end

function consensus!(ax, gmin, gmax, p; color)
    lines!(ax, [gmin, 1.0], [0.0, 0.0]; color = color, linewidth = 2.0)
    lines!(ax, [1.0, gmax], [0.0, 0.0]; color = color, linewidth = 1.2, linestyle = (:dash, :dense))
end

# --------------------------------------------------------------------------- 1. tipping landscape
function fig_tipping(p::Palette)
    fig = Figure(size = (560, 250), figure_padding = (6, 10, 4, 4))
    bif = REC["bifurcation"]
    for (k, (name, title, sub)) in enumerate((("up", "Slow shift", "hazard rising with |u|"),
                                              ("down", "Tipping and lock-in", "hazard falling with |u|")))
        ax = Axis(fig[1, k]; xlabel = "social gain × cumulative response,  gS", ylabel = k == 1 ? "stationary average,  m" : "",
                  title = title, subtitle = sub, subtitlecolor = p.muted, subtitlefont = :regular, subtitlesize = 9)
        b = readcsv(D("tip_branch_$name.csv"))
        h = readcsv(D("extras", "hysteresis_$name.csv"))
        half = h["t"] .<= 300.0
        # simulated population under a slow ramp of the gain (agent-based)
        lines!(ax, h["gS"][half], h["m"][half]; color = (p.event, 0.55), linewidth = 1.0)
        lines!(ax, h["gS"][.!half], h["m"][.!half]; color = (p.process, 0.6), linewidth = 1.0)
        branches!(ax, b, p; color = p.assess)
        consensus!(ax, 0.6, 1.3, p; color = p.assess)
        vlines!(ax, [1.0]; color = (p.rule, 0.6), linewidth = 0.7, linestyle = :dot)
        xlims!(ax, 0.6, 1.3)
        ylims!(ax, -0.15, k == 1 ? 1.35 : 3.6)
        if name == "down"
            fold = bif["down"]["fold_gS"]; fm = bif["down"]["fold_m"]; jm = bif["down"]["jump_m"]
            gu, gd = ramp_events(h, 0.5 * jm, 0.5 * fm)
            scatter!(ax, [fold], [fm]; color = p.bg, strokecolor = p.ink, strokewidth = 1.2, markersize = 6)
            text!(ax, fold - 0.012, fm + 0.12; text = "fold", align = (:right, :bottom), color = p.muted, fontsize = 9)
            isfinite(gu) && scatter!(ax, [gu], [0.5 * jm]; marker = :utriangle, color = p.event, markersize = 8)
            isfinite(gd) && scatter!(ax, [gd], [0.5 * fm]; marker = :dtriangle, color = p.process, markersize = 8)
            scatter!(ax, [0.75], [0.0]; marker = :rtriangle, color = p.event, markersize = 7)
            scatter!(ax, [1.15], [jm + 0.28]; marker = :ltriangle, color = p.process, markersize = 7)
            isfinite(gu) && text!(ax, gu + 0.012, 0.5 * jm; text = "tip", align = (:left, :center), color = p.event, fontsize = 9, font = :medium)
            text!(ax, 1.29, 0.30; text = "consensus\nunstable", align = (:right, :bottom), color = p.muted, fontsize = 8.5)
        else
            text!(ax, 1.29, 1.18; text = "tilted states\ngrow gradually", align = (:right, :top), color = p.muted, fontsize = 8.5)
            scatter!(ax, [0.75], [0.0]; marker = :rtriangle, color = p.event, markersize = 7)
        end
    end
    Legend(fig[2, 1:2],
           [LineElement(color = p.assess, linewidth = 2), LineElement(color = p.assess, linestyle = (:dash, :dense), linewidth = 1.2),
            LineElement(color = (p.event, 0.8)), LineElement(color = (p.process, 0.8))],
           ["stable stationary states", "unstable", "$AGENTS agents, gain ramped up", "… and back down"];
           orientation = :horizontal, tellheight = true, labelsize = 8.5, colgap = 14)
    rowgap!(fig.layout, 2)
    fig
end

# --------------------------------------------------------------------------- 2. critical slowing down
function fig_slowing(p::Palette)
    fig = Figure(size = (560, 230), figure_padding = (6, 10, 4, 4))
    tr = readcsv(D("tip_rate.csv"))
    st = REC["stability"]
    ax = Axis(fig[1, 1]; xlabel = "gS", ylabel = rich("dominant root  x", subscript("g")), title = "Critical slowing down",
              subtitle = "the rate at which consensus is restored (x < 0) or lost (x > 0)", subtitlecolor = p.muted, subtitlesize = 9)
    vspan!(ax, 1.0, 1.4; color = (p.event, 0.06))
    hlines!(ax, [0.0]; color = p.rule, linewidth = 0.7)
    cols = eta_colors(p)
    for (k, e) in enumerate(ETAS)
        lines!(ax, tr["gS"], tr["l$e"]; color = (cols[k], 0.9), linewidth = 0.9, linestyle = (:dash, :dense))
        lines!(ax, tr["gS"], tr["x$e"]; color = cols[k], linewidth = 2.0, label = ETALAB[k])
    end
    text!(ax, 1.39, -0.02; text = "dashed: (gS − 1) / mean lag", align = (:right, :top), color = p.muted, fontsize = 8.5)
    xlims!(ax, 0.6, 1.4)
    axislegend(ax; position = :lt, labelsize = 8.5)
    k = readcsv(D("stab_kernels.csv"))
    ax2 = Axis(fig[1, 2]; xlabel = "time since the push,  t", ylabel = "k(t) / S", title = "Response kernels",
               subtitle = "triangles: mean lags", subtitlecolor = p.muted, subtitlesize = 9)
    for (j, e) in enumerate(ETAS)
        S = st["eta$e"]["S"]; L = st["eta$e"]["Lbar"]
        lines!(ax2, k["t"], k["k$e"] ./ S; color = cols[j], linewidth = 2.0)
        scatter!(ax2, [L], [0.0]; marker = :utriangle, color = cols[j], markersize = 7)
    end
    xlims!(ax2, 0, 3.0)
    ylims!(ax2, -0.05, nothing)
    colsize!(fig.layout, 1, Relative(0.58))
    fig
end

# --------------------------------------------------------------------------- 3. Nyquist portrait
function fig_nyquist(p::Palette)
    fig = Figure(size = (480, 250), figure_padding = (6, 10, 4, 4))
    ax = Axis(fig[1, 1]; xlabel = L"\mathrm{Re}\;\hat{k}(i\omega)/S", ylabel = L"\mathrm{Im}\;\hat{k}(i\omega)/S", aspect = DataAspect(),
              title = "Where consensus breaks", subtitle = "the transform of the kernel meets the positive real axis only at ω = 0",
              subtitlecolor = p.muted, subtitlesize = 9)
    hlines!(ax, [0.0]; color = p.rule, linewidth = 0.6)
    vlines!(ax, [0.0]; color = p.rule, linewidth = 0.6)
    lines!(ax, [0.0, 1.1], [0.0, 0.0]; color = (p.event, 0.30), linewidth = 4)
    cols = eta_colors(p)
    for (k, e) in enumerate(ETAS)
        n = readcsv(D("stab_nyquist_eta$e.csv"))
        x, y = n["re"], n["im"]
        lines!(ax, x, y; color = cols[k], linewidth = 1.8, label = ETALAB[k])
        for f in (0.12, 0.35)
            i = clamp(round(Int, f * length(x)), 2, length(x) - 1)
            θ = atan(y[i+1] - y[i-1], x[i+1] - x[i-1])
            scatter!(ax, [x[i]], [y[i]]; marker = :rtriangle, rotation = θ, color = cols[k], markersize = 7)
        end
    end
    scatter!(ax, [1.0], [0.0]; color = p.event, markersize = 7, strokecolor = p.bg, strokewidth = 1)
    text!(ax, 1.03, -0.03; text = "gS = 1:\nthe first root\ncrosses here", align = (:left, :top), color = p.event, fontsize = 8.5)
    text!(ax, 0.5, -0.545; text = "← ω increasing", align = (:center, :top), color = p.muted, fontsize = 8.5)
    xlims!(ax, -0.05, 1.32)
    ylims!(ax, -0.63, 0.06)
    Legend(fig[1, 2], ax; labelsize = 8.5, tellheight = false, valign = :center)
    fig
end

# --------------------------------------------------------------------------- 4. stationary law on (u, a)
function grid_from_long(u, a, z)
    us = sort(unique(u)); as = sort(unique(a))
    iu = Dict(v => i for (i, v) in enumerate(us)); ia = Dict(v => i for (i, v) in enumerate(as))
    Z = fill(NaN, length(us), length(as))
    for k in eachindex(u)
        Z[iu[u[k]], ia[a[k]]] = z[k]
    end
    us, as, Z
end

"A sorted, evenly spaced vector as a range (heatmaps on ranges are drawn as one image, without seams)."
function as_range(v; rtol = 1e-6)
    r = range(first(v), last(v); length = length(v))
    maximum(abs.(v .- r)) <= rtol * step(r) || error("grid not evenly spaced")
    r
end

"Linear interpolation of the rows of Z (one row per time in ts) onto the times tg."
function regrid_rows(ts, Z, tg)
    out = similar(Z, length(tg), size(Z, 2))
    for (k, t) in enumerate(tg)
        i = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
        w = clamp((t - ts[i]) / (ts[i+1] - ts[i]), 0, 1)
        out[k, :] .= (1 - w) .* Z[i, :] .+ w .* Z[i+1, :]
    end
    out
end

"Joint density without the first processing cell, which holds the agents with A = 0 exactly,
interpolated (linearly in the logarithm of the density) from the solver's stretched processing grid
onto a regular grid for display."
function joint_without_atom(tag; amax = 2.2, na = 360)
    j = readcsv(D("extras", "stationary_joint_$tag.csv"))
    amin = minimum(j["a"])
    keep = j["a"] .> amin
    us, as, Z = grid_from_long(j["u"][keep], j["a"][keep], j["density"][keep])
    ag = range(0, amax; length = na)
    L = log.(max.(Z, 1e-300))
    Zg = similar(Z, length(us), na)
    for (k, a) in enumerate(ag)
        i = searchsortedlast(as, a)
        if i < 1
            Zg[:, k] .= Z[:, 1]
        elseif i >= length(as)
            Zg[:, k] .= Z[:, end]
        else
            w = (a - as[i]) / (as[i+1] - as[i])
            Zg[:, k] .= exp.((1 - w) .* L[:, i] .+ w .* L[:, i+1])
        end
    end
    as_range(us), ag, Zg
end

function fig_stationary(p::Palette)
    fig = Figure(size = (560, 320), figure_padding = (6, 10, 4, 4))
    hm = nothing
    data = Dict(tag => joint_without_atom(tag) for tag in ("full", "capped"))
    zmax = maximum(maximum(filter(isfinite, data[t][3])) for t in keys(data))
    for (k, (tag, title)) in enumerate((("full", "Full reset"), ("capped", "Capped reset")))
        us, as, Z = data[tag]
        m = readcsv(D("extras", "stationary_marginal_$tag.csv"))
        at = readcsv(D("extras", "stationary_atom_$tag.csv"))
        share = readjson(D("extras", "stationary_meta_$tag.json"))["atom_share"]
        axm = Axis(fig[1, k]; height = 58, title = title, subtitle = "A = 0 (dashed): $(round(Int, 100share))% of agents",
                   subtitlecolor = p.muted, subtitlesize = 8.5, xticklabelsvisible = false, yticklabelsvisible = false,
                   yticksvisible = false, leftspinevisible = false)
        sel = abs.(m["u"]) .<= 3
        band!(axm, m["u"][sel], zeros(count(sel)), m["density"][sel]; color = (p.assess, 0.22))
        lines!(axm, m["u"][sel], m["density"][sel]; color = p.assess, linewidth = 1.6)
        lines!(axm, at["u"][sel], at["density"][sel]; color = p.ink, linewidth = 1.0, linestyle = (:dash, :dense))
        lines!(axm, m["u"][sel], m["completion_rate"][sel]; color = p.event, linewidth = 1.2)
        xlims!(axm, -3, 3)
        ylims!(axm, 0, nothing)
        ax = Axis(fig[2, k]; xlabel = "assessment  u", ylabel = k == 1 ? "processing  a" : "", xticks = -2:2)
        hm = heatmap!(ax, us, as, Z; colormap = seqfade(p), colorscale = log10, colorrange = (zmax * 1e-6, zmax), interpolate = true,
                      lowclip = (p.bg, 1.0), nan_color = (p.bg, 1.0))
        for s in (-THETA, THETA)
            vlines!(ax, [s]; color = (p.bg, 0.9), linewidth = 0.9, linestyle = (:dash, :dense))
        end
        if tag == "capped"
            for s in (-THETA / 2, THETA / 2)
                vlines!(ax, [s]; color = (p.event, 0.9), linewidth = 0.9, linestyle = :dot)
            end
            text!(ax, 0.0, 2.15; text = "reset to ±θ/2", align = (:center, :top), color = p.event, fontsize = 8)
        end
        text!(ax, 2.9, 2.15; text = "|u| > θ:\nprocessing\naccumulates", align = (:right, :top), color = p.muted, fontsize = 8)
        xlims!(ax, -3, 3)
        ylims!(ax, 0, 2.2)
        linkxaxes!(ax, axm)
        k == 2 && (ax.yticklabelsvisible = false)
    end
    Colorbar(fig[2, 3], hm; label = "density of agents with A > 0", width = 8, ticklabelsize = 8)
    Legend(fig[1, 3], [PolyElement(color = (p.assess, 0.4)), LineElement(color = p.ink, linestyle = (:dash, :dense)),
                       LineElement(color = p.event)], ["all agents", "A = 0", "completions"];
           labelsize = 8, tellwidth = true, valign = :center, patchsize = (12, 6))
    rowgap!(fig.layout, 4)
    colgap!(fig.layout, 10)
    fig
end

# --------------------------------------------------------------------------- 5. propagation of a push
function fig_response(p::Palette)
    fig = Figure(size = (560, 280), figure_padding = (6, 10, 4, 4))
    r = readcsv(D("extras", "response_field.csv"))
    ts0, us, Z0 = grid_from_long(r["t"], r["u"], r["density_change"])
    # the solver records every step early on and every fourth step later: put it on one even time grid
    ts = range(0, 3; length = 601)
    Z = regrid_rows(ts0, Z0, ts)
    us = as_range(us)
    lim = maximum(abs.(filter(isfinite, Z)))
    ax = Axis(fig[1, 1]; xlabel = "time since a small common push,  t", ylabel = "assessment,  u",
              title = "A push travels through the population",
              subtitle = "change in the density of assessments after a unit push at t = 0; the average moves by k(t)", subtitlecolor = p.muted,
              subtitlesize = 9)
    cm = cgrad(p.div)
    hm = heatmap!(ax, ts, us, Z; colormap = cm, colorrange = (-0.35lim, 0.35lim), lowclip = cm[0.0], highclip = cm[1.0],
                  interpolate = true)
    for s in (-THETA, THETA)
        hlines!(ax, [s]; color = (p.muted, 0.7), linewidth = 0.8, linestyle = (:dash, :dense))
    end
    xlims!(ax, 0, 3)
    ylims!(ax, -3, 3)
    Colorbar(fig[1, 2], hm; label = "density change", width = 8, ticklabelsize = 8)
    text!(ax, 2.95, 2.8; text = "mass moves from u < 0 to u > 0", align = (:right, :top), color = p.muted, fontsize = 8.5)
    k = readcsv(D("stab_kernels.csv"))
    ax2 = Axis(fig[2, 1]; xlabel = "t", ylabel = "k(t)", height = 60)
    band!(ax2, k["t"], zeros(length(k["t"])), k["k5"]; color = (p.assess, 0.2))
    lines!(ax2, k["t"], k["k5"]; color = p.assess, linewidth = 1.8)
    xlims!(ax2, 0, 3)
    linkxaxes!(ax, ax2)
    ax.xlabelvisible = false
    ax.xticklabelsvisible = false
    rowgap!(fig.layout, 4)
    fig
end

# --------------------------------------------------------------------------- 6. one record, two economies
function fig_record(p::Palette)
    fig = Figure(size = (560, 360), figure_padding = (6, 10, 4, 4))
    rc = REC["record"]
    h = readcsv(D("record_hazard.csv")); band = readcsv(D("record_hazard_band.csv"))
    d = readcsv(D("record_duration.csv")); r = readcsv(D("record_revision.csv"))
    ax1 = Axis(fig[1, 1]; xlabel = "assessment,  u", ylabel = "completion intensity", title = "What the clock hides",
               subtitle = "at age 1: true hazards (10–90%) and their average", subtitlecolor = p.muted, subtitlesize = 9)
    band!(ax1, h["u"], smooth(h["q10"], 7), smooth(h["q90"], 7); color = (p.event, 0.22))
    lines!(ax1, h["u"], h["hbar"]; color = p.ink, linewidth = 2.0)
    vspan!(ax1, -THETA, THETA; color = (p.assess, 0.07))
    xlims!(ax1, -3, 3)
    ax2 = Axis(fig[1, 2]; xlabel = "duration,  τ", ylabel = "density", title = "Durations: identical",
               subtitle = "true economy B, projection P", subtitlecolor = p.muted, subtitlesize = 9)
    lines!(ax2, d["t"], d["durB"]; color = p.assess, linewidth = 3.2, label = "B (pausing clock)")
    lines!(ax2, d["t"], d["durP"]; color = p.event, linewidth = 1.3, linestyle = (:dash, :dense), label = "P (projection)")
    axislegend(ax2; position = :rt, labelsize = 8.5)
    ax3 = Axis(fig[2, 1]; xlabel = "revision,  Δ", ylabel = "density", title = "Revisions: identical")
    lines!(ax3, r["delta"], r["revB"]; color = p.assess, linewidth = 3.2)
    lines!(ax3, r["delta"], r["revP"]; color = p.event, linewidth = 1.3, linestyle = (:dash, :dense))
    xlims!(ax3, -3, 3)
    ax4 = Axis(fig[2, 2]; xlabel = "duration,  τ", ylabel = "E[ I² | τ ]", title = "Area at completion: different",
               subtitle = "the unidentified part of S", subtitlecolor = p.muted, subtitlesize = 9)
    lines!(ax4, d["t"], d["areaB"]; color = p.assess, linewidth = 2.2)
    lines!(ax4, d["t"], d["areaP"]; color = p.event, linewidth = 1.6, linestyle = (:dash, :dense))
    text!(ax4, 0.2, 0.92; space = :relative, align = (:left, :top), fontsize = 8.5, color = p.muted,
          text = rich("E", subscript("B"), "[I²] = $(round(rc["EI2_B"]; digits = 4))\n", "E", subscript("P"), "[I²] = $(round(rc["EI2_P"]; digits = 4))"))
    for ax in (ax2, ax4)
        xlims!(ax, 0, 6)
    end
    rowgap!(fig.layout, 8)
    fig
end

# --------------------------------------------------------------------------- 7. magnitudes
function fig_magnitudes(p::Palette)
    fig = Figure(size = (560, 210), figure_padding = (6, 10, 4, 4))
    m = readcsv(D("magnitudes.csv")); ml = readcsv(D("magnitudes_lag.csv"))
    xt = ([0, 1, 2, 3], ["0", "0.25", "1", "2"])
    fam_marker = [:circle, :rect, :diamond, :utriangle]
    fam_col = [p.assess, p.event, p.process, p.muted]
    panels = (("Kurtosis statistic", "error as a measure of S, %", m["kerr"], m),
              ("Location", "unidentified part of S, %", m["gap"], m),
              ("Speed", "unidentified part of the mean lag, %", ml["gapL"], ml))
    for (k, (title, yl, y, src)) in enumerate(panels)
        ax = Axis(fig[1, k]; xlabel = "pull,  λ", ylabel = yl, title = title, xticks = xt)
        hlines!(ax, [0.0]; color = p.rule, linewidth = 0.7)
        fam = haskey(src, "family") ? src["family"] : zeros(length(y))
        for f in 0:3
            sel = fam .== f
            any(sel) || continue
            scatter!(ax, src["x"][sel], y[sel]; marker = fam_marker[f+1], color = (fam_col[f+1], 0.85), markersize = 6,
                     strokecolor = p.bg, strokewidth = 0.5)
        end
        xlims!(ax, -0.5, 3.5)
    end
    Legend(fig[2, 1:3], [MarkerElement(marker = fam_marker[f], color = fam_col[f], markersize = 7) for f in 1:4],
           ["pausing clocks", "processing ∝ |U|", "processing ∝ U²", "threshold hazard"];
           orientation = :horizontal, labelsize = 8.5)
    rowgap!(fig.layout, 2)
    fig
end

# --------------------------------------------------------------------------- 8. anatomy of an episode
function fig_episode(p::Palette)
    fig = Figure(size = (560, 330), figure_padding = (6, 10, 4, 4))
    b = readcsv(D("episode_before.csv")); a = readcsv(D("episode_after.csv"))
    ep = REC["episode"]
    tc = ep["EpTc"]
    axs = [Axis(fig[i, 1]; ylabel = yl) for (i, yl) in enumerate(("assessment  U", "processing  A", "hazard  H"))]
    axs[1].title = "Anatomy of an episode"
    axs[1].subtitle = "diamonds: the same assessment and the same current hazard, with different processing behind them"
    axs[1].subtitlecolor = p.muted
    axs[1].subtitlesize = 9
    for ax in axs
        vlines!(ax, [tc]; color = (p.event, 0.5), linewidth = 0.9, linestyle = (:dash, :dense))
    end
    hspan!(axs[1], -THETA, THETA; color = (p.assess, 0.08))
    hspan!(axs[1], THETA, 2THETA; color = (p.muted, 0.06))
    hlines!(axs[1], [0.0]; color = p.rule, linewidth = 0.6)
    band!(axs[1], b["time"], zeros(length(b["time"])), b["assessment"]; color = (p.assess, 0.10))
    lines!(axs[1], b["time"], b["assessment"]; color = p.assess, linewidth = 2.0)
    lines!(axs[1], a["time"], a["assessment"]; color = p.assess, linewidth = 2.0)
    lines!(axs[1], [tc, tc], [b["assessment"][end], 0.0]; color = p.event, linewidth = 2.0)
    scatter!(axs[1], [tc], [b["assessment"][end]]; color = p.event, markersize = 7)
    text!(axs[1], tc + 0.08, b["assessment"][end] / 2; text = "revision Δ = −U", align = (:left, :center), color = p.event, fontsize = 9)
    text!(axs[1], 0.1, -THETA + 0.06; text = "tolerance band: processing pauses", align = (:left, :bottom), color = p.muted, fontsize = 8)
    lines!(axs[2], b["time"], b["processing"]; color = p.process, linewidth = 2.0)
    lines!(axs[2], a["time"], a["processing"]; color = p.process, linewidth = 2.0)
    band!(axs[2], b["time"], zeros(length(b["time"])), b["processing"]; color = (p.process, 0.12))
    lines!(axs[3], b["time"], b["hazard"]; color = p.event, linewidth = 2.0)
    lines!(axs[3], a["time"], a["hazard"]; color = p.event, linewidth = 2.0)
    for (t, u) in ((ep["EpTrOne"], ep["EpUstar"]), (ep["EpTrTwo"], ep["EpUstar"]))
        scatter!(axs[1], [t], [u]; marker = :diamond, color = p.bg, strokecolor = p.ink, strokewidth = 1.2, markersize = 7)
    end
    axs[3].xlabel = "time since the episode began"
    for ax in axs[1:2]
        ax.xticklabelsvisible = false
    end
    linkxaxes!(axs...)
    xlims!(axs[3], 0, 10.2)
    rowgap!(fig.layout, 6)
    fig
end

# --------------------------------------------------------------------------- 9. lumpy revisions
function fig_paths(p::Palette)
    fig = Figure(size = (560, 300), figure_padding = (6, 10, 4, 4))
    s = readcsv(D("extras", "sample_paths.csv")); e = readcsv(D("extras", "sample_events.csv"))
    ax = Axis(fig[1, 1]; xlabel = "time", ylabel = "agent", yticksvisible = false, yticklabelsvisible = false,
              leftspinevisible = false, title = "Lumpy revisions",
              subtitle = "six agents under the pausing clock; colour shows accumulated processing, dots the completions",
              subtitlecolor = p.muted, subtitlesize = 9)
    np = Int(maximum(s["path"])) + 1
    amax = maximum(s["a"])
    for k in 0:np-1
        sel = s["path"] .== k
        off = 3.2 * (np - 1 - k)
        hspan!(ax, off - THETA, off + THETA; color = (p.assess, 0.06))
        hlines!(ax, [off]; color = (p.rule, 0.4), linewidth = 0.5)
        lines!(ax, s["t"][sel], s["u"][sel] .+ off; color = s["a"][sel], colormap = [p.assess, p.process, p.event],
               colorrange = (0, 0.6amax), highclip = p.event, linewidth = 1.3)
        ev = e["path"] .== k
        scatter!(ax, e["t"][ev], e["u_before"][ev] .+ off; color = p.event, markersize = 4.5, strokecolor = p.bg, strokewidth = 0.5)
    end
    xlims!(ax, 0, maximum(s["t"]))
    Colorbar(fig[1, 2]; colormap = [p.assess, p.process, p.event], limits = (0, 0.6amax), label = "processing  A",
             width = 7, ticklabelsize = 8, height = Relative(0.6))
    fig
end

# --------------------------------------------------------------------------- 10. population dynamics
function fig_population(p::Palette)
    fig = Figure(size = (560, 230), figure_padding = (6, 10, 4, 4))
    gsl = ["0.8", "0.95", "1.05", "1.2"]
    cols = [p.ramp[1], p.ramp[2], p.event, RGBAf(0.55 * p.event.r, 0.4 * p.event.g, 0.35 * p.event.b, 1.0)]
    for (k, (name, title)) in enumerate((("up", "Slow shift"), ("down", "Tipping")))
        d = readcsv(D("extras", "population_$name.csv"))
        ax = Axis(fig[1, k]; xlabel = "time", ylabel = k == 1 ? "average assessment  m(t)" : "", title = title,
                  subtitle = "$AGENTS agents, social feedback g c tanh(m/c)", subtitlecolor = p.muted, subtitlesize = 9)
        hlines!(ax, [0.0]; color = p.rule, linewidth = 0.6)
        for (j, g) in enumerate(gsl)
            lines!(ax, d["t"], d["gS$g"]; color = cols[j], linewidth = 1.6, label = "gS = $g")
        end
        xlims!(ax, 0, maximum(d["t"]))
        k == 2 && axislegend(ax; position = :rc, labelsize = 8.5)
    end
    fig
end

# --------------------------------------------------------------------------- hero
function fig_hero(p::Palette)
    fig = Figure(size = (1000, 640), figure_padding = (18, 22, 14, 10), fontsize = 11)
    Label(fig[0, 1:2], "Aggregating inertia"; fontsize = 22, font = :bold, halign = :left, tellwidth = false)
    Label(fig[0, 1:2], "how lumpy, delayed revision aggregates — and what revision records reveal";
          fontsize = 12, color = p.muted, halign = :right, valign = :bottom, tellwidth = false)
    g1 = fig[1, 1] = GridLayout(); g2 = fig[1, 2] = GridLayout(); g3 = fig[2, 1] = GridLayout(); g4 = fig[2, 2] = GridLayout()
    # tipping with simulated hysteresis
    bif = REC["bifurcation"]
    ax = Axis(g1[1, 1]; xlabel = "gS", ylabel = "stationary average  m", title = "Tipping and lock-in",
              subtitle = "stable states (solid), unstable (dashed), $AGENTS agents under a slow ramp", subtitlecolor = p.muted, subtitlesize = 10)
    b = readcsv(D("tip_branch_down.csv")); h = readcsv(D("extras", "hysteresis_down.csv"))
    half = h["t"] .<= 300.0
    lines!(ax, h["gS"][half], h["m"][half]; color = (p.event, 0.6), linewidth = 1.1)
    lines!(ax, h["gS"][.!half], h["m"][.!half]; color = (p.process, 0.65), linewidth = 1.1)
    branches!(ax, b, p; color = p.assess, lw = 2.4)
    consensus!(ax, 0.6, 1.3, p; color = p.assess)
    gu, gd = ramp_events(h, 0.5 * bif["down"]["jump_m"], 0.5 * bif["down"]["fold_m"])
    isfinite(gu) && scatter!(ax, [gu], [0.5 * bif["down"]["jump_m"]]; marker = :utriangle, color = p.event, markersize = 10)
    isfinite(gd) && scatter!(ax, [gd], [0.5 * bif["down"]["fold_m"]]; marker = :dtriangle, color = p.process, markersize = 10)
    xlims!(ax, 0.6, 1.3); ylims!(ax, -0.2, 3.6)
    # critical slowing down
    tr = readcsv(D("tip_rate.csv"))
    ax2 = Axis(g2[1, 1]; xlabel = "gS", ylabel = rich("dominant root  x", subscript("g")), title = "Critical slowing down",
               subtitle = "near the threshold the population moves at the rate (gS − 1) / mean lag", subtitlecolor = p.muted, subtitlesize = 10)
    vspan!(ax2, 1.0, 1.4; color = (p.event, 0.06))
    hlines!(ax2, [0.0]; color = p.rule, linewidth = 0.7)
    for (k, e) in enumerate(ETAS)
        lines!(ax2, tr["gS"], tr["l$e"]; color = (p.ramp[k], 0.9), linewidth = 0.9, linestyle = (:dash, :dense))
        lines!(ax2, tr["gS"], tr["x$e"]; color = p.ramp[k], linewidth = 2.4, label = ETALAB[k])
    end
    xlims!(ax2, 0.6, 1.4)
    axislegend(ax2; position = :lt, labelsize = 9)
    # stationary law (agents with A > 0; the atom at A = 0 is reported in the subtitle)
    us, as, Z = joint_without_atom("full")
    share = readjson(D("extras", "stationary_meta_full.json"))["atom_share"]
    zmax = maximum(filter(isfinite, Z))
    ax3 = Axis(g3[1, 1]; xlabel = "assessment  u", ylabel = "processing  a", title = "The hidden state",
               subtitle = "joint law of assessment and processing; $(round(Int, 100share))% of agents sit at A = 0", subtitlecolor = p.muted, subtitlesize = 10)
    hm = heatmap!(ax3, us, as, Z; colormap = seqfade(p), colorscale = log10, colorrange = (zmax * 1e-6, zmax), lowclip = (p.bg, 1.0),
                  interpolate = true)
    for s in (-THETA, THETA)
        vlines!(ax3, [s]; color = (p.bg, 0.9), linewidth = 1.0, linestyle = (:dash, :dense))
    end
    text!(ax3, 0.0, 2.12; text = "tolerance band", align = (:center, :top), color = p.muted, fontsize = 9)
    text!(ax3, 2.9, 2.12; text = "processing accumulates\nonly outside it", align = (:right, :top), color = p.muted, fontsize = 9)
    xlims!(ax3, -3, 3); ylims!(ax3, 0, 2.2)
    Colorbar(g3[1, 2], hm; width = 8, ticklabelsize = 8, label = "density")
    # one record, two economies
    d = readcsv(D("record_duration.csv"))
    ax4 = Axis(g4[1, 1]; xlabel = "duration  τ", ylabel = "density", title = "One record, two economies",
               subtitle = "identical durations and revisions …", subtitlecolor = p.muted, subtitlesize = 10)
    lines!(ax4, d["t"], d["durB"]; color = p.assess, linewidth = 3.4, label = "true economy")
    lines!(ax4, d["t"], d["durP"]; color = p.event, linewidth = 1.4, linestyle = (:dash, :dense), label = "projection")
    xlims!(ax4, 0, 5)
    axislegend(ax4; position = :rt, labelsize = 9)
    ax5 = Axis(g4[1, 2]; xlabel = "duration  τ", ylabel = "E[I² | τ]", title = " ",
               subtitle = "… different areas, so S differs", subtitlecolor = p.muted, subtitlesize = 10)
    lines!(ax5, d["t"], d["areaB"]; color = p.assess, linewidth = 2.4)
    lines!(ax5, d["t"], d["areaP"]; color = p.event, linewidth = 1.6, linestyle = (:dash, :dense))
    xlims!(ax5, 0, 6)
    rowgap!(fig.layout, 12); colgap!(fig.layout, 22)
    fig
end

# --------------------------------------------------------------------------- driver
const FIGS = [("01_tipping_landscape", fig_tipping), ("02_critical_slowing_down", fig_slowing), ("03_nyquist", fig_nyquist),
              ("04_stationary_law", fig_stationary), ("05_push_propagation", fig_response), ("06_one_record_two_economies", fig_record),
              ("07_magnitudes", fig_magnitudes), ("08_episode_anatomy", fig_episode), ("09_lumpy_revisions", fig_paths),
              ("10_population_dynamics", fig_population), ("00_hero", fig_hero)]

const ONLY = ARGS[3:end]

function main()
    for p in (LIGHT, DARK)
        dir = joinpath(FIGDIR, p.name)
        with_theme(inertia_theme(p)) do
            for (name, f) in FIGS
                isempty(ONLY) || any(startswith(name, o) for o in ONLY) || continue
                t = @elapsed begin
                    fig = f(p)
                    savefig(dir, name, fig)
                end
                println("  $(p.name)/$name  ($(round(t; digits = 1)) s)")
            end
        end
    end
end

main()
