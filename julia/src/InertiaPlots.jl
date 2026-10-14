"""
    InertiaPlots

Figures for *Aggregating Inertia: Tipping, Slow Shifts and What Revision Records Reveal*, drawn with
Makie from the data files written by the Rust engine (`inertia all`). Two themes (light and dark)
share one palette, built on the three identity hues of the paper's figures.
"""
module InertiaPlots

using CairoMakie
using DelimitedFiles
using JSON

export Palette, LIGHT, DARK, inertia_theme, readcsv, readjson, savefig, arrowhead!, panel_label!, eta_colors, seqfade, thousands

const FONTDIR = joinpath(@__DIR__, "..", "fonts")

"""Colours of one theme."""
struct Palette
    name::String
    bg::RGBAf
    ink::RGBAf
    muted::RGBAf
    rule::RGBAf
    grid::RGBAf
    assess::RGBAf
    process::RGBAf
    event::RGBAf
    tolfill::RGBAf
    transfill::RGBAf
    pausefill::RGBAf
    ramp::Vector{RGBAf}       # four steps of the assessment hue, light to dark (eta = 0, 1, 5, 20)
    seq::Any                  # sequential colormap for densities (low end close to the background)
    div::Vector{RGBAf}        # diverging map: event - background - assess
end

c(s) = RGBAf(Makie.to_color(s))

const LIGHT = Palette("light", c("#FFFFFF"), c("#1D2733"), c("#56606B"), c("#8A929A"), c("#EEF1F4"),
    c("#1F6FA8"), c("#1E9E73"), c("#C5531F"), c("#E3EDF5"), c("#F2F4F6"), c("#E4F3EC"),
    [c("#9CC3E4"), c("#4F95CB"), c("#1F6FA8"), c("#123F63")], Reverse(:lipari),
    [c("#C5531F"), c("#E9A27E"), c("#FFFFFF"), c("#8DB7DB"), c("#1F6FA8")])

const DARK = Palette("dark", c("#0E1319"), c("#E6EAEE"), c("#9AA4AE"), c("#5B6672"), c("#1A222B"),
    c("#5AA7E0"), c("#3CCB9A"), c("#F08A4B"), c("#16273A"), c("#141B23"), c("#12291F"),
    [c("#2B5C86"), c("#3F86C0"), c("#6BB2E8"), c("#B5DCF8")], :lipari,
    [c("#F08A4B"), c("#7A4A30"), c("#0E1319"), c("#2F5878"), c("#5AA7E0")])

eta_colors(p::Palette) = p.ramp

"""
The density colormap of a theme, dissolving into the background over its lowest `frac` (a smoothstep blend),
so that densities near the floor of a logarithmic colour scale fade into the page instead of ending at an edge.
"""
function seqfade(p::Palette; frac = 0.2, n = 512)
    base = Makie.to_colormap(p.seq)
    m = length(base)
    pick(x) = begin
        y = 1 + x * (m - 1); i = clamp(floor(Int, y), 1, m - 1); w = y - i
        a, b = base[i], base[i+1]
        RGBAf((1 - w) * a.r + w * b.r, (1 - w) * a.g + w * b.g, (1 - w) * a.b + w * b.b, 1)
    end
    map(range(0, 1; length = n)) do x
        c = pick(x)
        s = clamp(x / frac, 0, 1); w = s * s * (3 - 2s)
        RGBAf((1 - w) * p.bg.r + w * c.r, (1 - w) * p.bg.g + w * c.g, (1 - w) * p.bg.b + w * c.b, 1)
    end
end

"""An integer with thousands separators: 50000 -> "50,000"."""
thousands(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

function inertia_theme(p::Palette; fontsize = 10)
    fonts = (; regular = joinpath(FONTDIR, "Inter-Regular.otf"), bold = joinpath(FONTDIR, "Inter-SemiBold.otf"),
              italic = joinpath(FONTDIR, "Inter-Italic.otf"), bold_italic = joinpath(FONTDIR, "Inter-Bold.otf"),
              medium = joinpath(FONTDIR, "Inter-Medium.otf"), light = joinpath(FONTDIR, "Inter-Light.otf"))
    Theme(
        fonts = fonts,
        fontsize = fontsize,
        backgroundcolor = p.bg,
        textcolor = p.ink,
        linewidth = 1.6,
        palette = (color = [p.assess, p.event, p.process, p.muted],),
        Axis = (
            backgroundcolor = p.bg,
            xgridvisible = false, ygridvisible = false,
            topspinevisible = false, rightspinevisible = false,
            leftspinecolor = p.rule, bottomspinecolor = p.rule, spinewidth = 0.8,
            xtickcolor = p.rule, ytickcolor = p.rule, xticksize = 3, yticksize = 3, xtickwidth = 0.8, ytickwidth = 0.8,
            xticklabelcolor = p.muted, yticklabelcolor = p.muted,
            xlabelcolor = p.ink, ylabelcolor = p.ink, titlecolor = p.ink,
            xlabelfont = :medium, ylabelfont = :medium, titlefont = :bold,
            titlealign = :left, titlesize = fontsize + 1.5, titlegap = 6,
            xlabelpadding = 4, ylabelpadding = 6,
        ),
        Legend = (framevisible = false, backgroundcolor = (p.bg, 0.0), labelcolor = p.ink, labelsize = fontsize - 0.5,
                  patchsize = (16, 8), rowgap = 1, padding = (2, 2, 2, 2)),
        Colorbar = (labelcolor = p.ink, ticklabelcolor = p.muted, tickcolor = p.rule, spinewidth = 0.0, ticksize = 3,
                    labelfont = :medium),
        Label = (color = p.ink,),
        Lines = (linecap = :round, joinstyle = :round),
    )
end

"""Read a numeric CSV with a header line; returns a Dict column name => vector."""
function readcsv(path)
    data, header = readdlm(path, ',', Float64; header = true)
    Dict(String(strip(h)) => data[:, k] for (k, h) in enumerate(vec(header)))
end

readjson(path) = JSON.parsefile(path)

"""Save a figure as PDF (vector) and PNG (high resolution)."""
function savefig(dir, name, fig; png_scale = 4)
    mkpath(dir)
    save(joinpath(dir, name * ".pdf"), fig; pt_per_unit = 1)
    save(joinpath(dir, name * ".png"), fig; px_per_unit = png_scale)
    return nothing
end

"""A filled arrowhead at `tip` pointing along `dir` (data coordinates rescaled by the axis aspect)."""
function arrowhead!(ax, tip, dir; color, size = 0.06, width = 0.55, aspect = 1.0)
    d = Point2f(dir[1], dir[2] * aspect)
    d = d / sqrt(d[1]^2 + d[2]^2)
    n = Point2f(-d[2], d[1])
    base = Point2f(tip...) .- Point2f(size * d[1], size * d[2] / aspect)
    p1 = base .+ Point2f(width * size * n[1] / 2, width * size * n[2] / 2 / aspect)
    p2 = base .- Point2f(width * size * n[1] / 2, width * size * n[2] / 2 / aspect)
    poly!(ax, [Point2f(tip...), p1, p2]; color = color, strokewidth = 0)
end

"""A bold panel label (A, B, ...) at the top left of a grid position."""
function panel_label!(fig, pos, s; p::Palette)
    Label(pos, s; font = :bold, fontsize = 12, color = p.ink, halign = :left, valign = :top,
          tellwidth = false, tellheight = false, padding = (-2, 0, 0, -4))
end

end # module
