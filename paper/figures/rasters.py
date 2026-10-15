"""Raster images for the two heatmaps of the paper's consensus figure, from the engine's output.

    python3 rasters.py DATA_DIR OUT_DIR

DATA_DIR holds extras/stationary_joint_full.csv and extras/response_field.csv (written by `inertia extras`).
OUT_DIR receives consensus_density.png, consensus_response.png and raster_values.tex, whose macros give the
colour ranges and the two colour maps, so that the colour bars drawn by PGFPlots in consensus.tex match the
images. Needs numpy only.
"""
import os
import struct
import sys
import zlib

import numpy as np

# (position, RGB) stops; PGFPlots interpolates linearly in RGB between them, and so does `paint`.
DENSITY = [(0.00, (255, 255, 255)), (0.08, (247, 252, 240)), (0.195, (224, 243, 219)), (0.31, (204, 235, 197)),
           (0.425, (168, 221, 181)), (0.54, (123, 204, 196)), (0.655, (78, 179, 211)), (0.77, (43, 140, 190)),
           (0.885, (8, 104, 172)), (1.00, (8, 64, 129))]           # white, then ColorBrewer GnBu (9 classes)
DIVERGING = [(0.00, (197, 83, 31)), (0.25, (233, 162, 126)), (0.50, (255, 255, 255)), (0.75, (141, 183, 219)),
             (1.00, (31, 111, 168))]
DECADES = 6.0       # the density colour scale spans six decades below the largest density shown
CLIP = 0.35         # the response colour scale saturates at this fraction of the largest change


NSTOPS = 41         # both maps are resampled at this many evenly spaced stops, used by the images and by PGFPlots


def resample(stops, n=NSTOPS):
    """The colour map `stops` at n evenly spaced positions, as integer RGB."""
    pos = np.array([p for p, _ in stops])
    rgb = np.array([c for _, c in stops], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    return np.round(np.stack([np.interp(x, pos, rgb[:, k]) for k in range(3)], axis=-1)).astype(int)


def paint(x, stops):
    """Colours of the values x in [0, 1] (any shape) under the resampled colour map, interpolated linearly in
    RGB between evenly spaced stops exactly as PGFPlots does for its colour bars, as uint8 RGB."""
    table = resample(stops).astype(float)
    grid = np.linspace(0.0, 1.0, len(table))
    x = np.clip(x, 0.0, 1.0)
    out = np.stack([np.interp(x, grid, table[:, k]) for k in range(3)], axis=-1)
    return np.round(out).astype(np.uint8)


def pgf_colormap(name, stops):
    return "\\pgfplotsset{colormap={%s}{%s}}\n" % (name, " ".join("rgb255=(%d,%d,%d)" % tuple(c) for c in resample(stops)))


def write_png(path, img):
    """An RGB uint8 array (rows from the top) as a PNG file."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def grid(path, xname, yname, zname):
    """A long-format CSV as sorted coordinates and a matrix Z[x, y]."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    xs, ix = np.unique(d[xname], return_inverse=True)
    ys, iy = np.unique(d[yname], return_inverse=True)
    z = np.full((len(xs), len(ys)), np.nan)
    z[ix, iy] = d[zname]
    return xs, ys, z


def density_image(data, out, amax=2.2, na=440):
    """The stationary joint density of (U, A) without the first processing cell, which holds the agents
    with A = 0 exactly, interpolated linearly in its logarithm onto an even grid in A, on a logarithmic
    colour scale; returns the range of log10 densities that the colours span."""
    us, a_s, z = grid(os.path.join(data, "extras", "stationary_joint_full.csv"), "u", "a", "density")
    a_s, z = a_s[1:], z[:, 1:]
    lz = np.log10(np.maximum(z, 1e-300))
    ag = np.linspace(0.0, amax, na)
    lg = np.empty((len(us), na))
    for i in range(len(us)):
        lg[i] = np.interp(ag, a_s, lz[i])          # constant beyond the ends of the solver's grid
    hi = float(lz.max())
    lo = hi - DECADES
    img = paint((lg - lo) / (hi - lo), DENSITY)     # [u, a, rgb]
    write_png(os.path.join(out, "consensus_density.png"), np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]))
    half = 0.5 * (us[1] - us[0])                    # the image spans whole cells: centres +- half a cell
    return lo, hi, float(us[0] - half), float(us[-1] + half)


def response_image(data, out, tmax=3.0, nt=600):
    """The change in the density of assessments after a unit common push, on an even time grid."""
    ts, us, z = grid(os.path.join(data, "extras", "response_field.csv"), "t", "u", "density_change")
    tg = np.linspace(0.0, tmax, nt)
    zg = np.empty((nt, len(us)))
    for j in range(len(us)):
        zg[:, j] = np.interp(tg, ts, z[:, j])
    lim = CLIP * float(np.nanmax(np.abs(z)))
    img = paint(0.5 * (1.0 + zg / lim), DIVERGING)  # [t, u, rgb]
    write_png(os.path.join(out, "consensus_response.png"), np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]))
    half = 0.5 * (us[1] - us[0])
    return lim, float(us[0] - half), float(us[-1] + half)


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    lo, hi, u0, u1 = density_image(data, out)
    lim, v0, v1 = response_image(data, out)
    macros = {"ConsLogLo": lo, "ConsLogHi": hi, "ConsRespLim": lim,
              "ConsUmin": u0, "ConsUmax": u1, "ConsVmin": v0, "ConsVmax": v1}
    with open(os.path.join(out, "raster_values.tex"), "w") as fh:
        fh.write("% Generated by rasters.py from the engine's output; do not edit.\n")
        for k, v in macros.items():
            fh.write("\\def\\%s{%.4f}\n" % (k, v))
        fh.write(pgf_colormap("consdensity", DENSITY))
        fh.write(pgf_colormap("consresponse", DIVERGING))
    print(" ".join(f"{k}={v:.4f}" for k, v in macros.items()))
