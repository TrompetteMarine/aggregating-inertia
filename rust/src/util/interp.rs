//! Interpolation with numpy / scipy semantics.

/// numpy.interp(x, xp, fp): piecewise linear, constant beyond the ends; xp increasing.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x <= xp[0] {
        return if x < xp[0] { fp[0] } else { fp[0] };
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // largest j with xp[j] <= x
    let j = match xp.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(k) => return fp[k],
        Err(k) => k - 1,
    };
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

pub fn interp_many(xs: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    xs.iter().map(|&x| interp(x, xp, fp)).collect()
}

/// Shape-preserving piecewise cubic Hermite interpolation, as scipy.interpolate.PchipInterpolator.
pub struct Pchip {
    x: Vec<f64>,
    // per interval: Horner coefficients in s = x - x_k
    c: Vec<[f64; 4]>,
}

fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn edge_case(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    let mask = sign(d) != sign(m0);
    let mask2 = sign(m0) != sign(m1) && d.abs() > 3.0 * m0.abs();
    if mask {
        d = 0.0;
    } else if mask2 {
        d = 3.0 * m0;
    }
    d
}

impl Pchip {
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        assert!(n >= 2);
        let hk: Vec<f64> = (0..n - 1).map(|k| x[k + 1] - x[k]).collect();
        let mk: Vec<f64> = (0..n - 1).map(|k| (y[k + 1] - y[k]) / hk[k]).collect();
        let mut dk = vec![0.0; n];
        if n == 2 {
            dk[0] = mk[0];
            dk[1] = mk[0];
        } else {
            for k in 1..n - 1 {
                let (m0, m1) = (mk[k - 1], mk[k]);
                let cond = sign(m1) != sign(m0) || m1 == 0.0 || m0 == 0.0;
                if cond {
                    dk[k] = 0.0;
                } else {
                    let w1 = 2.0 * hk[k] + hk[k - 1];
                    let w2 = hk[k] + 2.0 * hk[k - 1];
                    let whmean = (w1 / m0 + w2 / m1) / (w1 + w2);
                    dk[k] = 1.0 / whmean;
                }
            }
            dk[0] = edge_case(hk[0], hk[1], mk[0], mk[1]);
            dk[n - 1] = edge_case(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
        }
        // CubicHermiteSpline coefficients
        let mut c = Vec::with_capacity(n - 1);
        for k in 0..n - 1 {
            let dx = hk[k];
            let slope = (y[k + 1] - y[k]) / dx;
            let t = (dk[k] + dk[k + 1] - 2.0 * slope) / dx;
            c.push([t / dx, (slope - dk[k]) / dx - t, dk[k], y[k]]);
        }
        Pchip { x: x.to_vec(), c }
    }

    pub fn eval(&self, xv: f64) -> f64 {
        let n = self.x.len();
        // interval: x[k] <= xv < x[k+1]; the right end belongs to the last interval
        let k = if xv >= self.x[n - 1] {
            n - 2
        } else if xv <= self.x[0] {
            0
        } else {
            match self.x.binary_search_by(|v| v.partial_cmp(&xv).unwrap()) {
                Ok(k) => k.min(n - 2),
                Err(k) => k - 1,
            }
        };
        let s = xv - self.x[k];
        let c = &self.c[k];
        ((c[0] * s + c[1]) * s + c[2]) * s + c[3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pchip_reproduces_nodes_and_is_monotone() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 1.5, 4.0];
        let p = Pchip::new(&x, &y);
        for k in 0..4 {
            assert!((p.eval(x[k]) - y[k]).abs() < 1e-14);
        }
        let mut prev = -1.0;
        for i in 0..=300 {
            let v = p.eval(i as f64 * 0.01);
            assert!(v >= prev - 1e-14);
            prev = v;
        }
    }
    #[test]
    fn interp_basic() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 20.0];
        assert_eq!(interp(-1.0, &xp, &fp), 0.0);
        assert_eq!(interp(0.5, &xp, &fp), 5.0);
        assert_eq!(interp(1.0, &xp, &fp), 10.0);
        assert_eq!(interp(3.0, &xp, &fp), 20.0);
    }
}
