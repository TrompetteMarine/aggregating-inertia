//! Brent's method, transcribed from scipy's brentq.c (same steps, same defaults), so that roots agree
//! with the reference to the last iteration.

#[derive(Debug, Clone, Copy)]
pub struct BrentOpts {
    pub xtol: f64,
    pub rtol: f64,
    pub maxiter: usize,
}

impl Default for BrentOpts {
    fn default() -> Self {
        BrentOpts { xtol: 2e-12, rtol: 4.0 * f64::EPSILON, maxiter: 100 }
    }
}

#[derive(Debug)]
pub enum RootError {
    SameSign,
    NotConverged(f64),
}

impl std::fmt::Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::SameSign => write!(f, "f(a) and f(b) must have different signs"),
            RootError::NotConverged(x) => write!(f, "Brent's method did not converge (last x = {x})"),
        }
    }
}

impl std::error::Error for RootError {}

pub fn brentq<F: FnMut(f64) -> f64>(mut f: F, xa: f64, xb: f64, opts: BrentOpts) -> Result<f64, RootError> {
    let (mut xpre, mut xcur) = (xa, xb);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(RootError::SameSign);
    }
    for _ in 0..opts.maxiter {
        if fpre != 0.0 && fcur != 0.0 && (fpre.is_sign_negative() != fcur.is_sign_negative()) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (opts.xtol + opts.rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    Err(RootError::NotConverged(xcur))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finds_sqrt2() {
        let r = brentq(|x| x * x - 2.0, 0.0, 2.0, BrentOpts::default()).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-12);
    }
}
