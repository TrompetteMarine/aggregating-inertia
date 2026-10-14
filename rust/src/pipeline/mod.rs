//! The computations behind every figure, table and number of the paper, in the order of the
//! reference pipeline:
//!
//! 1. [`research`]: the parameter grids of the episode solver (Section 5 and Table S1.1), the
//!    mean-lag grid, the scans of the stationary solver (first crossings, kernel signs) and the
//!    unimodality check;
//! 2. [`validation`]: the controls of both solvers (renewal limits, refinements, the discrete form of
//!    Theorem 3) and the Monte Carlo checks;
//! 3. [`figures`]: the data files of the figures and the quoted numbers (`make_data.py`);
//! 4. [`tables`]: Table S1.1 and the validation numbers (`make_tables.py`);
//! 5. [`extras`]: additional data for the showcase plots of the repository.
//!
//! Every stage reads only what earlier stages wrote, so stages can be rerun separately.

pub mod extras;
pub mod figures;
pub mod research;
pub mod tables;
pub mod validation;

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Output locations (mirroring the paper's replication folder).
#[derive(Clone, Debug)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let p = Paths { root: root.into() };
        for d in [p.data(), p.grid(), p.validation(), p.tex(), p.extras()] {
            fs::create_dir_all(d).ok();
        }
        p
    }
    pub fn data(&self) -> PathBuf {
        self.root.join("data")
    }
    pub fn grid(&self) -> PathBuf {
        self.root.join("data").join("grid")
    }
    pub fn validation(&self) -> PathBuf {
        self.root.join("data").join("validation")
    }
    pub fn tex(&self) -> PathBuf {
        self.root.join("tex")
    }
    pub fn extras(&self) -> PathBuf {
        self.root.join("data").join("extras")
    }
}

pub fn write_json(path: &Path, v: &Value) -> Result<()> {
    let s = serde_json::to_string_pretty(v)?;
    fs::write(path, s + "\n").with_context(|| format!("writing {}", path.display()))
}

pub fn read_json(path: &Path) -> Result<Value> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // the reference files were written by Python and may contain NaN
    let s = s.replace("NaN", "null").replace("Infinity", "1e999");
    Ok(serde_json::from_str(&s)?)
}

/// Write a CSV file with a header and every value formatted with `decimals` decimals (as numpy.savetxt).
pub fn write_csv(path: &Path, header: &str, cols: &[&[f64]], decimals: usize) -> Result<()> {
    let n = cols[0].len();
    let mut f = fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    writeln!(f, "{header}")?;
    for i in 0..n {
        let row: Vec<String> = cols.iter().map(|c| format!("{:.*}", decimals, c[i])).collect();
        writeln!(f, "{}", row.join(","))?;
    }
    Ok(())
}

/// Write a CSV file with every value in scientific notation with `digits` significant digits (for densities
/// that span many orders of magnitude, where a fixed number of decimals would round the tails to zero).
pub fn write_csv_sig(path: &Path, header: &str, cols: &[&[f64]], digits: usize) -> Result<()> {
    let n = cols[0].len();
    let mut f = fs::File::create(path).with_context(|| format!("writing {}", path.display()))?;
    writeln!(f, "{header}")?;
    for i in 0..n {
        let row: Vec<String> = cols.iter().map(|c| format!("{:.*e}", digits.saturating_sub(1), c[i])).collect();
        writeln!(f, "{}", row.join(","))?;
    }
    Ok(())
}

/// Read a numeric CSV with a header line into columns.
pub fn read_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut lines = s.lines();
    let header: Vec<String> = lines.next().unwrap_or("").split(',').map(|x| x.trim().to_string()).collect();
    let mut cols = vec![vec![]; header.len()];
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        for (k, v) in l.split(',').enumerate() {
            cols[k].push(v.trim().parse::<f64>().unwrap_or(f64::NAN));
        }
    }
    Ok((header, cols))
}

pub fn f(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(f64::NAN)
}

/// Python's format(x, "g") for the simple values used in macros (1.0 -> "1", 0.7 -> "0.7").
pub fn py_g(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e16 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

/// Thousands separator as Python's "{:,d}".
pub fn py_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Progress messages, on standard error.
pub fn log(msg: &str) {
    eprintln!("[inertia] {msg}");
}
