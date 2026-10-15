//! Command-line entry point: every computation of the paper, stage by stage.
use anyhow::Result;
use clap::{Parser, Subcommand};
use inertia::pipeline::{self, extras, figures, research, tables, validation, Paths};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "inertia", version, about = "Computations for 'Aggregating Inertia: Tipping, Slow Shifts and What Revision Records Reveal'")]
struct Cli {
    /// Output folder (data/, data/grid/, data/validation/, data/extras/, tex/)
    #[arg(long, default_value = "output")]
    out: String,
    /// Number of threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Everything, in order: research grids, validation, figures, tables, extras
    All {
        /// Monte Carlo paths of the episode benchmark
        #[arg(long, default_value_t = 200_000)]
        paths: usize,
        /// Particles of the kernel, identity and perturbation checks
        #[arg(long, default_value_t = 400_000)]
        particles: usize,
        /// Agents of the population simulations (extras)
        #[arg(long, default_value_t = 50_000)]
        agents: usize,
    },
    /// Research grids: grid, variants, speed, scans, unimodality, certificates (or all)
    Research { which: Vec<String> },
    /// Validation: controls, refinement, running, identity, mc, kernel-mc, identities, perturbation (or all)
    Validate {
        which: Vec<String>,
        #[arg(long, default_value_t = 200_000)]
        paths: usize,
        #[arg(long, default_value_t = 400_000)]
        particles: usize,
    },
    /// Figure data, records.json, values.tex and tipping_values.tex (needs research)
    Figures,
    /// Table S1.1 and validation_values.tex (needs research and validation)
    Tables,
    /// Data for the showcase plots and the paper's simulation figures, and tex/extras_values.tex
    Extras {
        #[arg(long, default_value_t = 50_000)]
        agents: usize,
        /// Only rewrite tex/extras_values.tex from the extras already in the output folder
        #[arg(long)]
        values_only: bool,
    },
}

fn want(which: &[String], name: &str) -> bool {
    which.is_empty() || which.iter().any(|w| w == name || w == "all")
}

fn research_stage(p: &Paths, which: &[String]) -> Result<()> {
    if want(which, "grid") {
        research::grid(p, &research::LAMS)?;
    }
    if want(which, "variants") {
        research::variants(p)?;
    }
    if want(which, "speed") {
        research::speed_grid(p)?;
    }
    if want(which, "scans") {
        research::scans(p)?;
    }
    if want(which, "unimodality") {
        research::unimodality(p)?;
    }
    if want(which, "certificates") {
        research::certificates(p)?;
    }
    Ok(())
}

fn validation_stage(p: &Paths, which: &[String], paths: usize, particles: usize) -> Result<()> {
    if want(which, "controls") {
        validation::episode_controls(p)?;
    }
    if want(which, "refinement") {
        validation::stationary_refinement(p)?;
    }
    if want(which, "running") {
        validation::running_refinement(p)?;
    }
    if want(which, "identity") {
        validation::record_identity(p)?;
    }
    if want(which, "mc") {
        validation::mc_benchmark(p, paths)?;
    }
    if want(which, "kernel-mc") {
        validation::kernel_mc(p, particles)?;
    }
    if want(which, "identities") {
        validation::identity_checks(p, particles)?;
    }
    if want(which, "perturbation") {
        validation::perturbation(p, particles)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
    }
    let p = Paths::new(&cli.out);
    let t0 = Instant::now();
    match cli.cmd {
        Cmd::All { paths, particles, agents } => {
            research_stage(&p, &[])?;
            validation_stage(&p, &[], paths, particles)?;
            figures::run_all(&p)?;
            tables::run_all(&p)?;
            extras::run_all(&p, agents)?;
        }
        Cmd::Research { which } => research_stage(&p, &which)?,
        Cmd::Validate { which, paths, particles } => validation_stage(&p, &which, paths, particles)?,
        Cmd::Figures => figures::run_all(&p)?,
        Cmd::Tables => tables::run_all(&p)?,
        Cmd::Extras { agents, values_only } => {
            if values_only {
                extras::values(&p)?
            } else {
                extras::run_all(&p, agents)?
            }
        }
    }
    pipeline::log(&format!("done in {:.0}s", t0.elapsed().as_secs_f64()));
    Ok(())
}
