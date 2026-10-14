//! # inertia
//!
//! Numerical engine for *Aggregating Inertia: Tipping, Slow Shifts and What Revision Records Reveal*
//! (G. L. S. Bontemps). A Rust port of the paper's reference pipeline (Python and C), organised by
//! solver:
//!
//! * [`episode`]: one reassessment episode of the true economy B (assessment and processing on a
//!   (u, b) grid) and of its Markovian projection P; S_B, S_P, the area moments, the record laws.
//! * [`stationary`]: the forward generator of the stationary population on (u, a) with killing and
//!   reinjection; the response kernel k(t), its Laplace transform, the Nyquist curve, certificates.
//! * [`ghf`]: the one-dimensional solver for Markov completion rules (bifurcation diagrams).
//! * [`mc`]: Monte Carlo checks (episode, kernel, fluctuation-response, perturbation).
//! * [`pipeline`]: the computations behind every figure, table and quoted number.

pub mod episode;
pub mod ghf;
pub mod mc;
pub mod pipeline;
pub mod stationary;
pub mod util;
