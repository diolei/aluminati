//! Free-form thermal SIMP: density-based topology optimization for
//! minimum thermal compliance at fixed aluminum volume. Conductivity
//! interpolates k(ρ) = k_min + ρ^p·(k_al − k_min); a cone filter keeps
//! mesh-independence; an optimality-criteria update enforces the volume.
//! Convection uses the uniform open-space rate (frozen-boundary): no
//! channel model exists for dendritic shapes — disclosed, not hidden.

use crate::physics::{
    assemble_hetero, pcg, steady_cg, steady_cg_hetero, thermal_resistance, ThermalParams, CP_AL,
    K_AL, RHO_AL,
};
use serde::Serialize;

pub const K_MIN_REL: f64 = 1e-3;
pub const K_MIN_ABS: f64 = K_MIN_REL * K_AL;

#[derive(Clone)]
pub struct SimpParams {
    pub cols: usize,
    pub rows: usize,
    pub dx: f64,
    pub thermal: ThermalParams,
    /// Bottom rows held solid (heat enters here).
    pub base_rows: usize,
    /// Target Σρ over design cells (above the base).
    pub volume_cells: f64,
    pub filter_radius: f64,
    pub penal: f64,
    pub move_limit: f64,
    pub max_iter: usize,
    pub snapshot_every: usize,
    /// Optional warm start: initial raw densities (base cells forced to 1).
    /// None = volume-matched uniform field.
    pub init: Option<Vec<f64>>,
    /// Optional volume continuation: start Σρ ramps linearly from this
    /// value down to volume_cells over the first half of the run. Lets
    /// features form in conductive high-volume gray, then tightens.
    pub volume_start_cells: Option<f64>,
}

/// Cone density filter with design mask. Filtering spans the
/// base/design boundary (standard); only design cells update.
pub struct DensityFilter {
    design: Vec<bool>,
    neighbors: Vec<Vec<(usize, f64)>>,
    norm: Vec<f64>,
}

pub fn build_filter(cols: usize, rows: usize, base_rows: usize, radius: f64) -> DensityFilter {
    let mut design = vec![false; cols * rows];
    for r in 0..rows - base_rows {
        for c in 0..cols {
            design[r * cols + c] = true;
        }
    }
    let mut neighbors = vec![Vec::new(); cols * rows];
    let mut norm = vec![0.0; cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            for rr in 0..rows {
                for cc in 0..cols {
                    let dist =
                        ((r as f64 - rr as f64).powi(2) + (c as f64 - cc as f64).powi(2)).sqrt();
                    if dist <= radius {
                        let w = radius - dist;
                        neighbors[k].push((rr * cols + cc, w));
                        norm[k] += w;
                    }
                }
            }
        }
    }
    DensityFilter {
        design,
        neighbors,
        norm,
    }
}

fn apply_filter(filter: &DensityFilter, rho: &[f64]) -> Vec<f64> {
    filter
        .neighbors
        .iter()
        .zip(filter.norm.iter())
        .map(|(nbs, n)| nbs.iter().map(|(j, w)| w * rho[*j]).sum::<f64>() / n)
        .collect()
}

/// Absolute conductivity of a filtered density (SIMP interpolation).
fn conductivity(rho_tilde: f64, penal: f64) -> f64 {
    K_MIN_ABS + rho_tilde.powf(penal) * (K_AL - K_MIN_ABS)
}

/// Full pipeline at raw densities: filter → convecting mask → hetero
/// solve → thermal compliance Σ Q_i·T_i. conv_mask_override fixes the
/// convecting set (smoothness for finite-difference validation);
/// None thresholds filtered densities at 0.5.
// Numeric pipeline: explicit arguments kept stable on purpose — the
// finite-difference oracle pins these signatures exactly, and bundling
// would churn the validated call graph for style only.
#[allow(clippy::too_many_arguments)]
fn pipeline(
    rho: &[f64],
    filter: &DensityFilter,
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    penal: f64,
    conv_mask_override: Option<&[bool]>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let tilde = apply_filter(filter, rho);
    let conv: Vec<bool> = match conv_mask_override {
        Some(m) => m.to_vec(),
        None => tilde.iter().map(|&v| v >= 0.5).collect(),
    };
    let cond: Vec<f64> = tilde.iter().map(|&v| conductivity(v, penal)).collect();
    let result = steady_cg_hetero(&conv, &cond, cols, rows, dx, params, fixed);
    // Source spreads over convecting bottom-row cells: collect CELL
    // indices, not column numbers (a col-vs-cell mix-up here silently
    // scores the top row — caught by the FD oracle).
    let base: Vec<usize> = (0..cols)
        .filter(|&c| conv[(rows - 1) * cols + c])
        .map(|c| (rows - 1) * cols + c)
        .collect();
    let share = if base.is_empty() {
        0.0
    } else {
        params.power / base.len() as f64
    };
    let compliance: f64 = base.iter().map(|&k| share * result.temps[k]).sum();
    (compliance, result.temps, tilde)
}

#[allow(clippy::too_many_arguments)]
fn pipeline_compliance(
    rho: &[f64],
    filter: &DensityFilter,
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    penal: f64,
) -> f64 {
    pipeline(rho, filter, cols, rows, dx, params, fixed, penal, None).0
}

/// Exact adjoint sensitivities of the pipeline w.r.t. RAW densities:
/// self-adjoint face loop with harmonic-conductance derivatives, then
/// the filter transpose (top88 pattern). Convection coupling is frozen
/// (no dh/dρ) — documented approximation, invisible to the FD oracle
/// only when the mask can't flip, hence the override in tests.
#[allow(clippy::too_many_arguments)]
fn raw_sensitivities(
    rho: &[f64],
    filter: &DensityFilter,
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    penal: f64,
) -> Vec<f64> {
    raw_sensitivities_masked(rho, filter, cols, rows, dx, params, fixed, penal, None)
}

#[allow(clippy::too_many_arguments)]
fn raw_sensitivities_masked(
    rho: &[f64],
    filter: &DensityFilter,
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    penal: f64,
    conv_mask_override: Option<&[bool]>,
) -> Vec<f64> {
    let tilde = apply_filter(filter, rho);
    let conv: Vec<bool> = match conv_mask_override {
        Some(m) => m.to_vec(),
        None => tilde.iter().map(|&v| v >= 0.5).collect(),
    };
    let ktilde: Vec<f64> = tilde
        .iter()
        .map(|&v| v.powf(penal) * (1.0 - K_MIN_REL) + K_MIN_REL)
        .collect();
    let cond: Vec<f64> = ktilde.iter().map(|&k| k * K_AL).collect();
    let (diag, off, _rhs, idx) = assemble_hetero(&conv, &cond, cols, rows, dx, params, fixed);
    // Forward solve.
    let (x, _, _) = pcg(&diag, &off, &_rhs, params.t_inf);
    let mut t = vec![params.t_inf; cols * rows];
    for (k, f) in fixed.iter().enumerate() {
        if let Some(v) = f {
            t[k] = *v;
        } else {
            t[k] = x[idx[k]];
        }
    }
    // Adjoint solve K·λ = c: the compliance c = Σ_base share·T only
    // loads the base cells, while the forward right-hand side also
    // carries convection Tinf terms — so λ ≠ T and a second solve is
    // mandatory. (Assuming self-adjointness here fails the FD oracle
    // at ~100%.) The share matches the pipeline exactly: spread over ALL
    // convecting bottom-row cells (fixed ones contribute constants, so
    // they carry no load but DO dilute the share).
    let base_all = (0..cols).filter(|&c| conv[(rows - 1) * cols + c]).count();
    let base: Vec<usize> = (0..cols)
        .filter(|&c| conv[(rows - 1) * cols + c] && fixed[(rows - 1) * cols + c].is_none())
        .map(|c| idx[(rows - 1) * cols + c])
        .collect();
    let mut cvec = vec![0.0; x.len()];
    if !base.is_empty() && base_all > 0 {
        let share = params.power / base_all as f64;
        for &i in &base {
            cvec[i] = share;
        }
    }
    let (lam, _, _) = pcg(&diag, &off, &cvec, 0.0);
    let mut lambda = vec![0.0; cols * rows];
    for k in 0..cols * rows {
        if idx[k] != usize::MAX {
            lambda[k] = lam[idx[k]];
        }
    }

    // dC/dρ̃ via faces: dC = −Σ_faces (dg̃/dρ̃)(ΔT)(Δλ) with the harmonic
    // face conductance g̃ = 2·k̃_a·k̃_b/(k̃_a+k̃_b); each face contributes
    // to both endpoint cells through its own endpoint derivative.
    let dkdrho = |rt: f64| penal * rt.powf(penal - 1.0) * (1.0 - K_MIN_REL);
    let mut sens = vec![0.0; cols * rows];
    let mut face = |a: usize, b: usize| {
        let ka = ktilde[a];
        let kb = ktilde[b];
        let mixed = (t[a] - t[b]) * (lambda[a] - lambda[b]);
        let denom = (ka + kb).powi(2);
        sens[a] -= 2.0 * kb * kb / denom * dkdrho(tilde[a]) * mixed;
        sens[b] -= 2.0 * ka * ka / denom * dkdrho(tilde[b]) * mixed;
    };
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if c + 1 < cols {
                face(k, k + 1);
            }
            if r + 1 < rows {
                face(k, k + cols);
            }
        }
    }
    // Filter transpose (top88): sens_raw = H·(sens/Hs).
    let mut raw = vec![0.0; cols * rows];
    for (i, nbs) in filter.neighbors.iter().enumerate() {
        let s = sens[i] / filter.norm[i];
        for (j, w) in nbs {
            raw[*j] += w * s;
        }
    }
    // Passive base never updates.
    for (k, d) in filter.design.iter().enumerate() {
        if !d {
            raw[k] = 0.0;
        }
    }
    raw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{ThermalParams, CP_AL, K_AL, RHO_AL};

    fn params() -> ThermalParams {
        ThermalParams {
            k: K_AL,
            rho: RHO_AL,
            cp: CP_AL,
            h: 25.0,
            channel_h: None,
            t_inf: 300.0,
            power: 20.0,
        }
    }

    fn small() -> SimpParams {
        SimpParams {
            cols: 24,
            rows: 12,
            dx: 0.0025,
            thermal: params(),
            base_rows: 2,
            volume_cells: 60.0,
            filter_radius: 1.5,
            penal: 3.0,
            move_limit: 0.2,
            max_iter: 20,
            snapshot_every: 5,
            init: None,
            volume_start_cells: None,
        }
    }

    // Bridge oracle: heterogeneous conduction with a binary density field
    // must reproduce the thresholded uniform solve (leakage through k_min
    // is the only difference).
    #[test]
    fn hetero_matches_thresholded_on_binary() {
        use crate::physics::{steady_cg, thermal_resistance};
        let cols = 12;
        let rows = 6;
        let dx = 0.0025;
        let p = params();
        // Block with a void window: sharp binary contrast.
        let mut solid = vec![true; cols * rows];
        for r in 1..3 {
            for c in 4..8 {
                solid[r * cols + c] = false;
            }
        }
        let fixed = vec![None; cols * rows];
        let direct = steady_cg(&solid, cols, rows, dx, &p, &fixed);
        assert!(direct.converged);

        let cond: Vec<f64> = solid
            .iter()
            .map(|&s| if s { K_AL } else { K_MIN_ABS })
            .collect();
        let hetero = steady_cg_hetero(&solid, &cond, cols, rows, dx, &p, &fixed);
        assert!(hetero.converged);
        let a = thermal_resistance(&direct, &p);
        let b = thermal_resistance(&hetero, &p);
        assert!((a - b) / a < 0.03, "hetero {b} vs uniform {a}");
    }

    // Gold-standard adjoint check: analytic sensitivities against central
    // differences of the full pipeline (filter + solve + compliance).
    // Densities stay clear of the 0.5 convection threshold so frozen
    // boundaries don't pollute the difference.
    #[test]
    fn sensitivities_match_finite_differences() {
        let cols = 8;
        let rows = 5;
        let dx = 0.0025;
        let p = params();
        let fixed = vec![None; cols * rows];
        let filter = build_filter(cols, rows, 1, 1.5);
        let rho: Vec<f64> = (0..cols * rows)
            .map(|k| {
                let v = ((k * 7 + 3) % 5) as f64 / 5.0;
                0.15 + 0.55 * v // 0.15..0.70, never near 0.5 after filter
            })
            .collect();
        let analytic = raw_sensitivities_masked(
            &rho,
            &filter,
            cols,
            rows,
            dx,
            &p,
            &fixed,
            3.0,
            Some(&vec![true; cols * rows]),
        );
        let eps = 1e-6;
        let mut worst = 0.0f64;
        let all_solid = vec![true; cols * rows];
        // Design cells only: the passive base row gets exactly zero by
        // construction (the OC never updates it) while its true
        // derivative is nonzero — asserted separately below.
        for k in [0, 7, 13, 22] {
            let mut up = rho.clone();
            up[k] += eps;
            let mut dn = rho.clone();
            dn[k] -= eps;
            let fd = (pipeline(
                &up,
                &filter,
                cols,
                rows,
                dx,
                &p,
                &fixed,
                3.0,
                Some(&all_solid),
            )
            .0 - pipeline(
                &dn,
                &filter,
                cols,
                rows,
                dx,
                &p,
                &fixed,
                3.0,
                Some(&all_solid),
            )
            .0) / (2.0 * eps);
            let rel = ((analytic[k] - fd) / fd.abs().max(1e-12)).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 1e-3, "worst sensitivity error {worst}");
        for (k, v) in analytic.iter().enumerate().skip(32).take(8) {
            assert_eq!(*v, 0.0, "passive cell {k} must read zero");
        }
    }

    // OC bisection holds the volume by construction; pin it.
    #[test]
    fn oc_preserves_volume() {
        let p = small();
        let result = run_simp(&p);
        for f in &result.frames {
            assert!(
                (f.volume - p.volume_cells).abs() < 1e-6,
                "iter {} volume {}",
                f.iter,
                f.volume
            );
        }
    }

    // Symmetric problem, symmetric start: any asymmetry is a bug.
    #[test]
    fn simp_result_is_mirror_symmetric() {
        let p = small();
        let result = run_simp(&p);
        let last = &result.frames[result.frames.len() - 1];
        for r in 0..p.rows {
            for c in 0..p.cols / 2 {
                let a = last.densities[r * p.cols + c];
                let b = last.densities[r * p.cols + (p.cols - 1 - c)];
                assert!((a - b).abs() < 1e-9, "asymmetric at ({r},{c})");
            }
        }
    }

    // The point of optimizing: beat the uniform gray start.
    #[test]
    fn simp_improves_over_uniform() {
        let p = small();
        let result = run_simp(&p);
        assert!(
            result.final_compliance < result.initial_compliance,
            "final {} vs initial {}",
            result.final_compliance,
            result.initial_compliance
        );
    }

    #[test]
    fn simp_is_deterministic() {
        let p = small();
        let a = run_simp(&p);
        let b = run_simp(&p);
        let la = &a.frames[a.frames.len() - 1];
        let lb = &b.frames[b.frames.len() - 1];
        assert_eq!(la.densities, lb.densities);
    }

    #[test]
    fn snapshots_cover_the_run() {
        let p = small();
        let result = run_simp(&p);
        assert_eq!(result.frames[0].iter, 0);
        assert_eq!(result.frames[result.frames.len() - 1].iter, p.max_iter);
        for f in &result.frames {
            assert_eq!(f.densities.len(), p.cols * p.rows);
            assert_eq!(f.field.len(), p.cols * p.rows);
            assert!(f.field.iter().all(|t| t.is_finite()));
        }
    }

    #[test]
    fn warm_start_seeds_frame_zero() {
        let mut p = small();
        p.init = Some(vec![1.0; p.cols * p.rows]);
        let result = run_simp(&p);
        // Solid start: frame 0 holds the full design volume (over budget
        // by design), then OC pulls it to the target.
        let design = (p.rows - p.base_rows) * p.cols;
        assert!((result.frames[0].volume - design as f64).abs() < 1e-6);
        let last = &result.frames[result.frames.len() - 1];
        assert!((last.volume - p.volume_cells).abs() < 1e-6);
    }

    #[test]
    fn chunked_stepping_matches_one_shot() {
        let p = small();
        let whole = run_simp(&p);
        let mut run = SimpRun::new(p.clone());
        let mut frames = Vec::new();
        frames.extend(run.step(3));
        while !run.done() {
            frames.extend(run.step(7));
        }
        assert_eq!(frames.len(), whole.frames.len());
        for (a, b) in frames.iter().zip(whole.frames.iter()) {
            assert_eq!(a.iter, b.iter);
            assert_eq!(a.densities, b.densities);
            assert_eq!(a.field, b.field);
        }
    }

    #[test]
    fn thresholded_rth_is_self_consistent() {
        let p = small();
        let mut run = SimpRun::new(p.clone());
        while !run.done() {
            run.step(5);
        }
        let (rth, tmax) = run.thresholded_rth().expect("final must solve");
        assert!(rth.is_finite() && rth > 0.0 && tmax.is_finite());
        // Independent recomputation: threshold + uniform solve + Rth,
        // without touching the method under test.
        let filter = build_filter(p.cols, p.rows, p.base_rows, p.filter_radius);
        let tilde = apply_filter(&filter, run.densities());
        let mask: Vec<bool> = tilde.iter().map(|&v| v >= 0.5).collect();
        let fixed = vec![None; p.cols * p.rows];
        let s = crate::physics::steady_cg(&mask, p.cols, p.rows, p.dx, &p.thermal, &fixed);
        assert!(s.converged);
        let expected = crate::physics::thermal_resistance(&s, &p.thermal);
        assert!((rth - expected).abs() < 1e-12, "{rth} vs {expected}");
    }

    #[test]
    fn volume_ramp_descends_to_target() {
        let mut p = small();
        p.volume_start_cells = Some(2.0 * p.volume_cells);
        let result = run_simp(&p);
        assert!(
            (result.frames[0].volume - 2.0 * p.volume_cells).abs() < 1e-6,
            "ramp must start matched, got {}",
            result.frames[0].volume
        );
        let last = &result.frames[result.frames.len() - 1];
        assert!((last.volume - p.volume_cells).abs() < 1e-6);
    }
}

#[derive(Serialize, Clone)]
pub struct SimpFrameView {
    pub iter: usize,
    pub densities: Vec<f32>,
    pub field: Vec<f32>,
    pub compliance: f64,
    pub volume: f64,
}

pub struct SimpResult {
    pub frames: Vec<SimpFrameView>,
    pub initial_compliance: f64,
    pub final_compliance: f64,
}

/// Optimality-criteria update (top88 pattern): design cells move toward
/// B^η with move limits, λ bisected so filtered design volume hits the
/// target. B = −sens/λ; sens is negative (more material always helps),
/// so B > 0.
fn oc_update(
    rho: &[f64],
    sens: &[f64],
    filter: &DensityFilter,
    target: f64,
    move_limit: f64,
) -> Vec<f64> {
    let mut l1 = 0.0;
    let mut l2 = 1e9;
    let mut out = rho.to_vec();
    for _ in 0..200 {
        let lmid = 0.5 * (l1 + l2);
        let mut vol = 0.0;
        for (k, d) in filter.design.iter().enumerate() {
            if !d {
                continue;
            }
            let b = (-sens[k] / lmid).max(0.0).sqrt();
            out[k] = (rho[k] * b)
                .max(rho[k] - move_limit)
                .min(rho[k] + move_limit)
                .clamp(0.0, 1.0);
        }
        // Volume counts FILTERED design densities (physical material).
        let tilde = apply_filter(filter, &out);
        for (k, d) in filter.design.iter().enumerate() {
            if *d {
                vol += tilde[k];
            }
        }
        if vol > target {
            l1 = lmid;
        } else {
            l2 = lmid;
        }
        if (l2 - l1) / (l1 + l2).max(1e-300) < 1e-9 {
            break;
        }
    }
    out
}

/// Locked production SIMP: 120×60 mm domain at 48×24, same 240-cell
/// aluminum as the half-scale sweep reference (144 base + 96 free).
/// Volume continuation 0.35 → 0.095, 120 iterations in 5-iteration
/// chunks for progressive forging in the widget.
pub fn production_simp_params() -> SimpParams {
    SimpParams {
        cols: 48,
        rows: 24,
        dx: 0.0025,
        thermal: ThermalParams {
            k: K_AL,
            rho: RHO_AL,
            cp: CP_AL,
            h: 25.0,
            channel_h: None,
            t_inf: 300.0,
            power: 100.0,
        },
        base_rows: 3,
        volume_cells: 96.0,
        filter_radius: 1.5,
        penal: 3.0,
        move_limit: 0.3,
        max_iter: 200,
        snapshot_every: 5,
        init: None,
        volume_start_cells: Some(0.35 * 21.0 * 48.0),
    }
}

/// Stateful SIMP run for chunked forging: the widget advances a few
/// iterations per frame and renders progress live, instead of blocking
/// on one long call. Chunked stepping reproduces the one-shot run
/// exactly (same order, same snapshots).
pub struct SimpRun {
    p: SimpParams,
    filter: DensityFilter,
    fixed: Vec<Option<f64>>,
    rho: Vec<f64>,
    prev_sens: Vec<f64>,
    iter: usize,
    initial_compliance: f64,
    final_compliance: f64,
}

impl SimpRun {
    pub fn new(p: SimpParams) -> Self {
        let filter = build_filter(p.cols, p.rows, p.base_rows, p.filter_radius);
        let init_target = match p.volume_start_cells {
            Some(v0) => v0,
            None => p.volume_cells,
        };
        let mut rho = vec![0.0; p.cols * p.rows];
        match &p.init {
            Some(seed) => {
                for (k, d) in filter.design.iter().enumerate() {
                    rho[k] = if *d { seed[k].clamp(0.0, 1.0) } else { 1.0 };
                }
            }
            None => {
                let design_volume = |rr: &[f64]| -> f64 {
                    apply_filter(&filter, rr)
                        .iter()
                        .zip(filter.design.iter())
                        .filter(|pair| *pair.1)
                        .map(|(v, _)| *v)
                        .sum()
                };
                let mut probe0 = vec![0.0; p.cols * p.rows];
                let mut probe1 = vec![0.0; p.cols * p.rows];
                for (k, d) in filter.design.iter().enumerate() {
                    if !d {
                        probe0[k] = 1.0;
                        probe1[k] = 1.0;
                    } else {
                        probe1[k] = 1.0;
                    }
                }
                let b = design_volume(&probe0);
                let a = design_volume(&probe1) - b;
                let v0 = ((init_target - b) / a).clamp(0.0, 1.0);
                for (k, d) in filter.design.iter().enumerate() {
                    rho[k] = if *d { v0 } else { 1.0 };
                }
            }
        }
        let fixed = vec![None; p.cols * p.rows];
        // Reference compliance at full penalization (matches the final
        // schedule stage, so improvement comparisons are like-for-like).
        let initial_compliance = pipeline_compliance(
            &rho, &filter, p.cols, p.rows, p.dx, &p.thermal, &fixed, p.penal,
        );
        SimpRun {
            p,
            filter,
            fixed,
            rho,
            prev_sens: vec![0.0; 0],
            iter: 0,
            initial_compliance,
            final_compliance: 0.0,
        }
    }

    pub fn done(&self) -> bool {
        self.iter > self.p.max_iter
    }

    pub fn params(&self) -> &SimpParams {
        &self.p
    }

    pub fn densities(&self) -> &[f64] {
        &self.rho
    }

    /// Thermal resistance of the current densities via a thresholded
    /// (ρ ≥ 0.5) uniform-conductivity solve — the headline number for
    /// the finished design. Uniform open-space convection, same as the
    /// run itself.
    pub fn thresholded_rth(&self) -> Option<(f64, f64)> {
        let tilde = apply_filter(&self.filter, &self.rho);
        let mask: Vec<bool> = tilde.iter().map(|&v| v >= 0.5).collect();
        let fixed = vec![None; self.p.cols * self.p.rows];
        let result = steady_cg(
            &mask,
            self.p.cols,
            self.p.rows,
            self.p.dx,
            &self.p.thermal,
            &fixed,
        );
        if !result.converged {
            return None;
        }
        Some((thermal_resistance(&result, &self.p.thermal), result.t_max))
    }

    /// Advances up to n_iters iterations, returning newly recorded frames.
    pub fn step(&mut self, n_iters: usize) -> Vec<SimpFrameView> {
        if self.prev_sens.is_empty() {
            self.prev_sens = vec![0.0; self.p.cols * self.p.rows];
        }
        let mut frames = Vec::new();
        for _ in 0..n_iters {
            if self.done() {
                break;
            }
            let iter = self.iter;
            let penal = self.p.penal_at(iter);
            let half = (self.p.max_iter / 2).max(1) as f64;
            let target = match self.p.volume_start_cells {
                Some(v0) if (iter as f64) < half => {
                    v0 + (self.p.volume_cells - v0) * (iter as f64 / half)
                }
                _ => self.p.volume_cells,
            };
            let tilde = apply_filter(&self.filter, &self.rho);
            let (compliance, temps, _) = pipeline(
                &self.rho,
                &self.filter,
                self.p.cols,
                self.p.rows,
                self.p.dx,
                &self.p.thermal,
                &self.fixed,
                penal,
                None,
            );
            self.final_compliance = compliance;
            if iter.is_multiple_of(self.p.snapshot_every) || iter == self.p.max_iter {
                let volume: f64 = tilde
                    .iter()
                    .zip(self.filter.design.iter())
                    .filter(|pair| *pair.1)
                    .map(|(v, _)| *v)
                    .sum();
                frames.push(SimpFrameView {
                    iter,
                    densities: tilde.iter().map(|&v| v as f32).collect(),
                    field: temps.iter().map(|&t| t as f32).collect(),
                    compliance,
                    volume,
                });
            }
            if iter == self.p.max_iter {
                self.iter = iter + 1;
                break;
            }
            let sens = raw_sensitivities(
                &self.rho,
                &self.filter,
                self.p.cols,
                self.p.rows,
                self.p.dx,
                &self.p.thermal,
                &self.fixed,
                penal,
            );
            let damped: Vec<f64> = sens
                .iter()
                .zip(self.prev_sens.iter())
                .map(|(s, o)| 0.5 * (s + o))
                .collect();
            self.prev_sens = sens;
            self.rho = oc_update(&self.rho, &damped, &self.filter, target, self.p.move_limit);
            self.iter = iter + 1;
        }
        frames
    }
}

impl SimpParams {
    /// Penalization continuation thirds: convex start, ramp, full SIMP.
    pub fn penal_at(&self, iter: usize) -> f64 {
        if iter * 3 < self.max_iter {
            1.0
        } else if iter * 3 < self.max_iter * 2 {
            2.0
        } else {
            self.penal
        }
    }
}

/// One-shot run: drives a chunked SimpRun to completion. Identical
/// frames to manual stepping (same order, same snapshots).
pub fn run_simp(p: &SimpParams) -> SimpResult {
    let mut run = SimpRun::new(p.clone());
    let initial_compliance = run.initial_compliance;
    let mut frames = Vec::new();
    while !run.done() {
        frames.extend(run.step(1));
    }
    SimpResult {
        frames,
        initial_compliance,
        final_compliance: run.final_compliance,
    }
}
