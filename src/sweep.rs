//! Parametric straight-fin sweep under a fixed aluminum area.
//!
//! Family: solid base plate plus n identical rectangular fins. Given fin
//! count and width, height follows from the aluminum budget, so every
//! candidate spends at most the fixed amount. The solid block (n = 0) at
//! the same budget is the baseline fins must beat.

use crate::physics::{
    elenbaas_h, steady_cg, thermal_resistance, ThermalParams, CP_AL, K_AL, RHO_AL,
};
use serde::Serialize;

/// Fin layout in cell units. n_fins = 0 is the solid block baseline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FinGeometry {
    pub n_fins: u8,
    pub fin_width: usize,
    pub fin_height: usize,
}

pub struct Candidate {
    pub geometry: FinGeometry,
    pub area_cells: usize,
    pub rth: f64,
    pub t_max: f64,
    pub converged: bool,
}

pub struct SweepParams {
    pub cols: usize,
    pub rows: usize,
    pub dx: f64,
    pub thermal: ThermalParams,
    /// Solid base-plate thickness for finned designs, in rows.
    pub base_rows: usize,
    /// Aluminum budget: no candidate may exceed this many solid cells.
    pub aluminum_cells: usize,
    pub max_fin_count: u8,
    pub fin_widths: Vec<usize>,
}

pub struct SweepResult {
    pub baseline: Candidate,
    pub best: Candidate,
    pub candidates: Vec<Candidate>,
    pub evals: usize,
}

/// Builds the solid mask for a geometry: base plate at the bottom, fins
/// evenly spaced above it with at least one cell gap. Returns None when
/// the fins cannot fit (gap < 1) or the budget leaves no height.
pub fn mask_for(params: &SweepParams, geometry: FinGeometry) -> Option<(Vec<bool>, usize)> {
    let mut mask = vec![false; params.cols * params.rows];
    if geometry.n_fins == 0 {
        let block_rows = params.aluminum_cells / params.cols;
        if block_rows == 0 || block_rows > params.rows {
            return None;
        }
        for r in params.rows - block_rows..params.rows {
            for c in 0..params.cols {
                mask[r * params.cols + c] = true;
            }
        }
        return Some((mask, block_rows * params.cols));
    }
    let base_area = params.base_rows * params.cols;
    if base_area >= params.aluminum_cells {
        return None;
    }
    let n = geometry.n_fins as usize;
    let wf = geometry.fin_width;
    if params.cols < n * (wf + 1) + 1 {
        return None;
    }
    let height = (params.aluminum_cells - base_area) / (n * wf);
    let top = params.rows - params.base_rows;
    if height == 0 || height > top {
        return None;
    }
    for r in params.rows - params.base_rows..params.rows {
        for c in 0..params.cols {
            mask[r * params.cols + c] = true;
        }
    }
    // Even spacing with integer remainder spread left to right.
    let total_gap = params.cols - n * wf;
    let mut x = 0;
    for i in 0..n {
        x += total_gap / (n + 1);
        if i < total_gap % (n + 1) {
            x += 1;
        }
        for rr in top - height..top {
            for cc in x..x + wf {
                mask[rr * params.cols + cc] = true;
            }
        }
        x += wf;
    }
    let area = mask.iter().filter(|&&s| s).count();
    Some((mask, area))
}

fn evaluate(params: &SweepParams, geometry: FinGeometry) -> Option<Candidate> {
    let (mask, area) = mask_for(params, geometry)?;
    if area > params.aluminum_cells {
        return None;
    }
    let fixed = vec![None; params.cols * params.rows];
    // Pass 1 at the open-space rate gives the ΔT the Elenbaas number
    // needs; pass 2 applies the confined-channel rate to gap walls.
    let first = steady_cg(
        &mask,
        params.cols,
        params.rows,
        params.dx,
        &params.thermal,
        &fixed,
    );
    if !first.converged {
        return None;
    }
    let mut thermal = params.thermal;
    if geometry.n_fins > 0 {
        let n = geometry.n_fins as usize;
        let gap_cells = (params.cols - n * geometry.fin_width) / (n + 1);
        thermal.channel_h = Some(elenbaas_h(
            gap_cells as f64 * params.dx,
            geometry.fin_height as f64 * params.dx,
            (first.t_max - params.thermal.t_inf).max(1.0),
            params.thermal.h,
        ));
    }
    let result = steady_cg(&mask, params.cols, params.rows, params.dx, &thermal, &fixed);
    if !result.converged {
        return None;
    }
    Some(Candidate {
        geometry,
        area_cells: area,
        rth: thermal_resistance(&result, &params.thermal),
        t_max: result.t_max,
        converged: true,
    })
}

/// Locked production problem: 120×60 mm domain, 6063 aluminum,
/// 100 W base load, natural convection, 960-cell budget. The benchmarked
/// answer is 4 fins (see the flagship test below).
pub fn production_params() -> SweepParams {
    SweepParams {
        cols: 96,
        rows: 48,
        dx: 0.00125,
        thermal: ThermalParams {
            k: K_AL,
            rho: RHO_AL,
            cp: CP_AL,
            h: 25.0,
            channel_h: None,
            t_inf: 300.0,
            power: 100.0,
        },
        base_rows: 6,
        aluminum_cells: 960,
        max_fin_count: 10,
        fin_widths: vec![1, 2, 3, 4],
    }
}

#[derive(Serialize, Clone)]
pub struct CandidateView {
    pub n_fins: u8,
    pub fin_width: usize,
    pub fin_height: usize,
    pub area_cells: usize,
    pub rth: f64,
    pub t_max: f64,
}

#[derive(Serialize)]
pub struct FrameView {
    pub candidate: CandidateView,
    /// Row-major solid mask: the geometry this frame carves.
    pub mask: Vec<u8>,
}

#[derive(Serialize)]
pub struct HistoryView {
    pub cols: usize,
    pub rows: usize,
    pub dx: f64,
    pub baseline: CandidateView,
    /// The winner WITH its geometry mask: the field renderer needs it to
    /// tell cold solid apart from air.
    pub best: FrameView,
    /// Best-so-far geometry per improvement: the playback arc.
    pub frames: Vec<FrameView>,
    pub candidates: Vec<CandidateView>,
    pub baseline_field: Vec<f32>,
    pub winner_field: Vec<f32>,
}

fn view_of(c: &Candidate) -> CandidateView {
    CandidateView {
        n_fins: c.geometry.n_fins,
        fin_width: c.geometry.fin_width,
        fin_height: c.geometry.fin_height,
        area_cells: c.area_cells,
        rth: c.rth,
        t_max: c.t_max,
    }
}

/// Re-solves one geometry through both passes and returns its field.
fn field_for(params: &SweepParams, geometry: FinGeometry) -> Option<Vec<f32>> {
    let (mask, _) = mask_for(params, geometry)?;
    let fixed = vec![None; params.cols * params.rows];
    let first = steady_cg(
        &mask,
        params.cols,
        params.rows,
        params.dx,
        &params.thermal,
        &fixed,
    );
    if !first.converged {
        return None;
    }
    let mut thermal = params.thermal;
    if geometry.n_fins > 0 {
        let n = geometry.n_fins as usize;
        let gap_cells = (params.cols - n * geometry.fin_width) / (n + 1);
        thermal.channel_h = Some(elenbaas_h(
            gap_cells as f64 * params.dx,
            geometry.fin_height as f64 * params.dx,
            (first.t_max - params.thermal.t_inf).max(1.0),
            params.thermal.h,
        ));
    }
    let result = steady_cg(&mask, params.cols, params.rows, params.dx, &thermal, &fixed);
    if !result.converged {
        return None;
    }
    Some(result.temps.iter().map(|&t| t as f32).collect())
}

/// Runs the production sweep and packages everything the widget needs
/// in one call: no per-generation FFI chatter.
pub fn history_view() -> HistoryView {
    let params = production_params();
    let result = sweep(&params);
    let mut frames = vec![FrameView {
        candidate: view_of(&result.baseline),
        mask: mask_of(&params, result.baseline.geometry),
    }];
    let mut running = result.baseline.rth;
    for c in &result.candidates {
        if c.rth < running {
            running = c.rth;
            frames.push(FrameView {
                candidate: view_of(c),
                mask: mask_of(&params, c.geometry),
            });
        }
    }
    HistoryView {
        cols: params.cols,
        rows: params.rows,
        dx: params.dx,
        baseline_field: field_for(&params, result.baseline.geometry).unwrap_or_default(),
        winner_field: field_for(&params, result.best.geometry).unwrap_or_default(),
        baseline: view_of(&result.baseline),
        best: FrameView {
            candidate: view_of(&result.best),
            mask: mask_of(&params, result.best.geometry),
        },
        frames,
        candidates: result.candidates.iter().map(view_of).collect(),
    }
}

fn mask_of(params: &SweepParams, geometry: FinGeometry) -> Vec<u8> {
    mask_for(params, geometry)
        .map(|(mask, _)| mask.iter().map(|&s| u8::from(s)).collect())
        .unwrap_or_default()
}
/// Evaluates the baseline block plus every fitting (count, width) pair
/// and keeps the lowest thermal resistance. Deterministic: fixed order,
/// no randomness anywhere.
pub fn sweep(params: &SweepParams) -> SweepResult {
    let mut evals = 0;
    let mut candidates = Vec::new();
    // Baseline first: it is the initial best, so the result is valid
    // even if no finned design converges.
    let baseline = evaluate(
        params,
        FinGeometry {
            n_fins: 0,
            fin_width: 0,
            fin_height: 0,
        },
    )
    .expect("aluminum budget fits no baseline block");
    evals += 1;
    let mut best = Candidate {
        geometry: baseline.geometry,
        area_cells: baseline.area_cells,
        rth: baseline.rth,
        t_max: baseline.t_max,
        converged: true,
    };
    for n in 1..=params.max_fin_count {
        for &wf in &params.fin_widths {
            // Height preview without building: skip unfittable pairs.
            let base_area = params.base_rows * params.cols;
            if base_area >= params.aluminum_cells {
                continue;
            }
            let height = (params.aluminum_cells - base_area) / (n as usize * wf.max(1));
            let geometry = FinGeometry {
                n_fins: n,
                fin_width: wf,
                fin_height: height,
            };
            evals += 1;
            if let Some(c) = evaluate(params, geometry) {
                if c.rth < best.rth {
                    best = Candidate {
                        geometry: c.geometry,
                        area_cells: c.area_cells,
                        rth: c.rth,
                        t_max: c.t_max,
                        converged: true,
                    };
                }
                candidates.push(c);
            }
        }
    }
    SweepResult {
        baseline,
        best,
        candidates,
        evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{ThermalParams, CP_AL, K_AL, RHO_AL};

    // Small fast problem: 24×12 cells at 2.5 mm, fan cooling, 20 W.
    fn small() -> SweepParams {
        SweepParams {
            cols: 24,
            rows: 12,
            dx: 0.0025,
            thermal: ThermalParams {
                k: K_AL,
                rho: RHO_AL,
                cp: CP_AL,
                h: 100.0,
                channel_h: None,
                t_inf: 300.0,
                power: 20.0,
            },
            base_rows: 2,
            aluminum_cells: 120,
            max_fin_count: 6,
            fin_widths: vec![1, 2],
        }
    }

    #[test]
    fn winner_beats_the_solid_block() {
        let result = sweep(&small());
        assert!(
            result.best.rth < result.baseline.rth,
            "fins must help: best {} vs block {}",
            result.best.rth,
            result.baseline.rth
        );
    }

    #[test]
    fn sweep_is_deterministic() {
        let a = sweep(&small());
        let b = sweep(&small());
        assert_eq!(a.best.geometry, b.best.geometry);
        assert!((a.best.rth - b.best.rth).abs() < 1e-12);
        assert_eq!(a.evals, b.evals);
    }

    // The budget is a cap: candidates spend at most the fixed amount.
    #[test]
    fn every_candidate_honors_the_aluminum_budget() {
        let result = sweep(&small());
        assert!(!result.candidates.is_empty());
        for c in &result.candidates {
            assert!(c.area_cells <= 120);
            assert!(c.rth.is_finite());
        }
        assert!(result.best.area_cells <= 120);
    }

    // Flagship pin: the locked production problem answers 4 fins.
    // Deterministic (no RNG), so this is a regression gate, not luck.
    #[test]
    fn production_answers_four_fins() {
        let result = sweep(&production_params());
        assert_eq!(result.best.geometry.n_fins, 4);
        assert!(result.best.rth < 0.08, "Rth = {}", result.best.rth);
        assert!(result.best.rth < result.baseline.rth);
    }

    #[test]
    fn infeasible_geometries_fall_back_to_the_baseline() {
        // Absurd widths fit nothing: no finned candidates, but the
        // baseline still stands as a finite best instead of an error.
        let mut p = small();
        p.fin_widths = vec![23];
        p.max_fin_count = 8;
        let result = sweep(&p);
        assert!(result.candidates.is_empty());
        assert_eq!(result.best.geometry.n_fins, 0);
        assert!(result.best.rth.is_finite());
    }
}
