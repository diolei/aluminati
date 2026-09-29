//! Explicit FTCS conduction with base heat-flux source and Newton-cooling
//! loss at exposed faces. 6063 aluminum, SI units, 2D unit depth.
//!
//! Per solid cell, with D = α·dt/dx² and cell size dx:
//!
//! ```text
//! T += D·Σ(T[n] − T)  −  D·(h[k]·dx/k)·n_exp·(T − Tinf)  +  dt·Q/(ρ·cp)
//! ```
//!
//! over solid 4-neighbours, n_exp exposed faces, per-cell coefficient
//! h[k] (open or channel-confined), and volumetric source Q.
//! The loss term is the face flux h·(T − Tinf) through area dx·1 into
//! volume dx²·1. Stability needs D·(4 + 4·h·dx/k) ≤ 1; `stable_dt` holds
//! a 10% margin below that.

// 6063 aluminum: the alloy heat sinks are actually extruded from.
pub const K_AL: f64 = 200.0;
pub const RHO_AL: f64 = 2700.0;
pub const CP_AL: f64 = 900.0;

// Legacy alias kept for the test helper naming below.
pub const K_AL6063: f64 = K_AL;

#[derive(Clone, Copy)]
pub struct ThermalParams {
    pub k: f64,
    pub rho: f64,
    pub cp: f64,
    /// Convective coefficient at open exposed faces, W/(m²·K).
    pub h: f64,
    /// Optional confined-channel coefficient (Elenbaas): cells whose
    /// vertical faces look across a gap at another fin convect with
    /// this instead of `h`. None disables the distinction.
    pub channel_h: Option<f64>,
    /// Ambient temperature, K.
    pub t_inf: f64,
    /// Total source power, W at unit depth, spread uniformly over the
    /// solid cells of the bottom row.
    pub power: f64,
}

pub struct SteadyResult {
    /// Row-major temperatures, K.
    pub temps: Vec<f64>,
    pub iters: usize,
    pub converged: bool,
    pub t_max: f64,
}

/// Largest stable explicit step with a 10% safety margin. Uses the
/// open-space h (the worst case); channel cells only ever lose less.
pub fn stable_dt(dx: f64, params: &ThermalParams) -> f64 {
    let alpha = params.k / (params.rho * params.cp);
    0.9 * dx * dx / (4.0 * alpha * (1.0 + params.h * dx / params.k))
}

/// Effective per-cell coefficient: channel-classified cells use
/// `channel_h` when set, everything else the open-space value.
fn cell_h(channel: &[bool], k: usize, params: &ThermalParams) -> f64 {
    if channel[k] {
        params.channel_h.unwrap_or(params.h)
    } else {
        params.h
    }
}

/// Marks solid cells with a gap-facing exposed side: a vertical face
/// whose neighbour is void with solid beyond it (another fin across the
/// gap). Faces at the domain edge or against solid are open, so baseline
/// blocks and fin tips classify as open. Tips and outer edge faces of a
/// channel-classified cell inherit the channel value — a documented,
/// second-order misapplication.
pub fn classify_channel(solid: &[bool], cols: usize, rows: usize) -> Vec<bool> {
    let mut channel = vec![false; cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if !solid[k] {
                continue;
            }
            let left_gap = c > 0 && !solid[k - 1] && (0..c).any(|cc| solid[r * cols + cc]);
            let right_gap =
                c + 1 < cols && !solid[k + 1] && ((c + 1)..cols).any(|cc| solid[r * cols + cc]);
            channel[k] = left_gap || right_gap;
        }
    }
    channel
}

// Dry air at ~330 K film temperature.
const AIR_K: f64 = 0.0284;
const AIR_NU: f64 = 18.9e-6;
const AIR_ALPHA: f64 = 26.9e-6;
const AIR_BETA: f64 = 1.0 / 330.0;
const GRAVITY: f64 = 9.81;

/// Elenbaas (1942) average coefficient for an isothermal parallel-plate
/// channel: Nu_S = 0.60·[(S/L)·Ra_S]^1/4, h = Nu_S·k_air/S, with
/// Ra_S = g·β·ΔT·S³/(ν·α). Outside the correlation range the physics,
/// not the fit, decides: below (El < 10) the gap is stagnant and
/// conducts (Nu → 1); above (El > 1e3) the gap is unconfined open space
/// and convects at the open rate.
pub fn elenbaas_h(gap_s: f64, len_l: f64, delta_t: f64, h_open: f64) -> f64 {
    if gap_s <= 0.0 || len_l <= 0.0 || delta_t <= 0.0 {
        return h_open.min(1.0);
    }
    let ra = GRAVITY * AIR_BETA * delta_t * gap_s.powi(3) / (AIR_NU * AIR_ALPHA);
    let el = (gap_s / len_l) * ra;
    if el < 10.0 {
        return AIR_K / gap_s;
    }
    if el > 1000.0 {
        return h_open;
    }
    0.60 * el.powf(0.25) * AIR_K / gap_s
}

/// Counts faces of cell (r, c) exposed to void or the domain edge.
pub fn exposed_faces(solid: &[bool], cols: usize, rows: usize, r: usize, c: usize) -> u8 {
    let mut n = 0;
    if r == 0 || !solid[(r - 1) * cols + c] {
        n += 1;
    }
    if r + 1 == rows || !solid[(r + 1) * cols + c] {
        n += 1;
    }
    if c == 0 || !solid[r * cols + c - 1] {
        n += 1;
    }
    if c + 1 == cols || !solid[r * cols + c + 1] {
        n += 1;
    }
    n
}

/// One explicit step over all solid cells. Fixed (Dirichlet) cells are
/// never written. `channel` is the precomputed classification (uniform
/// false disables the distinction). Starts from ambient; callers loop.
// Numeric kernel: explicit field-per-argument signature kept stable on
// purpose (callers and stencil tests pin the order); bundling would
// churn the validated call graph for style only.
#[allow(clippy::too_many_arguments)]
pub fn step(
    temps: &mut [f64],
    solid: &[bool],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    channel: &[bool],
    dt: f64,
) {
    let alpha = params.k / (params.rho * params.cp);
    let d = alpha * dt / (dx * dx);
    let heat = dt / (params.rho * params.cp);
    // Source power spreads over the solid cells of the bottom row.
    let mut base_cells = 0;
    for c in 0..cols {
        if solid[(rows - 1) * cols + c] {
            base_cells += 1;
        }
    }
    let q = if base_cells > 0 {
        params.power / (base_cells as f64 * dx * dx)
    } else {
        0.0
    };

    let mut next = temps.to_vec();
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if !solid[k] || fixed[k].is_some() {
                continue;
            }
            let t = temps[k];
            let mut lap = 0.0;
            if r > 0 && solid[k - cols] {
                lap += temps[k - cols] - t;
            }
            if r + 1 < rows && solid[k + cols] {
                lap += temps[k + cols] - t;
            }
            if c > 0 && solid[k - 1] {
                lap += temps[k - 1] - t;
            }
            if c + 1 < cols && solid[k + 1] {
                lap += temps[k + 1] - t;
            }
            let n_exp = exposed_faces(solid, cols, rows, r, c) as f64;
            let loss = d * cell_h(channel, k, params) * dx / params.k;
            let src = if r + 1 == rows { q } else { 0.0 };
            next[k] = t + d * lap - loss * n_exp * (t - params.t_inf) + heat * src;
        }
    }
    temps.copy_from_slice(&next);
}

/// Marches from ambient to steady state and reports it. The stopping rule
/// is physical, not numerical:
/// - source-driven (power > 0): relative energy imbalance
///   |P − loss| / P below 1e-3, checked every 50 steps;
/// - pure Dirichlet relaxation (power = 0): largest cell change below
///   1e-6 K over a 50-step window.
///
/// A fixed per-step tolerance cannot work for both: slow convective
/// systems change microkelvins per step while still far from balance.
pub fn steady_state(
    solid: &[bool],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
    max_steps: usize,
) -> SteadyResult {
    let mut temps = vec![params.t_inf; cols * rows];
    for (k, f) in fixed.iter().enumerate() {
        if let Some(t) = f {
            temps[k] = *t;
        }
    }
    let dt = stable_dt(dx, params);
    let driven = params.power > 0.0;
    let channel = classify_channel(solid, cols, rows);
    let mut iters = 0;
    let mut converged = false;
    let mut snapshot = temps.clone();
    for _ in 0..max_steps {
        step(
            &mut temps, solid, cols, rows, dx, params, fixed, &channel, dt,
        );
        iters += 1;
        if iters % 50 == 0 {
            if driven {
                let imbalance = ((convective_loss(&temps, solid, cols, rows, dx, params)
                    - params.power)
                    / params.power)
                    .abs();
                if imbalance < 1e-3 {
                    converged = true;
                    break;
                }
            } else {
                let change = temps
                    .iter()
                    .zip(snapshot.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                if change < 1e-6 {
                    converged = true;
                    break;
                }
            }
            snapshot = temps.clone();
        }
    }
    let t_max = temps
        .iter()
        .zip(solid.iter())
        .filter(|(_, s)| **s)
        .map(|(t, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    SteadyResult {
        temps,
        iters,
        converged,
        t_max,
    }
}

/// Total convected power of a field: Σ h·dx·n_exp·(T − Tinf) over solid
/// cells. The engine's own balance meter; tests recompute it independently.
fn convective_loss(
    temps: &[f64],
    solid: &[bool],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
) -> f64 {
    let mut loss = 0.0;
    let channel = classify_channel(solid, cols, rows);
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if !solid[k] {
                continue;
            }
            loss += cell_h(&channel, k, params)
                * dx
                * exposed_faces(solid, cols, rows, r, c) as f64
                * (temps[k] - params.t_inf);
        }
    }
    loss
}

/// Steady state by Conjugate Gradient on the discrete system.
/// Rearranging the stencil at equilibrium (D cancels throughout):
///
/// ```text
/// (n_solid + β·n_exp)·T[c] − ΣT[n] = β·n_exp·Tinf + P/(k·n_base)
/// ```
///
/// with β = h·dx/k and the source uniform over the n_base bottom-row
/// solid cells. The matrix is symmetric positive definite (graph
/// Laplacian plus a positive diagonal), so CG with a Jacobi
/// preconditioner converges in O(√κ) iterations regardless of the
/// physical time constant — unlike transient marching, whose steps
/// scale with τ. Both solvers discretize the same physics; the analytic
/// fin oracle below pins them both.
pub fn steady_cg(
    solid: &[bool],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
) -> SteadyResult {
    let channel = classify_channel(solid, cols, rows);
    // Per-cell β = h[k]·dx/k.
    let beta_of = |k: usize| cell_h(&channel, k, params) * dx / params.k;
    // Unknown index per solid non-Dirichlet cell.
    let mut idx = vec![usize::MAX; cols * rows];
    let mut count = 0;
    for k in 0..cols * rows {
        if solid[k] && fixed[k].is_none() {
            idx[k] = count;
            count += 1;
        }
    }
    // Bottom-row solid cells share the source power.
    let mut base_cells = 0;
    for c in 0..cols {
        if solid[(rows - 1) * cols + c] {
            base_cells += 1;
        }
    }
    let src = if base_cells > 0 {
        params.power / (params.k * base_cells as f64)
    } else {
        0.0
    };

    // Assemble SPD system restricted to unknowns.
    let dirs = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    let mut diag = vec![0.0; count];
    let mut rhs = vec![0.0; count];
    let mut off: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if !solid[k] || fixed[k].is_some() {
                continue;
            }
            let i = idx[k];
            let n_exp = exposed_faces(solid, cols, rows, r, c) as f64;
            let beta = beta_of(k);
            diag[i] = 4.0 - missing_dirs(solid, cols, rows, r, c) as f64 + beta * n_exp;
            // Note: diag counts only solid neighbours; void faces convect
            // through n_exp, so missing graph edges never weaken the row.
            let mut b = beta * n_exp * params.t_inf;
            if r + 1 == rows {
                b += src;
            }
            for (dr, dc) in dirs {
                let rr = r as isize + dr;
                let cc = c as isize + dc;
                if rr < 0 || cc < 0 || rr >= rows as isize || cc >= cols as isize {
                    continue;
                }
                let nk = rr as usize * cols + cc as usize;
                if !solid[nk] {
                    continue;
                }
                if let Some(t) = fixed[nk] {
                    b += t;
                } else {
                    off[i].push((idx[nk], -1.0));
                }
            }
            rhs[i] = b;
        }
    }

    // Jacobi-preconditioned CG to 1e-10 relative residual.
    let (x, iters, converged) = pcg(&diag, &off, &rhs, params.t_inf);

    let mut temps = vec![params.t_inf; cols * rows];
    for (k, f) in fixed.iter().enumerate() {
        if let Some(t) = f {
            temps[k] = *t;
        }
    }
    for k in 0..cols * rows {
        if idx[k] != usize::MAX {
            temps[k] = x[idx[k]];
        }
    }
    let t_max = temps
        .iter()
        .zip(solid.iter())
        .filter(|(_, s)| **s)
        .map(|(t, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    SteadyResult {
        temps,
        iters,
        converged,
        t_max,
    }
}

/// Jacobi-preconditioned Conjugate Gradients to 1e-10 relative
/// residual over a sparse SPD system in (diag, off-diagonal) form.
/// Shared by the uniform and heterogeneous assemblies, and by the SIMP
/// adjoint re-solve.
pub(crate) fn pcg(
    diag: &[f64],
    off: &[Vec<(usize, f64)>],
    rhs: &[f64],
    x0: f64,
) -> (Vec<f64>, usize, bool) {
    let count = rhs.len();
    let mut x = vec![x0; count];
    let mut r: Vec<f64> = rhs
        .iter()
        .enumerate()
        .map(|(i, b)| b - (diag[i] * x[i] + off[i].iter().map(|(j, v)| v * x[*j]).sum::<f64>()))
        .collect();
    let mut z: Vec<f64> = r.iter().enumerate().map(|(i, v)| v / diag[i]).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
    let r0 = rz.sqrt().max(1e-300);
    let mut iters = 0;
    let mut converged = false;
    for _ in 0..10_000 {
        // Ap = A·p
        let mut ap = vec![0.0; count];
        for i in 0..count {
            ap[i] = diag[i] * p[i] + off[i].iter().map(|(j, v)| v * p[*j]).sum::<f64>();
        }
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            break; // No curvature left: mechanism-like nullspace, stop.
        }
        let alpha_step = rz / pap;
        for i in 0..count {
            x[i] += alpha_step * p[i];
            r[i] -= alpha_step * ap[i];
        }
        iters += 1;
        for i in 0..count {
            z[i] = r[i] / diag[i];
        }
        let rz_new: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        if rz_new.sqrt() < 1e-10 * r0 {
            converged = true;
            break;
        }
        let beta_cg = rz_new / rz;
        for i in 0..count {
            p[i] = z[i] + beta_cg * p[i];
        }
        rz = rz_new;
    }
    (x, iters, converged)
}

/// Steady state with per-cell conductivity (SIMP densities): every cell
/// conducts with harmonic face averages g̃ = 2·k̃_a·k̃_b/(k̃_a+k̃_b)
/// relative to k_al, while convection and source placement follow the
/// thresholded `solid_conv` mask (frozen-boundary approximation). With
/// uniform conductivity over an all-solid mask this is the same system
/// `steady_cg` solves.
pub fn steady_cg_hetero(
    solid_conv: &[bool],
    cond: &[f64],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
) -> SteadyResult {
    let (diag, off, rhs, idx) = assemble_hetero(solid_conv, cond, cols, rows, dx, params, fixed);

    let (x, iters, converged) = pcg(&diag, &off, &rhs, params.t_inf);

    let mut temps = vec![params.t_inf; cols * rows];
    for (k, f) in fixed.iter().enumerate() {
        if let Some(t) = f {
            temps[k] = *t;
        }
    }
    for k in 0..cols * rows {
        if idx[k] != usize::MAX {
            temps[k] = x[idx[k]];
        }
    }
    let t_max = temps
        .iter()
        .zip(solid_conv.iter())
        .filter(|(_, s)| **s)
        .map(|(t, _)| *t)
        .fold(f64::NEG_INFINITY, f64::max);
    SteadyResult {
        temps,
        iters,
        converged,
        t_max,
    }
}
/// Assembled heterogeneous system: diagonal, off-diagonal pairs,
/// right-hand side, and the cell→unknown index map.
type HeteroSystem = (Vec<f64>, Vec<Vec<(usize, f64)>>, Vec<f64>, Vec<usize>);

/// Assembles the heterogeneous conduction system (see
/// `steady_cg_hetero`): diagonal, off-diagonal pairs, right-hand side,
/// and the cell→unknown index map. Exposed for the SIMP adjoint, which
/// re-solves the same matrix with a second right-hand side.
pub fn assemble_hetero(
    solid_conv: &[bool],
    cond: &[f64],
    cols: usize,
    rows: usize,
    dx: f64,
    params: &ThermalParams,
    fixed: &[Option<f64>],
) -> HeteroSystem {
    let channel = classify_channel(solid_conv, cols, rows);
    let beta_of = |k: usize| cell_h(&channel, k, params) * dx / params.k;
    let krel = |k: usize| cond[k] / params.k;
    let mut idx = vec![usize::MAX; cols * rows];
    let mut count = 0;
    for k in 0..cols * rows {
        if fixed[k].is_none() {
            idx[k] = count;
            count += 1;
        }
    }
    let mut base_cells = 0;
    for c in 0..cols {
        if solid_conv[(rows - 1) * cols + c] {
            base_cells += 1;
        }
    }
    let src = if base_cells > 0 {
        params.power / (params.k * base_cells as f64)
    } else {
        0.0
    };

    let dirs = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    let mut diag = vec![0.0; count];
    let mut rhs = vec![0.0; count];
    let mut off: Vec<Vec<(usize, f64)>> = vec![Vec::new(); count];
    for r in 0..rows {
        for c in 0..cols {
            let k = r * cols + c;
            if fixed[k].is_some() {
                continue;
            }
            let i = idx[k];
            // Void cells conduct weakly (ersatz) but never convect: only
            // convecting-mask cells lose heat to ambient.
            let n_exp = if solid_conv[k] {
                exposed_faces(solid_conv, cols, rows, r, c) as f64
            } else {
                0.0
            };
            let mut diag_i = beta_of(k) * n_exp;
            let mut b = beta_of(k) * n_exp * params.t_inf;
            if solid_conv[k] && r + 1 == rows {
                b += src;
            }
            for (dr, dc) in dirs {
                let rr = r as isize + dr;
                let cc = c as isize + dc;
                if rr < 0 || cc < 0 || rr >= rows as isize || cc >= cols as isize {
                    continue;
                }
                let nk = rr as usize * cols + cc as usize;
                let ka = krel(k);
                let kb = krel(nk);
                let g = 2.0 * ka * kb / (ka + kb);
                diag_i += g;
                if let Some(t) = fixed[nk] {
                    b += g * t;
                } else {
                    off[i].push((idx[nk], -g));
                }
            }
            diag[i] = diag_i;
            rhs[i] = b;
        }
    }
    (diag, off, rhs, idx)
}

/// Solid-neighbour deficit: faces pointing at void or the domain edge.
/// (Those faces convect via n_exp instead of conducting.)
fn missing_dirs(solid: &[bool], cols: usize, rows: usize, r: usize, c: usize) -> u8 {
    let mut n = 0;
    if r == 0 || !solid[(r - 1) * cols + c] {
        n += 1;
    }
    if r + 1 == rows || !solid[(r + 1) * cols + c] {
        n += 1;
    }
    if c == 0 || !solid[r * cols + c - 1] {
        n += 1;
    }
    if c + 1 == cols || !solid[r * cols + c + 1] {
        n += 1;
    }
    n
}

/// Thermal resistance of a settled field: (Tmax − Tinf) / P.
pub fn thermal_resistance(result: &SteadyResult, params: &ThermalParams) -> f64 {
    (result.t_max - params.t_inf) / params.power
}

#[cfg(test)]
mod tests {
    use super::*;

    fn al_params(h: f64, power: f64) -> ThermalParams {
        ThermalParams {
            k: K_AL6063,
            rho: RHO_AL,
            cp: CP_AL,
            h,
            channel_h: None,
            t_inf: 300.0,
            power,
        }
    }

    // Hand-computed Elenbaas point: S = 5 mm, L = 40 mm, ΔT = 30 K.
    // Ra = 9.81·(1/330)·30·(5e-3)³/(18.9e-6·26.9e-6) ≈ 219,
    // El = (5/40)·219 ≈ 27.4, Nu = 0.6·27.4^0.25 ≈ 1.37,
    // h = 1.37·0.0284/5e-3 ≈ 7.8 W/(m²·K).
    #[test]
    fn elenbaas_matches_hand_computation() {
        let h = elenbaas_h(0.005, 0.04, 30.0, 25.0);
        assert!((h - 7.8) / 7.8 < 0.02, "Elenbaas h = {h}");
    }

    #[test]
    fn elenbaas_degenerates_safely() {
        assert_eq!(elenbaas_h(0.0, 0.04, 30.0, 25.0), 1.0);
        // Stagnant tight gap conducts: k_air/S exactly.
        let tight = elenbaas_h(0.001, 0.04, 30.0, 25.0);
        assert!((tight - 28.4).abs() < 0.5, "tight channel h = {tight}");
        // Unconfined wide gap recovers the open-space value.
        let wide = elenbaas_h(0.05, 0.04, 30.0, 25.0);
        assert!((wide - 25.0).abs() < 1e-9, "wide channel h = {wide}");
    }

    #[test]
    fn channel_marks_fin_walls_not_blocks() {
        // Three 1-wide fins over a base: side walls face gaps, the block
        // below faces only solid and edges.
        let cols = 9;
        let rows = 6;
        let mut solid = vec![false; cols * rows];
        for r in 4..6 {
            for c in 0..cols {
                solid[r * cols + c] = true;
            }
        }
        for f in [1, 4, 7] {
            for r in 1..4 {
                solid[r * cols + f] = true;
            }
        }
        let channel = classify_channel(&solid, cols, rows);
        for f in [1, 4, 7] {
            assert!(channel[2 * cols + f], "fin wall ({f}) not channel");
        }
        for c in 0..cols {
            assert!(!channel[4 * cols + c], "base cell ({c}) marked channel");
            assert!(!channel[5 * cols + c], "base cell ({c}) marked channel");
        }
    }
    // Closed-form 1D fin with convective tip — derived without any stencil
    // code, so it cannot pass by construction. Fin: single cell column,
    // base row held at Tb (Dirichlet), both flanks and the tip convect.
    // (T-Tinf)/(Tb-Tinf) = [cosh(m(L-x)) + (h/mk)sinh(m(L-x))]
    //                      / [cosh(mL) + (h/mk)sinh(mL)], m² = hP/kA.
    #[test]
    fn fin_matches_analytic_profile() {
        let cols = 1;
        let rows = 100;
        let dx = 0.001;
        let tb = 350.0;
        let t_inf = 300.0;
        let h = 100.0;
        let k = K_AL6063;
        let params = ThermalParams {
            k,
            rho: RHO_AL,
            cp: CP_AL,
            h,
            channel_h: None,
            t_inf,
            power: 0.0,
        };
        let solid = vec![true; cols * rows];
        let mut fixed = vec![None; cols * rows];
        fixed[0] = Some(tb);

        let result = steady_state(&solid, cols, rows, dx, &params, &fixed, 200_000);
        assert!(result.converged, "fin did not reach steady state");

        // 2D unit depth: A = dx·1, P = 2·1 (both flanks convect).
        let m = ((h * 2.0) / (k * dx)).sqrt();
        let l = rows as f64 * dx;
        let hmk = h / (m * k);
        let denom = (m * l).cosh() + hmk * (m * l).sinh();
        let mut worst = 0.0f64;
        for r in 1..rows {
            let x = r as f64 * dx;
            let numer = (m * (l - x)).cosh() + hmk * (m * (l - x)).sinh();
            let expected = t_inf + (tb - t_inf) * numer / denom;
            let actual = result.temps[r * cols];
            let rel = ((actual - expected) / (tb - t_inf)).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 0.03, "worst relative error {worst}");
    }

    // The CG solver discretizes the same physics by a different route
    // (direct equilibrium vs transient marching), so it must satisfy the
    // same closed-form oracle.
    #[test]
    fn cg_matches_analytic_fin_profile() {
        let cols = 1;
        let rows = 100;
        let dx = 0.001;
        let tb = 350.0;
        let t_inf = 300.0;
        let h = 100.0;
        let k = K_AL6063;
        let params = ThermalParams {
            k,
            rho: RHO_AL,
            cp: CP_AL,
            h,
            channel_h: None,
            t_inf,
            power: 0.0,
        };
        let solid = vec![true; cols * rows];
        let mut fixed = vec![None; cols * rows];
        fixed[0] = Some(tb);

        let result = steady_cg(&solid, cols, rows, dx, &params, &fixed);
        assert!(result.converged);

        let m = ((h * 2.0) / (k * dx)).sqrt();
        let l = rows as f64 * dx;
        let hmk = h / (m * k);
        let denom = (m * l).cosh() + hmk * (m * l).sinh();
        let mut worst = 0.0f64;
        for r in 1..rows {
            let x = r as f64 * dx;
            let numer = (m * (l - x)).cosh() + hmk * (m * (l - x)).sinh();
            let expected = t_inf + (tb - t_inf) * numer / denom;
            let rel = ((result.temps[r * cols] - expected) / (tb - t_inf)).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 0.03, "CG worst relative error {worst}");
    }

    // Two independent routes to the same equilibrium must agree: transient
    // marching and direct CG on the driven block.
    #[test]
    fn cg_agrees_with_transient_marching() {
        let cols = 10;
        let rows = 5;
        let dx = 0.001;
        let params = al_params(100.0, 50.0);
        let solid = vec![true; cols * rows];
        let fixed = vec![None; cols * rows];

        let marched = steady_state(&solid, cols, rows, dx, &params, &fixed, 200_000);
        assert!(marched.converged);
        let direct = steady_cg(&solid, cols, rows, dx, &params, &fixed);
        assert!(direct.converged);
        let rel = ((direct.t_max - marched.t_max) / (marched.t_max - params.t_inf)).abs();
        assert!(rel < 0.01, "methods disagree by {rel}");
    }
    // Independent oracle: at steady state with no Dirichlet cells, every
    // watt from the source must leave through convection. The loss is
    // recomputed here face by face, not via engine internals.
    // Small block under a fan (h = 100): τ = ρcV/hA ≈ 40 s, so ~9τ of
    // 2.7 ms steps fits comfortably in max_steps.
    #[test]
    fn energy_balances_at_steady_state() {
        let cols = 10;
        let rows = 5;
        let dx = 0.001;
        let power = 50.0;
        let h = 100.0;
        let t_inf = 300.0;
        let params = al_params(h, power);
        let solid = vec![true; cols * rows];
        let fixed = vec![None; cols * rows];

        let result = steady_state(&solid, cols, rows, dx, &params, &fixed, 200_000);
        assert!(result.converged, "block did not reach steady state");

        let mut loss = 0.0;
        for r in 0..rows {
            for c in 0..cols {
                let exposed = exposed_faces(&solid, cols, rows, r, c);
                if exposed > 0 {
                    loss += h * dx * exposed as f64 * (result.temps[r * cols + c] - t_inf);
                }
            }
        }
        assert!(
            ((loss - power) / power).abs() < 0.01,
            "convected {loss} W vs sourced {power} W"
        );
    }

    // Von Neumann discipline, tested through a single explicit step: with
    // no source and insulated faces (h = 0), the checkerboard mode must
    // shrink by |1 − 8D| per step, never grow.
    #[test]
    fn checkerboard_decays_in_one_step() {
        let cols = 8;
        let rows = 8;
        let dx = 0.001;
        let params = al_params(0.0, 0.0);
        let solid = vec![true; cols * rows];
        let fixed = vec![None; cols * rows];

        let mut temps = vec![0.0; cols * rows];
        for r in 0..rows {
            for c in 0..cols {
                temps[r * cols + c] = params.t_inf + if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let alpha = params.k / (params.rho * params.cp);
        let open = vec![false; cols * rows];
        step(
            &mut temps,
            &solid,
            cols,
            rows,
            dx,
            &params,
            &fixed,
            &open,
            0.2 * dx * dx / alpha,
        );

        let amplitude = temps
            .iter()
            .map(|t| (t - params.t_inf).abs())
            .fold(0.0, f64::max);
        assert!(amplitude < 1.0, "checkerboard grew: amplitude {amplitude}");
    }
}
