use aluminati::physics::*;
use aluminati::simp::*;
fn main() {
    let p = production_simp_params();
    let mut run = SimpRun::new(p.clone());
    while !run.done() {
        run.step(20);
    }
    let fixed = vec![None; p.cols * p.rows];
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let mask: Vec<bool> = run.densities().iter().map(|&v| v >= thr).collect();
        let n = mask.iter().filter(|&&b| b).count();
        let s = steady_cg(&mask, p.cols, p.rows, p.dx, &p.thermal, &fixed);
        let rth = if s.converged {
            thermal_resistance(&s, &p.thermal)
        } else {
            f64::NAN
        };
        println!("thr={thr}: cells={n} Rth={rth:.4}");
    }
}
