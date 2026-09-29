use aluminati::simp::*;
use std::time::Instant;
fn main() {
    for iters in [200usize, 300, 400] {
        let mut p = production_simp_params();
        p.max_iter = iters;
        p.snapshot_every = 20;
        let t = Instant::now();
        let mut run = SimpRun::new(p.clone());
        while !run.done() {
            run.step(20);
        }
        let (rth, tmax) = run.thresholded_rth().unwrap();
        println!(
            "iters={iters}: {:.1}s Rth={rth:.4} Tmax={tmax:.1}K",
            t.elapsed().as_secs_f32()
        );
    }
}
