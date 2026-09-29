//! Aluminati: heat-sink geometry engine.
//!
//! Fixed aluminum, lowest thermal resistance wins. Physics core
//! (`physics`), parametric sweep (`sweep`), FFI surface only here.

pub mod physics;
pub mod simp;
pub mod sweep;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// FFI surface: one call runs the whole production sweep and returns the
/// playback history. Narrow by design — no physics crosses the boundary
/// piecemeal.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn sweep_history() -> JsValue {
    serde_wasm_bindgen::to_value(&sweep::history_view()).unwrap_or(JsValue::NULL)
}

/// Progressive SIMP forging: the widget advances a few iterations per
/// animation frame and renders live, instead of blocking on one long
/// call. Same frames as a one-shot run, chunked. The production problem
/// is baked in (constructor takes no arguments) to keep the boundary
/// narrow; replays are bit-identical.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SimpForge {
    run: simp::SimpRun,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SimpForge {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SimpForge {
        SimpForge {
            run: simp::SimpRun::new(simp::production_simp_params()),
        }
    }

    /// Total iterations in the production run (for progress readouts).
    pub fn total(&self) -> usize {
        self.run.params().max_iter
    }

    /// Grid dimensions for sizing the renderer. Separate scalar
    /// methods — not a Vec — because raw Vec returns do not survive
    /// the wasm-bindgen boundary as arrays.
    pub fn cols(&self) -> u32 {
        self.run.params().cols as u32
    }

    pub fn rows(&self) -> u32 {
        self.run.params().rows as u32
    }

    /// Advances up to n iterations, returning newly recorded frames.
    pub fn step(&mut self, n: usize) -> JsValue {
        serde_wasm_bindgen::to_value(&self.run.step(n)).unwrap_or(JsValue::NULL)
    }

    /// Thermal resistance of the finished design, or null while forging.
    pub fn final_rth(&self) -> JsValue {
        match self.run.thresholded_rth() {
            Some((rth, tmax)) => {
                serde_wasm_bindgen::to_value(&(rth, tmax)).unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
        }
    }

    pub fn done(&self) -> bool {
        self.run.done()
    }
}
