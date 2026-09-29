# Aluminati

## What it does

Designs a heat sink from a 120×60 mm footprint, 100 W of heat at the base, still air, and a fixed budget of 6063 aluminum, returning the geometry with the lowest thermal resistance plus its full temperature field.

It tries two approaches on the same budget. A sweep simulates 41 straight-fin designs and keeps the best. Free-form SIMP then grows branching trees from gray and lands at 0.097 K/W.

The solver is explicit conduction with Newton cooling at exposed faces, finished by a conjugate-gradient steady solve. Everything is deterministic.

![Final seconds of a free-form run: gray density silhouette landing into the true temperature field](assets/evolution.gif)

## Layout

- `src/physics.rs` — explicit FTCS conduction with base heat-flux source and Newton-cooling loss at exposed faces. 6063 aluminum.
- `src/sweep.rs` — parametric straight-fin family under a fixed-area constraint, ranked by thermal resistance.
- `src/lib.rs` — FFI surface: one sweep call plus a stateful forge.

## Checks

- `cargo test` (unit + analytic-oracle tests, deterministic per seed)
- `cargo build --release --target wasm32-unknown-unknown` (via the site build script, which pins the wasm-bindgen CLI from `Cargo.lock`)
