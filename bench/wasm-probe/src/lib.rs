//! The WASM module the size budget and the node latency are measured on: the library with
//! its `wasm` feature (every default language), nothing else. Not published.
//!
//! ```sh
//! cargo build --release --target wasm32-unknown-unknown -p gon-wasm-probe
//! ```

// Link the library: its wasm-bindgen exports are the module's API.
pub use gibberish_or_not::wasm::{detect, is_gibberish, is_password, languages, score};
