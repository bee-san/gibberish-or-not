//! The built-in language packs, parsed once.
//!
//! One line per language, under its marker: a language's pull request adds only its line
//! here (and files under `languages/<code>/`), so the language PRs don't conflict
//! (`docs/languages.md`). The Cargo feature of each language is declared in Cargo.toml
//! already; it is a no-op until the language's line is here.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use once_cell::race::OnceBox;

use crate::pack::Pack;

#[allow(unused_macros)]
macro_rules! builtin {
    ($v:ident, $code:literal) => {
        $v.push(Arc::new(
            Pack::from_static(
                include_bytes!(concat!("../languages/", $code, "/model.bin")),
                include_str!(concat!("../languages/", $code, "/calibration.txt")),
            )
            .expect(concat!("the built-in ", $code, " pack is valid")),
        ));
    };
}

#[allow(unused_mut)]
fn load() -> Vec<Arc<Pack>> {
    let mut v = Vec::new();
    // --- en ---
    #[cfg(feature = "en")]
    builtin!(v, "en");
    // --- es ---
    // --- fr ---
    // --- de ---
    // --- it ---
    // --- pt ---
    // --- nl ---
    // --- pl ---
    // --- tr ---
    // --- ru ---
    // --- zh ---
    // --- ja ---
    // --- ko ---
    v
}

/// The packs of every language compiled in.
pub(crate) fn builtin() -> &'static [Arc<Pack>] {
    static PACKS: OnceBox<Vec<Arc<Pack>>> = OnceBox::new();
    PACKS.get_or_init(|| Box::new(load()))
}
