//! JavaScript bindings (feature `wasm`), for `wasm32-unknown-unknown` with wasm-bindgen.
//!
//! ```js
//! import init, { detect, score, isGibberish } from "./gibberish_or_not.js";
//! await init();
//! detect("Bonjour tout le monde");   // Detection { isLanguage, language: "fr", score }
//! score("Bonjour tout le monde");    // the score alone, no object allocated
//! isGibberish("Wkh txlfn eurzq ira", "medium");   // true
//! ```

use alloc::string::String;
use wasm_bindgen::prelude::*;

use crate::{default_detector, Detector, Sensitivity};

fn sensitivity(name: Option<String>) -> Result<Sensitivity, JsError> {
    match name.as_deref() {
        None | Some("medium") | Some("Medium") => Ok(Sensitivity::Medium),
        Some("low") | Some("Low") => Ok(Sensitivity::Low),
        Some("high") | Some("High") => Ok(Sensitivity::High),
        Some(other) => Err(JsError::new(&alloc::format!(
            "unknown sensitivity {other:?}: use \"low\", \"medium\" or \"high\""
        ))),
    }
}

/// The result of `detect`, for JavaScript.
#[wasm_bindgen(js_name = Detection)]
pub struct JsDetection {
    is_language: bool,
    language: Option<String>,
    score: f32,
}

#[wasm_bindgen(js_class = Detection)]
impl JsDetection {
    /// Whether the text is natural language.
    #[wasm_bindgen(getter, js_name = isLanguage)]
    pub fn is_language(&self) -> bool {
        self.is_language
    }
    /// The ISO 639-1 code of the language, when it is language.
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> Option<String> {
        self.language.clone()
    }
    /// The calibrated score in [0, 1].
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }
}

/// Detects `text` with every built-in language at `sensitivity` ("low", "medium" or
/// "high"; default "medium"; anything else throws).
#[wasm_bindgen]
pub fn detect(text: &str, sensitivity_name: Option<String>) -> Result<JsDetection, JsError> {
    let s = sensitivity(sensitivity_name)?;
    let (score, lang) = default_detector().best(text);
    let is_language = score >= s.min_score();
    Ok(JsDetection {
        is_language,
        language: lang.filter(|_| is_language).map(|l| String::from(l.code())),
        score,
    })
}

/// The calibrated score of `text` in [0, 1] (0.5, 0.75, 0.9: High, Medium, Low accept),
/// without allocating a result object: for hot loops.
#[wasm_bindgen]
pub fn score(text: &str) -> f32 {
    default_detector().best(text).0
}

/// Whether `text` is not natural language at `sensitivity` ("low", "medium" or "high";
/// default "medium"; anything else throws).
#[wasm_bindgen(js_name = isGibberish)]
pub fn is_gibberish(text: &str, sensitivity_name: Option<String>) -> Result<bool, JsError> {
    Ok(crate::is_gibberish(text, sensitivity(sensitivity_name)?))
}

/// Whether `text` is a common password.
#[wasm_bindgen(js_name = isPassword)]
pub fn is_password(text: &str) -> bool {
    crate::is_password(text)
}

/// The ISO 639-1 codes of the built-in languages.
#[wasm_bindgen]
pub fn languages() -> alloc::vec::Vec<String> {
    let d: &Detector = default_detector();
    d.enabled().map(|l| String::from(l.code())).collect()
}
