//! The benchmark corpus: `bench/corpus/<version>/{train,dev,test}.tsv`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Label {
    Pos,
    Neg,
    Info,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub id: String,
    /// Language of a positive, or the language a negative was derived from (`-`: none).
    pub lang: String,
    pub label: Label,
    pub category: String,
    pub source: String,
    pub text: String,
}

impl Sample {
    /// Characters, for the length buckets.
    pub fn chars(&self) -> usize {
        self.text.chars().count()
    }
}

/// The repository root (the parent of `bench/`).
pub fn root() -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    here.parent().unwrap_or(here).to_path_buf()
}

pub fn corpus_dir(version: &str) -> PathBuf {
    root().join("bench").join("corpus").join(version)
}

/// Undoes the corpus escapes: `\t \n \r \\`, `\xHH`, `\uHHHH`.
pub fn unescape(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let mut chars = field.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some(k @ ('x' | 'u')) => {
                let n = if k == 'x' { 2 } else { 4 };
                let hex: String = chars.by_ref().take(n).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(c) if hex.len() == n => out.push(c),
                    _ => {
                        out.push('\\');
                        out.push(k);
                        out.push_str(&hex);
                    }
                }
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

pub fn load(version: &str, split: &str) -> Result<Vec<Sample>, String> {
    let path = corpus_dir(version).join(format!("{split}.tsv"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(6, '\t').collect();
        if f.len() != 6 {
            return Err(format!("{}:{}: expected 6 fields", path.display(), n + 1));
        }
        let label = match f[2] {
            "pos" => Label::Pos,
            "neg" => Label::Neg,
            "info" => Label::Info,
            other => return Err(format!("{}:{}: label {other}", path.display(), n + 1)),
        };
        out.push(Sample {
            id: f[0].to_string(),
            lang: f[1].to_string(),
            label,
            category: f[3].to_string(),
            source: f[4].to_string(),
            text: unescape(f[5]),
        });
    }
    Ok(out)
}

/// Frozen verdicts of a baseline: id -> accepted at (Low, Medium, High), and ns per check.
pub struct Baseline {
    pub name: &'static str,
    pub verdicts: HashMap<String, [bool; 3]>,
    pub ns: HashMap<String, f64>,
}

/// `bench/baselines/<version>/<split>.tsv` (written by the scratch Ciphey example
/// `gon_baselines`, see bench/baselines/README.md): gibberish-or-not 5.0.7 and Ciphey's
/// English checker.
pub fn load_baselines(version: &str, split: &str) -> Vec<Baseline> {
    let path = root()
        .join("bench")
        .join("baselines")
        .join(version)
        .join(format!("{split}.tsv"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut gon = Baseline {
        name: "5.0.7",
        verdicts: HashMap::new(),
        ns: HashMap::new(),
    };
    let mut eng = Baseline {
        name: "Ciphey-English",
        verdicts: HashMap::new(),
        ns: HashMap::new(),
    };
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 {
            continue;
        }
        let b = |i: usize| f[i] == "1";
        gon.verdicts.insert(f[0].to_string(), [b(1), b(2), b(3)]);
        eng.verdicts.insert(f[0].to_string(), [b(4), b(5), b(6)]);
        gon.ns
            .insert(f[0].to_string(), f[7].parse().unwrap_or(f64::NAN));
        eng.ns
            .insert(f[0].to_string(), f[8].parse().unwrap_or(f64::NAN));
    }
    vec![gon, eng]
}
