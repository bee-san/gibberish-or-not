//! gib-bench: gibberish-or-not's benchmark, in one command, and the tools that fit packs.
//!
//! ```text
//! cargo run --release -p gib-bench -- --corpus v1                # report on the dev split
//! cargo run --release -p gib-bench -- --corpus v1 --split test   # test: once per released model
//! cargo run --release -p gib-bench -- --corpus v1 --langs en     # only some languages
//! cargo run --release -p gib-bench -- --corpus v1 --wasm         # + WASM size and latency
//! cargo run --release -p gib-bench -- fit --lang en              # writes languages/en/calibration.txt
//! cargo run --release -p gib-bench -- gate                       # CI accuracy gate (dev split)
//! cargo run --release -p gib-bench -- latency                    # latency only (also under wasmtime)
//! cargo run --release -p gib-bench -- check < lines              # score each line of stdin
//! ```
//!
//! See docs/benchmark.md for what every number means.

// Linear algebra and table code reads best with index loops
#![allow(clippy::needless_range_loop)]

mod corpus;
mod fit;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use corpus::{Baseline, Label, Sample};
use gibberish_or_not::{Detector, Lang, Pack, Sensitivity};

const SENS: [(&str, Sensitivity); 3] = [
    ("Low", Sensitivity::Low),
    ("Medium", Sensitivity::Medium),
    ("High", Sensitivity::High),
];
/// Reporting buckets, by characters.
const BUCKETS: [(usize, usize, &str); 5] = [
    (1, 9, "1-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, 199, "50-199"),
    (200, usize::MAX, "200+"),
];
/// Inputs up to this many characters are timed.
const LATENCY_MAX_CHARS: usize = 256;

struct Args {
    cmd: String,
    corpus: String,
    split: String,
    langs: Option<Vec<Lang>>,
    wasm: bool,
    lang: Option<Lang>,
    fpr_scale: f64,
    l2: f64,
    tsv: Option<String>,
    examples: usize,
    no_latency: bool,
    write: bool,
    /// `sizes --check`: fail when over a budget.
    write_check: bool,
    /// `fit --mode targets|budget` (default: targets for en, budget otherwise).
    fit_mode: Option<String>,
    /// For latency estimates: this many extra copies of the first pack, under other codes.
    clones: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        cmd: "report".into(),
        corpus: "v1".into(),
        split: "dev".into(),
        langs: None,
        wasm: false,
        lang: None,
        fpr_scale: 1.0,
        l2: 1.0,
        tsv: None,
        examples: 0,
        no_latency: false,
        write: false,
        write_check: false,
        fit_mode: None,
        clones: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let (k, v) = match arg.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (arg.clone(), None),
        };
        let mut val = || {
            v.clone()
                .or_else(|| it.next())
                .ok_or(format!("{k} needs a value"))
        };
        match k.as_str() {
            "report" | "fit" | "gate" | "latency" | "check" | "sizes" | "dump" | "slow"
            | "ranks" => a.cmd = k,
            "--corpus" => a.corpus = val()?,
            "--split" => a.split = val()?,
            "--langs" => {
                let list = val()?;
                a.langs = Some(
                    list.split(',')
                        .map(|c| Lang::from_code(c).ok_or(format!("bad language {c}")))
                        .collect::<Result<_, _>>()?,
                );
            }
            "--lang" => a.lang = Some(Lang::from_code(&val()?).ok_or("bad --lang")?),
            "--wasm" => a.wasm = true,
            "--no-latency" => a.no_latency = true,
            "--write" => a.write = true,
            "--check" => a.write_check = true,
            "--mode" => a.fit_mode = Some(val()?),
            "--clones" => a.clones = val()?.parse().map_err(|_| "bad --clones")?,
            "--fpr-scale" => a.fpr_scale = val()?.parse().map_err(|_| "bad --fpr-scale")?,
            "--l2" => a.l2 = val()?.parse().map_err(|_| "bad --l2")?,
            "--tsv" => a.tsv = Some(val()?),
            "--examples" => a.examples = val()?.parse().map_err(|_| "bad --examples")?,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("main.rs")
                        .lines()
                        .take_while(|l| l.starts_with("//!"))
                        .map(|l| l.trim_start_matches("//!").trim_start_matches(' '))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gib-bench: {e}");
            std::process::exit(2);
        }
    };
    let code = match args.cmd.as_str() {
        "fit" => cmd_fit(&args),
        "gate" => cmd_gate(&args),
        "latency" => cmd_latency(&args),
        "check" => cmd_check(&args),
        "dump" => cmd_dump(&args),
        "slow" => cmd_slow(&args),
        "ranks" => cmd_ranks(&args),
        "sizes" => cmd_sizes(&args),
        _ => cmd_report(&args),
    };
    if let Err(e) = code {
        eprintln!("gib-bench: {e}");
        std::process::exit(1);
    }
}

fn detector(args: &Args) -> Detector {
    let d = Detector::new();
    let mut d = match &args.langs {
        Some(l) => d.languages(l),
        None => d,
    };
    if args.clones > 0 {
        let first = d.packs()[0].to_bytes();
        let codes = [
            "sv", "da", "fi", "no", "hu", "cs", "ro", "hr", "sk", "sl", "et", "lv",
        ];
        for code in codes.iter().take(args.clones) {
            let mut blob = first.clone();
            // The model's language code is at bytes 6..8 of the model, after the 8-byte container header
            blob[8 + 6..8 + 8].copy_from_slice(code.as_bytes());
            let cal = String::from_utf8_lossy(
                &blob[8 + u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize..],
            )
            .replace("lang en", &format!("lang {code}"));
            let model_len = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
            let pack = Pack::from_parts(&blob[8..8 + model_len], &cal).expect("clone");
            d = d.with_pack(pack);
        }
    }
    d
}

// ------------------------------------------------------------------ metrics --

#[derive(Default, Clone, Copy)]
struct Counts {
    tp: usize,
    fnn: usize,
    fp: usize,
    tn: usize,
}

impl Counts {
    fn add(&mut self, pos: bool, hit: bool) {
        match (pos, hit) {
            (true, true) => self.tp += 1,
            (true, false) => self.fnn += 1,
            (false, true) => self.fp += 1,
            (false, false) => self.tn += 1,
        }
    }
    fn recall(&self) -> f64 {
        ratio(self.tp, self.tp + self.fnn)
    }
    fn fpr(&self) -> f64 {
        ratio(self.fp, self.fp + self.tn)
    }
    fn precision(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }
    fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

fn pct(v: f64) -> String {
    if v.is_nan() {
        "–".into()
    } else if v > 0.0 && v < 0.001 {
        format!("{:.3}%", v * 100.0)
    } else if v < 0.1 {
        format!("{:.2}%", v * 100.0)
    } else {
        format!("{:.1}%", v * 100.0)
    }
}

fn bucket_of(chars: usize) -> usize {
    BUCKETS
        .iter()
        .position(|&(lo, hi, _)| chars >= lo && chars <= hi)
        .unwrap_or(0)
}

/// One system's verdicts at the three sensitivities, per sample (None: no verdict).
struct System {
    name: String,
    verdicts: Vec<Option<[bool; 3]>>,
}

fn our_system(det: &Detector, samples: &[Sample]) -> (System, Vec<(f32, Option<Lang>)>) {
    let mut verdicts = Vec::with_capacity(samples.len());
    let mut detail = Vec::with_capacity(samples.len());
    for s in samples {
        let d = det.detect(&s.text);
        let v = [
            d.score >= Sensitivity::Low.min_score(),
            d.score >= Sensitivity::Medium.min_score(),
            d.score >= Sensitivity::High.min_score(),
        ];
        verdicts.push(Some(v));
        // The language whatever the sensitivity: detect at the most lenient level
        let lang = if d.score >= Sensitivity::High.min_score() {
            det.clone()
                .sensitivity(Sensitivity::High)
                .detect(&s.text)
                .language
        } else {
            None
        };
        detail.push((d.score, lang));
    }
    (
        System {
            name: "gon6".into(),
            verdicts,
        },
        detail,
    )
}

fn baseline_system(b: &Baseline, samples: &[Sample]) -> System {
    System {
        name: b.name.to_string(),
        verdicts: samples
            .iter()
            .map(|s| b.verdicts.get(&s.id).copied())
            .collect(),
    }
}

/// Whether sample `s` counts in the totals for the languages `scope`.
fn counted(s: &Sample, scope: &BTreeSet<String>) -> Option<bool> {
    match s.label {
        Label::Pos if scope.contains(&s.lang) => Some(true),
        Label::Neg => Some(false),
        _ => None,
    }
}

// ------------------------------------------------------------------- report --

fn uptime() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "?".into())
}

fn cmd_report(args: &Args) -> Result<(), String> {
    let samples = corpus::load(&args.corpus, &args.split)?;
    let det = detector(args);
    let enabled: Vec<Lang> = det.enabled().collect();
    let scope: BTreeSet<String> = enabled.iter().map(|l| l.code().to_string()).collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# gibberish-or-not benchmark: corpus {}, split {}\n",
        args.corpus, args.split
    );
    let _ = writeln!(
        out,
        "Detector: gibberish-or-not {} with {} (counted positives: these languages; negatives: all junk). Host load average: {}.\n",
        gibberish_or_not::VERSION,
        enabled.iter().map(|l| l.code()).collect::<Vec<_>>().join(", "),
        uptime()
    );
    let (ours, detail) = our_system(&det, &samples);
    let mut systems = vec![ours];
    for b in corpus::load_baselines(&args.corpus, &args.split) {
        systems.push(baseline_system(&b, &samples));
    }
    let mut tsv = String::new();

    // Samples
    let mut by: BTreeMap<(String, Label), usize> = BTreeMap::new();
    for s in &samples {
        *by.entry((s.lang.clone(), s.label)).or_default() += 1;
    }
    let langs: BTreeSet<String> = samples.iter().map(|s| s.lang.clone()).collect();
    let _ = writeln!(out, "## Samples\n\n| lang | positives | negatives derived from it | info |\n|---|---:|---:|---:|");
    for l in &langs {
        let g = |lab| by.get(&(l.clone(), lab)).copied().unwrap_or(0);
        let _ = writeln!(
            out,
            "| {l} | {} | {} | {} |",
            g(Label::Pos),
            g(Label::Neg),
            g(Label::Info)
        );
    }

    // Overall
    let _ = writeln!(out, "\n## Overall\n\n| system | sensitivity | precision | recall | F1 | false positives | FPR |\n|---|---|---:|---:|---:|---:|---:|");
    for sys in &systems {
        for (i, (sn, _)) in SENS.iter().enumerate() {
            let mut c = Counts::default();
            for (s, v) in samples.iter().zip(&sys.verdicts) {
                if let (Some(pos), Some(v)) = (counted(s, &scope), v) {
                    c.add(pos, v[i]);
                }
            }
            let _ = writeln!(
                out,
                "| {} | {sn} | {} | {} | {:.3} | {} / {} | {} |",
                sys.name,
                pct(c.precision()),
                pct(c.recall()),
                c.f1(),
                c.fp,
                c.fp + c.tn,
                pct(c.fpr())
            );
            let _ = writeln!(
                tsv,
                "overall\t{}\t{sn}\trecall\t{:.6}",
                sys.name,
                c.recall()
            );
            let _ = writeln!(tsv, "overall\t{}\t{sn}\tfpr\t{:.6}", sys.name, c.fpr());
        }
    }

    // By length
    let _ = writeln!(out, "\n## By length (characters)\n\nRecall, FPR and F1 per bucket; the rule for 6.0 is higher recall at an equal or lower FPR than both baselines in every bucket from 10 characters, and at least as good below.\n");
    let _ = write!(out, "| bucket | positives / negatives |");
    for sys in &systems {
        for (sn, _) in SENS {
            let _ = write!(out, " {}@{sn} |", sys.name);
        }
    }
    let _ = write!(out, "\n|---|---:|");
    for _ in 0..systems.len() * 3 {
        let _ = write!(out, "---|");
    }
    let _ = writeln!(out);
    for (bi, &(_, _, bn)) in BUCKETS.iter().enumerate() {
        let (mut np, mut nn) = (0, 0);
        for s in &samples {
            if bucket_of(s.chars()) == bi {
                match counted(s, &scope) {
                    Some(true) => np += 1,
                    Some(false) => nn += 1,
                    None => {}
                }
            }
        }
        let _ = write!(out, "| {bn} | {np} / {nn} |");
        for sys in &systems {
            for (i, (sn, _)) in SENS.iter().enumerate() {
                let mut c = Counts::default();
                for (s, v) in samples.iter().zip(&sys.verdicts) {
                    if bucket_of(s.chars()) != bi {
                        continue;
                    }
                    if let (Some(pos), Some(v)) = (counted(s, &scope), v) {
                        c.add(pos, v[i]);
                    }
                }
                let _ = write!(
                    out,
                    " {} / {} / {:.3} |",
                    pct(c.recall()),
                    pct(c.fpr()),
                    c.f1()
                );
                let _ = writeln!(
                    tsv,
                    "bucket\t{}\t{sn}\t{bn}\trecall\t{:.6}",
                    sys.name,
                    c.recall()
                );
                let _ = writeln!(tsv, "bucket\t{}\t{sn}\t{bn}\tfpr\t{:.6}", sys.name, c.fpr());
            }
        }
        let _ = writeln!(out);
    }

    // Positives by category
    let _ = writeln!(
        out,
        "\n## Positives by language and category (recall at Low / Medium / High)\n"
    );
    let _ = write!(out, "| lang | category | n |");
    for sys in &systems {
        let _ = write!(out, " {} |", sys.name);
    }
    let _ = write!(out, "\n|---|---|---:|");
    for _ in &systems {
        let _ = write!(out, "---|");
    }
    let _ = writeln!(out);
    let mut cats: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, s) in samples.iter().enumerate() {
        if s.label == Label::Pos {
            cats.entry((s.lang.clone(), s.category.clone()))
                .or_default()
                .push(i);
        }
    }
    for ((lang, cat), idx) in &cats {
        let _ = write!(out, "| {lang} | {cat} | {} |", idx.len());
        for sys in &systems {
            let mut cells = Vec::new();
            for i in 0..3 {
                let (mut hit, mut n) = (0, 0);
                for &j in idx {
                    if let Some(v) = sys.verdicts[j] {
                        n += 1;
                        hit += usize::from(v[i]);
                    }
                }
                cells.push(pct(ratio(hit, n)));
            }
            let _ = write!(out, " {} |", cells.join(" / "));
        }
        let _ = writeln!(out);
    }

    // Negatives by category
    let _ = writeln!(
        out,
        "\n## Negatives by category (false-positive rate at Low / Medium / High)\n"
    );
    let _ = write!(out, "| category | n |");
    for sys in &systems {
        let _ = write!(out, " {} |", sys.name);
    }
    let _ = write!(out, "\n|---|---:|");
    for _ in &systems {
        let _ = write!(out, "---|");
    }
    let _ = writeln!(out);
    let mut ncats: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, s) in samples.iter().enumerate() {
        if s.label == Label::Neg {
            ncats.entry(s.category.clone()).or_default().push(i);
        }
    }
    for (cat, idx) in &ncats {
        let _ = write!(out, "| {cat} | {} |", idx.len());
        for sys in &systems {
            let mut cells = Vec::new();
            for i in 0..3 {
                let (mut hit, mut n) = (0, 0);
                for &j in idx {
                    if let Some(v) = sys.verdicts[j] {
                        n += 1;
                        hit += usize::from(v[i]);
                    }
                }
                cells.push(pct(ratio(hit, n)));
                let _ = writeln!(
                    tsv,
                    "negcat\t{}\t{}\t{cat}\tfpr\t{:.6}",
                    sys.name,
                    SENS[i].0,
                    ratio(hit, n)
                );
            }
            let _ = write!(out, " {} |", cells.join(" / "));
        }
        let _ = writeln!(out);
    }

    // Languages: recall and confusion
    let _ = writeln!(out, "\n## Languages\n\nRecall of each language's positives (Low / Medium / High), and which language the detector reports for them (at High, the most lenient level, so every accepted text has a language).\n");
    let _ = write!(out, "| lang | n | gon6 recall |");
    for l in &enabled {
        let _ = write!(out, " → {} |", l.code());
    }
    let mut own_junk: Vec<(String, (usize, String))> = Vec::new();
    let _ = write!(out, " → none |\n|---|---:|---|");
    for _ in &enabled {
        let _ = write!(out, "---:|");
    }
    let _ = writeln!(out, "---:|");
    for l in &langs {
        let idx: Vec<usize> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.label == Label::Pos && &s.lang == l)
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let rec: Vec<String> = (0..3)
            .map(|i| {
                pct(ratio(
                    idx.iter()
                        .filter(|&&j| systems[0].verdicts[j].is_some_and(|v| v[i]))
                        .count(),
                    idx.len(),
                ))
            })
            .collect();
        let _ = write!(out, "| {l} | {} | {} |", idx.len(), rec.join(" / "));
        for e in &enabled {
            let n = idx.iter().filter(|&&j| detail[j].1 == Some(*e)).count();
            let _ = write!(out, " {} |", pct(ratio(n, idx.len())));
        }
        let none = idx.iter().filter(|&&j| detail[j].1.is_none()).count();
        let _ = writeln!(out, " {} |", pct(ratio(none, idx.len())));
        own_junk.push((l.clone(), {
            let own: Vec<usize> = samples
                .iter()
                .enumerate()
                .filter(|(_, s)| s.label == Label::Neg && &s.lang == l)
                .map(|(i, _)| i)
                .collect();
            let r: Vec<String> = (0..3)
                .map(|i| {
                    pct(ratio(
                        own.iter()
                            .filter(|&&j| systems[0].verdicts[j].is_some_and(|v| v[i]))
                            .count(),
                        own.len(),
                    ))
                })
                .collect();
            (own.len(), r.join(" / "))
        }));
        for (i, (sn, _)) in SENS.iter().enumerate() {
            let _ = writeln!(
                tsv,
                "lang\tgon6\t{sn}\t{l}\trecall\t{:.6}",
                ratio(
                    idx.iter()
                        .filter(|&&j| systems[0].verdicts[j].is_some_and(|v| v[i]))
                        .count(),
                    idx.len()
                )
            );
        }
    }

    let _ = writeln!(out, "\nFalse-positive rate of each language's own junk (the negatives derived from its text; Low / Medium / High; docs/languages.md gate 3):\n\n| lang | negatives | gon6 FPR |\n|---|---:|---|");
    for (l, (n, r)) in &own_junk {
        let _ = writeln!(out, "| {l} | {n} | {r} |");
    }

    // Examples
    if args.examples > 0 {
        let _ = writeln!(out, "\n## Misclassified examples (gon6 at Medium)\n");
        let mut shown: BTreeMap<String, usize> = BTreeMap::new();
        for (j, s) in samples.iter().enumerate() {
            let Some(pos) = counted(s, &scope) else {
                continue;
            };
            let hit = systems[0].verdicts[j].is_some_and(|v| v[1]);
            if hit == pos {
                continue;
            }
            let key = format!(
                "{} {} {}",
                if pos { "FN" } else { "FP" },
                s.lang,
                s.category
            );
            let n = shown.entry(key.clone()).or_default();
            if *n < args.examples {
                *n += 1;
                let t: String = s.text.chars().take(100).collect();
                let _ = writeln!(
                    out,
                    "- {key} ({:.2}): `{}`",
                    detail[j].0,
                    t.replace('`', "'").replace('\n', "\\n")
                );
            }
        }
    }

    // Latency and sizes
    if !args.no_latency {
        out.push_str(&latency_report(
            &det,
            &samples,
            &corpus::load_baselines(&args.corpus, &args.split),
        ));
    }
    out.push_str(&sizes(args));
    print!("{out}");
    if let Some(path) = &args.tsv {
        std::fs::write(path, tsv).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

// ------------------------------------------------------------------ latency --

/// Times one check; checks under 2 µs are repeated so the timer's cost doesn't count.
fn timed(det: &Detector, text: &str) -> f64 {
    let t0 = Instant::now();
    black_box(det.detect(black_box(text)));
    let first = t0.elapsed();
    if first >= Duration::from_micros(2) {
        return first.as_nanos() as f64;
    }
    let reps = (2_000 / first.as_nanos().max(1)).clamp(2, 256) as u32;
    let t1 = Instant::now();
    for _ in 0..reps {
        black_box(det.detect(black_box(text)));
    }
    t1.elapsed().as_nanos() as f64 / f64::from(reps)
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    (q(0.5), q(0.99), v.iter().sum::<f64>() / v.len() as f64)
}

fn us(ns: f64) -> String {
    if ns.is_nan() {
        "–".into()
    } else if ns < 1000.0 {
        format!("{:.0} ns", ns)
    } else {
        format!("{:.1} µs", ns / 1000.0)
    }
}

fn latency_samples(samples: &[Sample]) -> Vec<&Sample> {
    samples
        .iter()
        .filter(|s| s.chars() <= LATENCY_MAX_CHARS)
        .collect()
}

fn latency_report(det: &Detector, samples: &[Sample], baselines: &[Baseline]) -> String {
    let timed_samples = latency_samples(samples);
    for s in timed_samples.iter().take(500) {
        black_box(det.detect(&s.text));
    }
    let ns: Vec<f64> = timed_samples.iter().map(|s| timed(det, &s.text)).collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n## Latency (native, this build)\n\nPer check, over the {} samples of up to {LATENCY_MAX_CHARS} characters (positives, negatives and info), with {} enabled. Host load average: {}.\n",
        timed_samples.len(),
        det.enabled().map(|l| l.code()).collect::<Vec<_>>().join(", "),
        uptime()
    );
    let _ = writeln!(out, "| checker | p50 | p99 | mean |\n|---|---:|---:|---:|");
    let (p50, p99, mean) = quantiles(ns.clone());
    let _ = writeln!(
        out,
        "| gibberish-or-not 6 (any sensitivity) | {} | {} | {} |",
        us(p50),
        us(p99),
        us(mean)
    );
    for b in baselines {
        let v: Vec<f64> = timed_samples
            .iter()
            .filter_map(|s| b.ns.get(&s.id).copied())
            .filter(|x| x.is_finite())
            .collect();
        let (p50, p99, mean) = quantiles(v);
        let _ = writeln!(
            out,
            "| {} @Medium (frozen, measured when the baselines were run) | {} | {} | {} |",
            b.name,
            us(p50),
            us(p99),
            us(mean)
        );
    }
    for (name, label) in [("negatives", Label::Neg), ("positives", Label::Pos)] {
        let v: Vec<f64> = timed_samples
            .iter()
            .zip(&ns)
            .filter(|(s, _)| s.label == label)
            .map(|(_, &t)| t)
            .collect();
        let (p50, p99, mean) = quantiles(v);
        let _ = writeln!(
            out,
            "| gibberish-or-not 6, {name} only | {} | {} | {} |",
            us(p50),
            us(p99),
            us(mean)
        );
    }
    out
}

fn cmd_latency(args: &Args) -> Result<(), String> {
    let samples = corpus::load(&args.corpus, &args.split)?;
    let det = detector(args);
    print!("{}", latency_report(&det, &samples, &[]));
    Ok(())
}

// -------------------------------------------------------------------- sizes --

fn sizes(args: &Args) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n## Sizes\n\n| language | model.bin | calibration.txt | words |\n|---|---:|---:|---:|"
    );
    let mut total = 0usize;
    for p in Detector::new().packs() {
        let dir = corpus::root().join("languages").join(p.lang().code());
        let cal = std::fs::metadata(dir.join("calibration.txt"))
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        total += p.model_size() + cal;
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} |",
            p.lang().code(),
            kb(p.model_size()),
            kb(cal),
            p.word_count()
        );
    }
    let _ = writeln!(out, "| all | {} | | |", kb(total));
    if args.wasm {
        #[cfg(not(target_os = "wasi"))]
        out.push_str(&wasm::report(args));
    }
    out
}

/// Size budgets (docs/languages.md): bytes of model.bin + calibration.txt per language, and
/// of the WASM module with every language after wasm-opt -Oz.
const BUDGET_EN: usize = 1 << 20;
const BUDGET_OTHER: usize = 400 << 10;
const BUDGET_WASM: usize = 5_500 << 10;

/// `sizes [--wasm] [--check]`: the sizes table; with `--check`, fails over a budget.
fn cmd_sizes(args: &Args) -> Result<(), String> {
    print!("{}", sizes(args));
    if !args.write_check {
        return Ok(());
    }
    let mut over = Vec::new();
    for p in Detector::new().packs() {
        let dir = corpus::root().join("languages").join(p.lang().code());
        let cal = std::fs::metadata(dir.join("calibration.txt"))
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        let size = p.model_size() + cal;
        let budget = if p.lang() == Lang::En {
            BUDGET_EN
        } else {
            BUDGET_OTHER
        };
        if size > budget {
            over.push(format!(
                "{}: {} > {}",
                p.lang().code(),
                kb(size),
                kb(budget)
            ));
        }
    }
    #[cfg(not(target_os = "wasi"))]
    if args.wasm {
        let target = std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| corpus::root().join("target"));
        let opt = target.join("gon-wasm").join("gon_wasm_probe_bg.opt.wasm");
        let size = std::fs::metadata(&opt)
            .map(|m| m.len() as usize)
            .map_err(|e| format!("{}: {e}", opt.display()))?;
        if size > BUDGET_WASM {
            over.push(format!("wasm: {} > {}", kb(size), kb(BUDGET_WASM)));
        }
    }
    if over.is_empty() {
        println!(
            "SIZES_OK (budgets: en {}, other languages {}, wasm {})",
            kb(BUDGET_EN),
            kb(BUDGET_OTHER),
            kb(BUDGET_WASM)
        );
        Ok(())
    } else {
        Err(format!("over budget: {}", over.join("; ")))
    }
}

fn kb(n: usize) -> String {
    format!("{:.1} KB", n as f64 / 1024.0)
}

#[cfg(not(target_os = "wasi"))]
mod wasm {
    //! Builds the WASM module (bench/wasm-probe: the library with its `wasm` feature),
    //! shrinks it with wasm-opt -Oz, and times it under node (wasm32-unknown-unknown with
    //! wasm-bindgen) and wasmtime (wasm32-wasip1: this tool's `latency` command).
    use super::*;
    use std::process::Command;

    fn run(cmd: &mut Command) -> Result<String, String> {
        let o = cmd.output().map_err(|e| format!("{cmd:?}: {e}"))?;
        if !o.status.success() {
            return Err(format!(
                "{cmd:?} failed: {}",
                String::from_utf8_lossy(&o.stderr)
                    .lines()
                    .rev()
                    .take(15)
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        Ok(String::from_utf8_lossy(&o.stdout).into_owned())
    }

    pub fn report(args: &Args) -> String {
        match build_and_time(args) {
            Ok(s) => s,
            Err(e) => format!("\nWASM: not measured: {e}\n"),
        }
    }

    fn build_and_time(args: &Args) -> Result<String, String> {
        let root = corpus::root();
        let target = std::env::var("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.join("target"));
        let out_dir = target.join("gon-wasm");
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        run(Command::new("cargo").current_dir(&root).args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "-p",
            "gon-wasm-probe",
        ]))?;
        let raw = target.join("wasm32-unknown-unknown/release/gon_wasm_probe.wasm");
        run(Command::new("wasm-bindgen")
            .arg(&raw)
            .args(["--target", "nodejs", "--out-dir"])
            .arg(&out_dir))?;
        let bg = out_dir.join("gon_wasm_probe_bg.wasm");
        let opt = out_dir.join("gon_wasm_probe_bg.opt.wasm");
        run(Command::new("wasm-opt")
            .args([
                "-Oz",
                "--enable-bulk-memory",
                "--enable-nontrapping-float-to-int",
                "--enable-sign-ext",
                "--enable-mutable-globals",
            ])
            .arg(&bg)
            .arg("-o")
            .arg(&opt))?;
        let size =
            |p: &std::path::Path| std::fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0);
        let mut out = String::new();
        let _ = writeln!(out, "\n## WASM\n\n| artefact | size |\n|---|---:|");
        let _ = writeln!(
            out,
            "| cargo build --release (wasm32-unknown-unknown, every language) | {} |",
            kb(size(&raw))
        );
        let _ = writeln!(out, "| after wasm-bindgen | {} |", kb(size(&bg)));
        let _ = writeln!(
            out,
            "| after wasm-opt -Oz (what a web page ships) | {} |",
            kb(size(&opt))
        );
        // node latency: the wasm-bindgen API, string passing included
        std::fs::copy(&opt, &bg).map_err(|e| e.to_string())?;
        let samples = corpus::load(&args.corpus, &args.split)?;
        let texts: Vec<String> = latency_samples(&samples)
            .iter()
            .map(|s| s.text.clone())
            .collect();
        let json = format!(
            "[{}]",
            texts
                .iter()
                .map(|t| json_string(t))
                .collect::<Vec<_>>()
                .join(",")
        );
        std::fs::write(out_dir.join("texts.json"), json).map_err(|e| e.to_string())?;
        if args.no_latency {
            return Ok(out);
        }
        std::fs::write(out_dir.join("latency.js"), NODE_SCRIPT).map_err(|e| e.to_string())?;
        let node = run(Command::new("node")
            .arg(out_dir.join("latency.js"))
            .current_dir(&out_dir))?;
        let _ = writeln!(
            out,
            "\n| WASM latency (same samples as native) | p50 | p99 | mean |\n|---|---:|---:|---:|"
        );
        out.push_str(&node);
        // wasmtime latency: this tool built for wasm32-wasip1
        let built = run(Command::new("cargo").current_dir(&root).args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip1",
            "-p",
            "gib-bench",
        ]));
        match built {
            Ok(_) => {
                let wasm = target.join("wasm32-wasip1/release/gib-bench.wasm");
                let dir = format!("{}::{}", root.display(), root.display());
                let mut cmd = Command::new("wasmtime");
                cmd.args(["run", "--dir", &dir]).arg(&wasm).args([
                    "latency",
                    "--corpus",
                    &args.corpus,
                    "--split",
                    &args.split,
                ]);
                if let Some(l) = &args.langs {
                    cmd.arg("--langs")
                        .arg(l.iter().map(|l| l.code()).collect::<Vec<_>>().join(","));
                }
                let o = run(&mut cmd)?;
                for line in o
                    .lines()
                    .filter(|l| l.starts_with("| gibberish-or-not 6 (any"))
                {
                    let _ = writeln!(
                        out,
                        "{}",
                        line.replace(
                            "gibberish-or-not 6 (any sensitivity)",
                            "wasm32-wasip1 under wasmtime"
                        )
                    );
                }
            }
            Err(e) => {
                let _ = writeln!(
                    out,
                    "| wasm32-wasip1 under wasmtime | not built: {} | | |",
                    e.lines().next().unwrap_or("")
                );
            }
        }
        Ok(out)
    }

    fn json_string(s: &str) -> String {
        let mut o = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""),
                '\\' => o.push_str("\\\\"),
                c if (c as u32) < 0x20 || c == '\u{2028}' || c == '\u{2029}' => {
                    let _ = write!(o, "\\u{:04x}", c as u32);
                }
                c => o.push(c),
            }
        }
        o.push('"');
        o
    }

    const NODE_SCRIPT: &str = r#"
const g = require('./gon_wasm_probe.js');
const texts = require('./texts.json');
// score(): no result object, so no garbage collection in the measurement
for (let r = 0; r < 3; r++) for (const t of texts) g.score(t);
const ns = [];
for (const t of texts) {
  let reps = 1, total = 0n;
  for (;;) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < reps; i++) g.score(t);
    total = process.hrtime.bigint() - t0;
    if (total >= 20000n || reps >= 256) break;
    reps *= 4;
  }
  ns.push(Number(total) / reps);
}
ns.sort((a, b) => a - b);
const q = p => ns[Math.round((ns.length - 1) * p)];
const f = n => n < 1000 ? `${n.toFixed(0)} ns` : `${(n / 1000).toFixed(1)} µs`;
const mean = ns.reduce((a, b) => a + b, 0) / ns.length;
console.log(`| wasm32-unknown-unknown under node ${process.versions.node} (wasm-bindgen score(), string passing included) | ${f(q(0.5))} | ${f(q(0.99))} | ${f(mean)} |`);
"#;
}

// ---------------------------------------------------------------------- fit --

fn cmd_fit(args: &Args) -> Result<(), String> {
    let lang = args.lang.ok_or("fit needs --lang")?;
    let dir = corpus::root().join("languages").join(lang.code());
    let model =
        std::fs::read(dir.join("model.bin")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let cal_path = dir.join("calibration.txt");
    let cal = std::fs::read_to_string(&cal_path).unwrap_or_default();
    let pack = Pack::from_parts(&model, &cal)
        .or_else(|_| Pack::from_parts(&model, &placeholder_calibration(lang)))
        .map_err(|e| e.to_string())?;
    let train = corpus::load(&args.corpus, "train")?;
    let dev = corpus::load(&args.corpus, "dev")?;
    let r = fit::fit(
        pack,
        &train,
        &dev,
        fit::Targets {
            scale: args.fpr_scale,
            // English is the base pack: it sets the per-bucket targets every later pack
            // shares; every other pack spends gate 2's budget (docs/languages.md)
            mode: match args.fit_mode.as_deref() {
                Some("targets") => fit::Mode::Targets,
                Some("budget") => fit::Mode::Budget,
                Some(other) => return Err(format!("--mode {other}: targets or budget")),
                None if lang == Lang::En => fit::Mode::Targets,
                None => fit::Mode::Budget,
            },
        },
        args.l2,
    )?;
    // Check that the library accepts it
    Pack::from_parts(&model, &r.calibration)
        .map_err(|e| format!("fitted calibration doesn't load: {e}"))?;
    std::fs::write(&cal_path, &r.calibration).map_err(|e| e.to_string())?;
    for l in &r.lines {
        println!("{l}");
    }
    println!("wrote {}", cal_path.display());
    Ok(())
}

fn placeholder_calibration(lang: Lang) -> String {
    let names = Detector::feature_names();
    format!(
        "gon-calibration 1\nlang {}\nfeatures {}\nbucket 1 inf {} | -1 0 1 2 3\n",
        lang.code(),
        names.join(" "),
        vec!["0"; names.len()].join(" ")
    )
}

// --------------------------------------------------------------------- gate --

/// CI: the dev split, with every language compiled in, must keep
///
/// * each language's recall (overall and per length bucket, at each sensitivity) within
///   RECALL_TOLERANCE of `bench/expected/<corpus>/<code>.tsv` (written by `gate --write
///   --lang <code>` in the PR that changes that language on purpose);
/// * the junk false-positive rate of each length bucket and sensitivity at most
///   FPR_CEILING_FACTOR times that bucket's calibration target (bench/src/fit.rs), plus
///   FPR_CEILING_SLACK: the budget all languages share.
///
/// Per-language files, so the language PRs don't conflict.
fn cmd_gate(args: &Args) -> Result<(), String> {
    const RECALL_TOLERANCE: f64 = 0.002;
    use fit::{FPR_CEILING_FACTOR, FPR_CEILING_SLACK};
    let samples = corpus::load(&args.corpus, "dev")?;
    let det = Detector::new();
    let (ours, _) = our_system(&det, &samples);
    let dir = corpus::root()
        .join("bench")
        .join("expected")
        .join(&args.corpus);
    let recall_lines = |code: &str| -> Vec<(String, f64)> {
        let mut out = Vec::new();
        for (i, (sn, _)) in SENS.iter().enumerate() {
            let idx: Vec<usize> = samples
                .iter()
                .enumerate()
                .filter(|(_, s)| s.label == Label::Pos && s.lang == code)
                .map(|(j, _)| j)
                .collect();
            let r = ratio(
                idx.iter()
                    .filter(|&&j| ours.verdicts[j].is_some_and(|v| v[i]))
                    .count(),
                idx.len(),
            );
            out.push((format!("{code}\t{sn}\tall\trecall"), r));
            for (bi, &(_, _, bn)) in BUCKETS.iter().enumerate() {
                let in_b: Vec<usize> = idx
                    .iter()
                    .copied()
                    .filter(|&j| bucket_of(samples[j].chars()) == bi)
                    .collect();
                if in_b.len() >= 20 {
                    let r = ratio(
                        in_b.iter()
                            .filter(|&&j| ours.verdicts[j].is_some_and(|v| v[i]))
                            .count(),
                        in_b.len(),
                    );
                    out.push((format!("{code}\t{sn}\t{bn}\trecall"), r));
                }
            }
        }
        out
    };
    if args.write {
        let lang = args.lang.ok_or("gate --write needs --lang")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut text = format!("# gib-bench gate --write --lang {}: dev-split recall with every language compiled in\n", lang.code());
        for (k, v) in recall_lines(lang.code()) {
            let _ = writeln!(text, "{k}\t{v:.6}");
        }
        let path = dir.join(format!("{}.tsv", lang.code()));
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
        return Ok(());
    }
    let mut failures = 0;
    let mut checked = 0;
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .collect();
    files.sort();
    for path in &files {
        let code = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let now: BTreeMap<String, f64> = recall_lines(&code).into_iter().collect();
        for line in std::fs::read_to_string(path)
            .map_err(|e| e.to_string())?
            .lines()
        {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let Some((k, v)) = line.rsplit_once('\t') else {
                continue;
            };
            let Ok(e) = v.parse::<f64>() else { continue };
            checked += 1;
            match now.get(k) {
                None => {
                    println!("MISSING {k}");
                    failures += 1;
                }
                Some(n) if *n < e - RECALL_TOLERANCE => {
                    println!("REGRESSION {k}: expected {e:.4}, got {n:.4}");
                    failures += 1;
                }
                Some(_) => {}
            }
        }
    }
    for (bi, &(lo, _, bn)) in BUCKETS.iter().enumerate() {
        for (i, (sn, _)) in SENS.iter().enumerate() {
            let mut c = Counts::default();
            for (s, v) in samples.iter().zip(&ours.verdicts) {
                if s.label == Label::Neg && bucket_of(s.chars()) == bi {
                    c.add(false, v.is_some_and(|v| v[i]));
                }
            }
            // fit::targets is (High, Medium, Low); SENS is (Low, Medium, High)
            let target = fit::targets(lo as u32)[2 - i];
            let ceiling = FPR_CEILING_FACTOR * target + FPR_CEILING_SLACK;
            checked += 1;
            if c.fpr() > ceiling {
                println!(
                    "FPR {bn} {sn}: {:.4}% is above the ceiling {:.4}%",
                    c.fpr() * 100.0,
                    ceiling * 100.0
                );
                failures += 1;
            }
        }
    }
    println!(
        "gate: {checked} checks ({} language files in {}; recall tolerance {RECALL_TOLERANCE}, junk FPR ceiling {FPR_CEILING_FACTOR} x target + {FPR_CEILING_SLACK}): {failures} failures",
        files.len(),
        dir.display()
    );
    if failures > 0 {
        return Err("accuracy gate failed".into());
    }
    println!("GATE_OK");
    Ok(())
}

// --------------------------------------------------------------------- dump --

/// Every sample of the split with its score and language, and the baselines' verdicts
/// (Low/Medium/High as three digits), as TSV on stdout: for analysis outside Rust.
fn cmd_dump(args: &Args) -> Result<(), String> {
    use std::io::Write;
    let samples = corpus::load(&args.corpus, &args.split)?;
    let det = detector(args).sensitivity(Sensitivity::High);
    let baselines = corpus::load_baselines(&args.corpus, &args.split);
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let _ = writeln!(
        out,
        "id\tlang\tlabel\tcategory\tchars\tscore\tdetected\t{}\ttext",
        baselines
            .iter()
            .map(|b| b.name)
            .collect::<Vec<_>>()
            .join("\t")
    );
    for s in &samples {
        let d = det.detect(&s.text);
        let label = match s.label {
            Label::Pos => "pos",
            Label::Neg => "neg",
            Label::Info => "info",
        };
        let _ = write!(
            out,
            "{}\t{}\t{label}\t{}\t{}\t{}\t{}",
            s.id,
            s.lang,
            s.category,
            s.chars(),
            d.score,
            d.language.map_or("-", |l| l.code())
        );
        for b in &baselines {
            let v = b.verdicts.get(&s.id).map_or("---".to_string(), |v| {
                v.iter().map(|&x| if x { '1' } else { '0' }).collect()
            });
            let _ = write!(out, "\t{v}");
        }
        let t: String = s.text.chars().take(200).collect();
        let _ = writeln!(
            out,
            "\t{}\t{}",
            s.source,
            t.replace('\t', "\\t")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
        );
    }
    Ok(())
}

/// Quantiles of the best n-gram rank (see model::NgramPass::rank) of positives and
/// negatives, by length bucket: for choosing the junk gate.
fn cmd_ranks(args: &Args) -> Result<(), String> {
    let samples = corpus::load(&args.corpus, &args.split)?;
    let det = detector(args);
    let mut by: BTreeMap<(usize, bool), Vec<f64>> = BTreeMap::new();
    for s in &samples {
        let pos = match s.label {
            Label::Pos => true,
            Label::Neg => false,
            Label::Info => continue,
        };
        if pos && !det.enabled().any(|l| l.code() == s.lang) {
            continue;
        }
        if let Some(r) = det.best_rank(&s.text) {
            by.entry((bucket_of(s.chars()), pos))
                .or_default()
                .push(f64::from(r));
        }
    }
    for ((b, pos), mut v) in by {
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!(
            "{} {}: n={} min={:.3} q001={:.3} q01={:.3} q10={:.3} q50={:.3} q90={:.3} max={:.3}",
            BUCKETS[b].2,
            if pos { "pos" } else { "neg" },
            v.len(),
            v[0],
            q(0.001),
            q(0.01),
            q(0.1),
            q(0.5),
            q(0.9),
            v[v.len() - 1]
        );
    }
    Ok(())
}

/// The slowest checks of the split (up to 256 characters), for profiling.
fn cmd_slow(args: &Args) -> Result<(), String> {
    let samples = corpus::load(&args.corpus, &args.split)?;
    let det = detector(args);
    let timed_samples = latency_samples(&samples);
    let mut v: Vec<(f64, &Sample)> = timed_samples
        .iter()
        .map(|s| (timed(&det, &s.text), *s))
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut by_cat: BTreeMap<String, (usize, f64)> = BTreeMap::new();
    for (t, s) in &v {
        let e = by_cat
            .entry(format!("{:?} {}", s.label, s.category))
            .or_default();
        e.0 += 1;
        e.1 += t;
    }
    let mut cats: Vec<_> = by_cat.into_iter().collect();
    cats.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
    println!("total time by category:");
    for (c, (n, t)) in cats.iter().take(15) {
        println!(
            "  {c}: {n} samples, {:.1} ms total, {:.2} µs mean",
            t / 1e6,
            t / *n as f64 / 1e3
        );
    }
    println!("slowest:");
    for (t, s) in v.iter().take(args.examples.max(20)) {
        let text: String = s.text.chars().take(80).collect();
        println!(
            "  {:>8.1} µs {} {} {} | {}",
            t / 1e3,
            s.lang,
            s.category,
            s.chars(),
            text.replace('\n', "\\n")
        );
    }
    Ok(())
}

// -------------------------------------------------------------------- check --

fn cmd_check(args: &Args) -> Result<(), String> {
    use std::io::BufRead;
    let det = detector(args);
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let text = corpus::unescape(&line);
        let d = det.clone().sensitivity(Sensitivity::High).detect(&text);
        let feats = det.features(&text);
        println!(
            "{:.3} {:<5} {}",
            d.score,
            d.language.map_or("-".to_string(), |l| l.code().to_string()),
            text
        );
        for (l, x, chars, rank) in feats {
            let names = Detector::feature_names();
            let shown: Vec<String> = names
                .iter()
                .zip(&x)
                .skip(1)
                .map(|(n, v)| format!("{n}={v:.2}"))
                .collect();
            println!("      {l} chars={chars} rank={rank:.2} {}", shown.join(" "));
        }
    }
    Ok(())
}
