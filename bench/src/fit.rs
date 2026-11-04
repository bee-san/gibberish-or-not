//! `gib-bench fit --lang <code>`: fits a pack's classifier on the train split and its score
//! anchors on the dev split, and writes `languages/<code>/calibration.txt`.
//!
//! Per length bucket (letters in the pack's scripts): an L2-regularised, class-balanced
//! logistic regression on the library's features (Newton's method on standardised
//! features), trained on the language's positives against all junk of the train split.
//! The anchors are the logits at which the dev split's junk false-positive rate reaches each
//! sensitivity's target (High, Medium, Low), so that score 0.5, 0.75 and 0.9 mean the same
//! false-positive rate in every bucket.

use gibberish_or_not::{Detector, Lang, Pack};

use crate::corpus::{Label, Sample};

/// Calibration buckets, by characters: lower bounds (the last is open). They refine the
/// report's buckets (1-9, 10-19, 20-49, 50-199, 200+).
pub const BUCKETS: [u32; 11] = [1, 4, 6, 8, 10, 15, 20, 30, 50, 100, 200];
/// The classifier's weights are fitted per group of buckets (more data per fit), the
/// anchors per bucket: the first bucket of each group.
pub const WEIGHT_GROUPS: [u32; 4] = [1, 10, 20, 50];

fn group_of(chars: u32) -> usize {
    WEIGHT_GROUPS
        .iter()
        .rposition(|&lo| chars >= lo)
        .unwrap_or(0)
}

/// One training example.
pub struct Example {
    pub x: Vec<f64>,
    pub y: bool,
    /// Length in characters.
    pub chars: u32,
    /// The pack's n-gram rank.
    pub rank: f32,
}

/// The features of every positive of `lang` and every negative, for `lang`'s pack.
/// Samples the pack doesn't apply to (vetoed, or no letters in its scripts) are counted as
/// rejected: `None`.
pub fn examples(
    detector: &Detector,
    lang: Lang,
    samples: &[Sample],
) -> Vec<(Option<Example>, bool)> {
    let mut out = Vec::new();
    for s in samples {
        let y = match s.label {
            Label::Pos if s.lang == lang.code() => true,
            Label::Neg => false,
            _ => continue,
        };
        let feats = detector.features(&s.text);
        let ex = feats
            .into_iter()
            .find(|(l, _, _, _)| *l == lang)
            .map(|(_, x, chars, rank)| Example {
                x: x.iter().map(|&v| f64::from(v)).collect(),
                y,
                chars,
                rank,
            });
        out.push((ex, y));
    }
    out
}

fn bucket_of(chars: u32) -> usize {
    BUCKETS.iter().rposition(|&lo| chars >= lo).unwrap_or(0)
}

/// Solves `a x = b` for symmetric positive definite `a` (Cholesky).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        if d <= 1e-12 {
            return None;
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in j + 1..n {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k][i] * b[k];
        }
        b[i] = s / a[i][i];
    }
    Some(b)
}

/// Class-balanced L2 logistic regression. Feature 0 is the bias. Returns weights in the
/// original (unstandardised) feature space.
pub fn logistic(xs: &[&Example], l2: f64) -> Vec<f64> {
    let d = xs[0].x.len();
    let n = xs.len() as f64;
    let n_pos = xs.iter().filter(|e| e.y).count() as f64;
    let n_neg = n - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        // Nothing to separate: a constant logit far on the side of the only class
        let mut w = vec![0.0; d];
        w[0] = if n_pos > 0.0 { 10.0 } else { -10.0 };
        return w;
    }
    let (w_pos, w_neg) = (n / (2.0 * n_pos), n / (2.0 * n_neg));
    // Standardise (feature 0 is the bias; constant features get weight 0)
    let mut mu = vec![0.0; d];
    let mut sd = vec![0.0; d];
    for e in xs {
        for j in 1..d {
            mu[j] += e.x[j];
        }
    }
    for m in mu.iter_mut() {
        *m /= n;
    }
    for e in xs {
        for j in 1..d {
            sd[j] += (e.x[j] - mu[j]).powi(2);
        }
    }
    // Features (nearly) constant in this group get weight 0: standardising them would blow
    // their weights up
    let active: Vec<usize> = (1..d).filter(|&j| (sd[j] / n).sqrt() > 1e-2).collect();
    for s in sd.iter_mut() {
        *s = (*s / n).sqrt();
    }
    let m = active.len() + 1;
    let z = |e: &Example| -> Vec<f64> {
        let mut v = Vec::with_capacity(m);
        v.push(1.0);
        for &j in &active {
            v.push((e.x[j] - mu[j]) / sd[j]);
        }
        v
    };
    let rows: Vec<(Vec<f64>, f64, f64)> = xs
        .iter()
        .map(|e| {
            (
                z(e),
                if e.y { 1.0 } else { 0.0 },
                if e.y { w_pos } else { w_neg },
            )
        })
        .collect();
    let mut beta = vec![0.0; m];
    for _ in 0..60 {
        let mut g = vec![0.0; m];
        let mut h = vec![vec![0.0; m]; m];
        for (v, y, w) in &rows {
            let t: f64 = v.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let p = 1.0 / (1.0 + (-t).exp());
            let r = w * (p - y);
            let q = w * p * (1.0 - p);
            for i in 0..m {
                g[i] += r * v[i];
                for k in 0..=i {
                    h[i][k] += q * v[i] * v[k];
                }
            }
        }
        for i in 1..m {
            g[i] += l2 * beta[i];
            h[i][i] += l2;
        }
        h[0][0] += 1e-9;
        for i in 0..m {
            for k in i + 1..m {
                h[i][k] = h[k][i];
            }
        }
        let Some(step) = solve(h, g) else { break };
        let mut norm = 0.0;
        for i in 0..m {
            beta[i] -= step[i];
            norm += step[i] * step[i];
        }
        if norm.sqrt() < 1e-7 {
            break;
        }
    }
    let mut w = vec![0.0; d];
    w[0] = beta[0];
    for (k, &j) in active.iter().enumerate() {
        w[j] = beta[k + 1] / sd[j];
        w[0] -= beta[k + 1] * mu[j] / sd[j];
    }
    w
}

/// The junk false-positive rates each sensitivity targets (High, Medium, Low), by length
/// in characters. Short text is stricter: a wrong Caesar shift of a short word is often
/// another word. The rates are below both baselines' (gibberish-or-not 5.0.7 and Ciphey's
/// English checker) in each of the report's length buckets of corpus v1's dev split, which
/// is what 6.0 must beat (docs/benchmark.md).
pub fn targets(chars_lo: u32) -> [f64; 3] {
    match chars_lo {
        0..=9 => [0.001, 0.0008, 0.0002],
        10..=19 => [0.004, 0.002, 0.001],
        20..=49 => [0.012, 0.005, 0.003],
        50..=199 => [0.015, 0.005, 0.002],
        _ => [0.015, 0.005, 0.0],
    }
}

/// The gate sits this far (in rank units: log10 per letter) below the lowest positive.
pub const GATE_MARGIN: f32 = 0.3;

/// Sensitivity targets: a multiplier on [`targets`] (1 for the shipped calibration).
#[derive(Clone, Copy, Debug)]
pub struct Targets {
    pub scale: f64,
}

/// The logit at which at most `fpr` of `negatives` (sorted ascending, rejected ones
/// excluded but counted in `total`) are at or above it.
fn threshold(sorted_neg: &[f64], total: usize, fpr: f64) -> f64 {
    let allowed = (fpr * total as f64).floor() as usize;
    if sorted_neg.is_empty() || allowed >= sorted_neg.len() {
        return f64::NEG_INFINITY;
    }
    // Strictly above the (allowed+1)-th highest
    let k = sorted_neg.len() - 1 - allowed;
    let v = sorted_neg[k];
    v + 1e-4 * v.abs().max(1.0)
}

pub struct FitReport {
    pub calibration: String,
    pub lines: Vec<String>,
}

/// Fits `lang`'s pack: weights on `train`, anchors on `dev`.
pub fn fit(
    pack: Pack,
    train: &[Sample],
    dev: &[Sample],
    targets: Targets,
    l2: f64,
) -> Result<FitReport, String> {
    let lang = pack.lang();
    let names = Detector::feature_names();
    let detector = Detector::empty().with_pack(pack);
    let tr = examples(&detector, lang, train);
    let dv = examples(&detector, lang, dev);
    let mut lines = Vec::new();
    let mut cal = String::new();
    cal.push_str("gon-calibration 1\n");
    cal.push_str(&format!("lang {}\n", lang.code()));
    cal.push_str("features");
    for f in names {
        cal.push(' ');
        cal.push_str(f);
    }
    cal.push('\n');
    cal.push_str(&format!(
        "# written by `gib-bench fit --lang {}`: weights from the train split (l2 {l2}), anchors from the dev split at the junk FPR targets of bench/src/fit.rs (x{})\n",
        lang.code(),
        targets.scale
    ));
    // The gate: below every train and dev positive's rank, with a margin
    let min_pos = tr
        .iter()
        .chain(dv.iter())
        .filter_map(|(e, _)| e.as_ref())
        .filter(|e| e.y)
        .map(|e| e.rank)
        .fold(f32::INFINITY, f32::min);
    let gate = if min_pos.is_finite() {
        min_pos - GATE_MARGIN
    } else {
        f32::NEG_INFINITY
    };
    let gated_neg = dv
        .iter()
        .filter_map(|(e, _)| e.as_ref())
        .filter(|e| !e.y && e.rank < gate)
        .count();
    let dev_neg = dv.iter().filter(|(_, y)| !y).count();
    lines.push(format!(
        "gate {gate:.3}: rejects {gated_neg} of {dev_neg} dev negatives without the word model"
    ));
    cal.push_str(&format!("gate {gate}\n"));
    let mut prev_anchors: Option<[f64; 5]> = None;
    for (b, &lo) in BUCKETS.iter().enumerate() {
        let hi = BUCKETS.get(b + 1).map(|n| n - 1);
        let in_b = |e: &Example| bucket_of(e.chars) == b;
        let group = group_of(lo);
        let train_b: Vec<&Example> = tr
            .iter()
            .filter_map(|(e, _)| e.as_ref())
            .filter(|e| in_b(e))
            .collect();
        let train_g: Vec<&Example> = tr
            .iter()
            .filter_map(|(e, _)| e.as_ref())
            .filter(|e| group_of(e.chars) == group)
            .collect();
        let w = if train_g.is_empty() {
            vec![0.0; names.len()]
        } else {
            logistic(&train_g, l2)
        };
        let logit = |e: &Example| -> f64 { e.x.iter().zip(&w).map(|(a, b)| a * b).sum() };
        // Dev junk of this bucket: the applicable ones by logit; the rest count as rejected.
        // A negative the pack doesn't apply to has no letter count: it is spread over the
        // buckets by its share of all negatives (counted in every bucket's total in
        // proportion), which we approximate by counting it in no bucket's numerator and
        // in the total of the bucket its text length would fall in (by characters).
        let mut neg: Vec<f64> = Vec::new();
        let mut pos: Vec<f64> = Vec::new();
        let mut neg_total = 0usize;
        for (e, y) in &dv {
            match e {
                Some(e) if in_b(e) && e.rank < gate => {
                    if !*y {
                        neg_total += 1;
                    }
                }
                Some(e) if in_b(e) => {
                    if *y {
                        pos.push(logit(e));
                    } else {
                        neg.push(logit(e));
                        neg_total += 1;
                    }
                }
                _ => {}
            }
        }
        neg.sort_by(f64::total_cmp);
        pos.sort_by(f64::total_cmp);
        let mut a = [0.0f64; 5];
        // High, Medium, Low thresholds
        for (i, f) in self::targets(lo).iter().enumerate() {
            a[i + 1] = threshold(&neg, neg_total, f * targets.scale);
        }
        // Without enough junk in the bucket, fall back to the previous bucket's anchors
        // relative position, or to the positives' spread
        let finite = a[1..4].iter().all(|v| v.is_finite());
        if !finite {
            let pos_med = pos.get(pos.len() / 2).copied().unwrap_or(0.0);
            let base = prev_anchors.map(|p| p[1]).unwrap_or(pos_med - 3.0);
            for i in 1..4 {
                if !a[i].is_finite() {
                    a[i] = base + (i as f64 - 1.0);
                }
            }
        }
        // Monotone: High <= Medium <= Low
        a[2] = a[2].max(a[1]);
        a[3] = a[3].max(a[2]);
        let spread = (a[3] - a[1]).max(1.0);
        let neg_med = neg.get(neg.len() / 2).copied().unwrap_or(a[1] - spread);
        a[0] = (a[1] - spread).min(neg_med).min(a[1] - 0.5);
        let pos_hi = pos
            .get(pos.len() * 9 / 10)
            .copied()
            .unwrap_or(a[3] + spread);
        a[4] = (a[3] + spread).max(pos_hi).max(a[3] + 0.5);
        prev_anchors = Some(a);
        let recall = |t: f64| {
            if pos.is_empty() {
                f64::NAN
            } else {
                pos.iter().filter(|&&z| z >= t).count() as f64 / pos.len() as f64
            }
        };
        let fpr = |t: f64| {
            if neg_total == 0 {
                f64::NAN
            } else {
                neg.iter().filter(|&&z| z >= t).count() as f64 / neg_total as f64
            }
        };
        lines.push(format!(
            "bucket {lo}-{}: train {} pos / {} neg; dev {} pos / {} neg; dev recall H/M/L {:.3}/{:.3}/{:.3}, FPR {:.4}/{:.4}/{:.4}",
            hi.map_or("inf".into(), |h| h.to_string()),
            train_b.iter().filter(|e| e.y).count(),
            train_b.iter().filter(|e| !e.y).count(),
            pos.len(),
            neg_total,
            recall(a[1]),
            recall(a[2]),
            recall(a[3]),
            fpr(a[1]),
            fpr(a[2]),
            fpr(a[3]),
        ));
        cal.push_str(&format!(
            "bucket {lo} {}",
            hi.map_or("inf".into(), |h| h.to_string())
        ));
        for v in &w {
            cal.push_str(&format!(" {}", *v as f32));
        }
        cal.push_str(" |");
        for v in a {
            cal.push_str(&format!(" {}", v as f32));
        }
        cal.push('\n');
    }
    Ok(FitReport {
        calibration: cal,
        lines,
    })
}
