//! `gib-bench fit --lang <code>`: fits a pack's classifier on the train split and its score
//! anchors on the dev split, and writes `languages/<code>/calibration.txt`.
//!
//! Per length bucket (characters, identifiers left out): an L2-regularised, class-balanced
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
    /// The pack's pre-rank (texts of 12+ letters, dense models; else +inf).
    pub pre: f32,
}

/// The features of every positive of `lang` and every negative, for `lang`'s pack.
/// Samples the pack doesn't apply to (vetoed, or no letters in its scripts) are counted as
/// rejected: `None`.
pub fn examples(
    detector: &Detector,
    lang: Lang,
    samples: &[Sample],
) -> Vec<(Option<Example>, bool, bool)> {
    let mut out = Vec::new();
    for s in samples {
        let y = match s.label {
            Label::Pos if s.lang == lang.code() => true,
            Label::Neg => false,
            _ => continue,
        };
        let feats = detector.features(&s.text);
        let own = !y && s.lang == lang.code();
        let ex =
            feats
                .into_iter()
                .find(|(l, _, _, _, _)| *l == lang)
                .map(|(_, x, chars, rank, pre)| Example {
                    x: x.iter().map(|&v| f64::from(v)).collect(),
                    y,
                    chars,
                    rank,
                    pre,
                });
        out.push((ex, y, own));
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
/// is what 6.0 must beat (docs/benchmark.md). Under 10 characters the rate is for the
/// four short buckets together (`allocate`).
pub fn targets(chars_lo: u32) -> [f64; 3] {
    match chars_lo {
        0..=9 => [0.001, 0.0008, 0.0003],
        10..=19 => [0.004, 0.002, 0.001],
        20..=49 => [0.012, 0.005, 0.003],
        50..=199 => [0.015, 0.005, 0.002],
        _ => [0.015, 0.005, 0.0],
    }
}

/// Buckets of texts shorter than this many characters share one false-positive budget per
/// sensitivity (`allocate`), the target of the 1-9 character report bucket.
pub const SHORT_CHARS: u32 = 10;

/// A bucket's own-language junk is used for the thresholds when it has at least this many
/// samples (fewer can't estimate the targets).
pub const OWN_JUNK_MIN: usize = 500;

/// The gate sits this far (in rank units: log10 per letter) below the lowest positive.
pub const GATE_MARGIN: f32 = 0.3;

/// In budget mode, texts of 12 or more letters whose pre-rank is this far below the lowest
/// such train or dev positive's are rejected before the full n-gram pass (`pregate`).
pub const PREGATE_MARGIN: f32 = 0.3;

/// In budget mode a length bucket with at least this many train and dev positives gets an
/// n-gram gate of its own: below its lowest positive, with the margin. Longer texts fit a
/// language better, so the one gate of all lengths (set by the shortest texts) lets most
/// junk of 20+ characters through to the word model.
pub const BUCKET_GATE_MIN: usize = 20;

/// How a pack's thresholds are set (docs/languages.md, "Calibration").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Each length bucket at its own junk false-positive target ([`targets`]). For the
    /// base pack, English, which defines those targets.
    Targets,
    /// One false-positive budget per sensitivity, gate 2's allowance ([`GATE2_RISE`]), spent
    /// over all length buckets where it buys the most recall, counting only the junk that
    /// the packs already built in don't accept, and capped per bucket below the CI gate's
    /// ceilings. For every pack after the base one.
    Budget,
}

/// Options of a fit: the mode, and a multiplier on its targets (Targets) or on its budget
/// (Budget); 1 for a shipped calibration unless the pack's README says why not.
#[derive(Clone, Copy, Debug)]
pub struct Targets {
    pub scale: f64,
    pub mode: Mode,
}

/// Gate 2 (docs/languages.md): the rise of the shared junk false-positive rate one pack may
/// cause, as a share of all dev negatives, at High, Medium and Low.
pub const GATE2_RISE: [f64; 3] = [0.0005, 0.0002, 0.0001];

/// In budget mode a pack may spend in any one length bucket up to this many times its
/// proportional share of gate 2's allowance there: `GATE2_RISE` times the bucket's dev
/// negatives (all of them, as the CI gate counts). Under 10 characters that is up to 70% of
/// the budget, where short phrases need it; and the CI gate's ceilings grow by the same
/// amount per pack ([`fpr_ceiling`]), so packs that keep to their caps fit in any order.
pub const BUCKET_CONCENTRATION: f64 = 4.0;

/// The CI gate's per-bucket junk FPR ceiling (bench/src/main.rs `cmd_gate`):
/// `FPR_CEILING_FACTOR * target + FPR_CEILING_SLACK` with the base pack alone ([`fpr_ceiling`]).
pub const FPR_CEILING_FACTOR: f64 = 1.5;
pub const FPR_CEILING_SLACK: f64 = 0.0002;

/// The CI gate's junk FPR ceiling for the bucket starting at `chars_lo` characters, at
/// sensitivity `i` (0 High, 1 Medium, 2 Low, as [`targets`]), with `packs` packs built in:
/// the base pack's targets with some slack, plus each further pack's largest allowed spend
/// in one bucket (`BUCKET_CONCENTRATION` times its share of gate 2's allowance).
pub fn fpr_ceiling(chars_lo: u32, i: usize, packs: usize) -> f64 {
    FPR_CEILING_FACTOR * targets(chars_lo)[i]
        + FPR_CEILING_SLACK
        + packs.saturating_sub(1) as f64 * BUCKET_CONCENTRATION * GATE2_RISE[i]
}

/// Negatives above a threshold below which the empirical quantile is used; with fewer, the
/// tail is modelled.
pub const TAIL_SAMPLES: usize = 25;

/// The logit above which a share `fpr` of the negatives lies. `sorted_neg` holds the
/// logits of the negatives the pack scores, ascending; `total` counts the gated ones too.
///
/// A zero target is just above the highest negative. With at least [`TAIL_SAMPLES`]
/// negatives expected above it, the empirical quantile.
/// With fewer (short buckets, strict targets) the empirical quantile is one or two samples
/// and doesn't generalise (a split-half check on dev gave 5x the target), so the tail is
/// modelled instead: the excesses of the top [`TAIL_SAMPLES`] logits over the next one are
/// taken as exponential (peaks over threshold), and the threshold is where that tail
/// reaches `fpr`.
fn threshold(sorted_neg: &[f64], total: usize, fpr: f64) -> f64 {
    let n = sorted_neg.len();
    if n == 0 || total == 0 {
        return f64::NEG_INFINITY;
    }
    let expected = fpr * total as f64;
    // A zero target: just above the highest negative
    if fpr <= 0.0 {
        let v = sorted_neg[n - 1];
        return v + 1e-4 * v.abs().max(1.0);
    }
    if expected >= TAIL_SAMPLES as f64 || n <= TAIL_SAMPLES + 1 {
        let allowed = expected.floor() as usize;
        if allowed >= n {
            return f64::NEG_INFINITY;
        }
        let v = sorted_neg[n - 1 - allowed];
        return v + 1e-4 * v.abs().max(1.0);
    }
    let m = TAIL_SAMPLES;
    let u = sorted_neg[n - 1 - m];
    let beta = (sorted_neg[n - m..].iter().map(|z| z - u).sum::<f64>() / m as f64).max(0.05);
    let rate = expected.max(1e-3);
    u + beta * (m as f64 / rate).ln()
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Expected number of `sorted_neg` (ascending) at or above `t`: the count, or, where fewer
/// than [`TAIL_SAMPLES`] are, the exponential tail fitted to the top ones (see
/// [`threshold`]).
fn tail_count(sorted_neg: &[f64], t: f64) -> f64 {
    let n = sorted_neg.len();
    let above = n - sorted_neg.partition_point(|&z| z < t);
    if above >= TAIL_SAMPLES || n <= TAIL_SAMPLES + 1 {
        return above as f64;
    }
    let m = TAIL_SAMPLES;
    let u = sorted_neg[n - 1 - m];
    let beta = (sorted_neg[n - m..].iter().map(|z| z - u).sum::<f64>() / m as f64).max(0.05);
    m as f64 * (-(t - u) / beta).exp()
}

/// Thresholds for several short buckets that share one false-positive budget: `budget`
/// expected false positives over all of them, spent where it buys the most recall (greedy
/// on the upper convex hull of each bucket's recall against its tail-estimated false
/// positives). Real short words and junk that happens to be a word overlap in the
/// shortest buckets, so an equal target per bucket would reject them all there.
fn allocate(buckets: &[(Vec<f64>, Vec<f64>)], budget: f64) -> Vec<f64> {
    allocate_with(buckets, budget, None, false)
}

/// Expected false positives at or above `t`: in conservative mode the larger of the
/// observed count and the tail model, so that the budget is never exceeded on dev.
fn cost(sorted_neg: &[f64], t: f64, conservative: bool) -> f64 {
    let modelled = tail_count(sorted_neg, t);
    if conservative {
        let above = (sorted_neg.len() - sorted_neg.partition_point(|&z| z < t)) as f64;
        modelled.max(above)
    } else {
        modelled
    }
}

/// [`allocate`] with optional per-bucket caps on the false positives **seen on dev** (what
/// the CI gate counts) and the conservative cost. A cap under one false positive (Low at
/// 200+ characters) still lets a bucket accept the positives above all its negatives.
fn allocate_with(
    buckets: &[(Vec<f64>, Vec<f64>)],
    budget: f64,
    caps: Option<&[f64]>,
    conservative: bool,
) -> Vec<f64> {
    // Per bucket: the k-th step accepts the k best positives: (threshold, cost, seen)
    let steps: Vec<Vec<(f64, f64, usize)>> = buckets
        .iter()
        .map(|(pos, neg)| {
            let mut p: Vec<f64> = pos.clone();
            p.sort_by(|a, b| b.total_cmp(a));
            let mut v = vec![(f64::INFINITY, 0.0, 0usize)];
            for (k, &t) in p.iter().enumerate() {
                // ties: only the last of equal logits is a step
                if p.get(k + 1).is_some_and(|&next| next == t) {
                    continue;
                }
                let seen = neg.len() - neg.partition_point(|&z| z < t);
                v.push((t, cost(neg, t, conservative), seen));
            }
            v
        })
        .collect();
    // The last step each bucket may take under its cap
    let last: Vec<usize> = steps
        .iter()
        .enumerate()
        .map(|(b, v)| match caps {
            Some(c) => v
                .iter()
                .rposition(|&(_, _, seen)| seen as f64 <= c[b])
                .unwrap_or(0),
            None => v.len() - 1,
        })
        .collect();
    let mut at = vec![0usize; buckets.len()];
    let mut spent = 0.0;
    loop {
        // The best next step of each bucket: the most positives per false positive
        let mut best: Option<(f64, usize, usize, f64)> = None; // ratio, bucket, step, cost
        for (b, v) in steps.iter().enumerate() {
            let (k0, fp0) = (at[b], v[at[b]].1);
            let pos_at = |k: usize| -> usize {
                // positives accepted at step k: count of logits >= v[k].0
                let (pos, _) = &buckets[b];
                pos.iter().filter(|&&z| z >= v[k].0).count()
            };
            let p0 = pos_at(k0);
            for k in k0 + 1..=last[b] {
                let cost = (v[k].1 - fp0).max(1e-9);
                // Only steps the rest of the budget pays for: a dear step in one bucket
                // doesn't stop the cheaper ones of the others
                if spent + cost > budget {
                    continue;
                }
                let ratio = (pos_at(k) - p0) as f64 / cost;
                if best.is_none_or(|(r, _, _, _)| ratio > r) {
                    best = Some((ratio, b, k, cost));
                }
            }
        }
        let Some((_, b, k, cost)) = best else { break };
        spent += cost;
        at[b] = k;
    }
    at.iter()
        .zip(&steps)
        .zip(buckets)
        .map(|((&k, v), (pos, neg))| {
            if k == 0 {
                // Nothing accepted: above every positive and negative
                let top = pos
                    .iter()
                    .chain(neg.iter())
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if top.is_finite() {
                    top + 1.0
                } else {
                    0.0
                }
            } else {
                v[k].0 - 1e-6
            }
        })
        .collect()
}

pub struct FitReport {
    pub calibration: String,
    pub lines: Vec<String>,
}

/// Fits `lang`'s pack: weights on `train`, anchors on `dev`.
/// One bucket's weights and dev logits (see `fit`).
struct Dev {
    w: Vec<f64>,
    pos: Vec<f64>,
    neg: Vec<f64>,
    own: Vec<f64>,
    neg_total: usize,
    own_total: usize,
    /// The dev negatives in two halves (by sample order; gated ones as -inf)
    halves: [Vec<f64>; 2],
    train_pos: usize,
    train_neg: usize,
    /// The bucket's n-gram gate: the calibration's, or the bucket's own if higher.
    gate: f32,
    /// The calibration's pre-gate (`-inf`: none).
    pregate: f32,
}

/// Mode::Targets: each bucket at its own targets; the short buckets (under SHORT_CHARS
/// characters) share one budget per sensitivity, spent where it buys the most recall
/// (`allocate`).
fn targets_thresholds(
    dev: &[Dev],
    short: &[usize],
    targets: Targets,
    thresholds: &mut [[f64; 3]],
    lines: &mut Vec<String>,
) {
    for (b, &lo) in BUCKETS.iter().enumerate() {
        if short.contains(&b) {
            continue;
        }
        let d = &dev[b];
        for (i, f) in self::targets(lo).iter().enumerate() {
            let t_all = threshold(&d.neg, d.neg_total, f * targets.scale);
            let t_own = if i == 2 && d.own_total >= OWN_JUNK_MIN {
                threshold(&d.own, d.own_total, f * targets.scale)
            } else {
                f64::NEG_INFINITY
            };
            thresholds[b][i] = t_all.max(t_own);
        }
    }
    if !short.is_empty() {
        let f = self::targets(1);
        let total: usize = short.iter().map(|&b| dev[b].neg_total).sum();
        for (i, f) in f.iter().enumerate() {
            let pairs: Vec<(Vec<f64>, Vec<f64>)> = short
                .iter()
                .map(|&b| (dev[b].pos.clone(), dev[b].neg.clone()))
                .collect();
            let t_all = allocate(&pairs, f * targets.scale * total as f64);
            // No own-language constraint here: at these rates the own junk of one bucket
            // is a handful of samples, and the tail estimate of each is already
            // conservative (split-half FPR below the target on dev)
            for (k, &b) in short.iter().enumerate() {
                thresholds[b][i] = t_all[k];
            }
        }
        // Split-half check of the short group: allocate on one half of the negatives,
        // count false positives on the other
        let mut fp = [0usize; 3];
        let mut n = 0usize;
        for h in 0..2 {
            let half_total: usize = short.iter().map(|&b| dev[b].halves[h].len()).sum();
            n += short
                .iter()
                .map(|&b| dev[b].halves[1 - h].len())
                .sum::<usize>();
            for (i, f) in self::targets(1).iter().enumerate() {
                let pairs: Vec<(Vec<f64>, Vec<f64>)> = short
                    .iter()
                    .map(|&b| {
                        let mut neg: Vec<f64> = dev[b].halves[h]
                            .iter()
                            .copied()
                            .filter(|z| z.is_finite())
                            .collect();
                        neg.sort_by(f64::total_cmp);
                        (dev[b].pos.clone(), neg)
                    })
                    .collect();
                let t = allocate(&pairs, f * targets.scale * half_total as f64);
                for (k, &b) in short.iter().enumerate() {
                    fp[2 - i] += dev[b].halves[1 - h].iter().filter(|&&z| z >= t[k]).count();
                }
            }
        }
        let tg = self::targets(1);
        lines.push(format!(
            "buckets 1-{}: split-half FPR L/M/H {:.4}/{:.4}/{:.4} (targets {:.4}/{:.4}/{:.4}, shared)",
            SHORT_CHARS - 1,
            ratio(fp[0], n),
            ratio(fp[1], n),
            ratio(fp[2], n),
            tg[2] * targets.scale,
            tg[1] * targets.scale,
            tg[0] * targets.scale,
        ));
    }
}

/// The packs built in, without `lang`'s: what a new pack's junk is measured against.
fn detector_others(lang: Lang) -> Detector {
    let others: Vec<Lang> = Detector::new().enabled().filter(|&l| l != lang).collect();
    Detector::new().languages(&others)
}

/// Mode::Budget: per sensitivity, gate 2's allowance of new false positives on dev
/// (`GATE2_RISE` of all dev negatives, times `targets.scale`), spent over all buckets where
/// it buys the most recall. Negatives the other built-in packs already accept at that
/// sensitivity cost nothing (they are false positives already). Each bucket may spend at
/// most its Targets-mode allowance, and costs are the larger of the observed count and the
/// tail model, so the budget holds on dev.
#[allow(clippy::too_many_arguments)]
fn budget_thresholds(
    others: &Detector,
    lang: Lang,
    dev_samples: &[Sample],
    dv: &[(Option<Example>, bool, bool)],
    dev: &[Dev],
    targets: Targets,
    thresholds: &mut [[f64; 3]],
    lines: &mut Vec<String>,
) {
    // The other packs' verdicts on the same samples, in the order of `examples`
    let mut taken: Vec<[bool; 3]> = Vec::new();
    for s in dev_samples {
        match s.label {
            Label::Pos if s.lang == lang.code() => {}
            Label::Neg => {}
            _ => continue,
        }
        let score = others.detect(&s.text).score;
        // High, Medium, Low (the order of `targets`)
        taken.push([score >= 0.5, score >= 0.75, score >= 0.9]);
    }
    assert_eq!(taken.len(), dv.len());
    let n_all = dv.iter().filter(|(_, y, _)| !y).count();
    // Every dev negative per bucket, whether this pack scores it or not: what the CI gate's
    // ceilings count (a pack for one script scores a small share of them)
    let mut bucket_neg = vec![0usize; BUCKETS.len()];
    for s in dev_samples.iter().filter(|s| s.label == Label::Neg) {
        bucket_neg[bucket_of(s.chars() as u32)] += 1;
    }
    for i in 0..3 {
        let budget = GATE2_RISE[i] * n_all as f64 * targets.scale;
        // Per bucket: positives, and the negatives not already taken by the other packs
        let mut pairs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        let mut caps: Vec<f64> = Vec::new();
        for b in 0..BUCKETS.len() {
            let w = &dev[b].w;
            let in_b = |e: &&Example| bucket_of(e.chars) == b;
            let mut neg: Vec<f64> = dv
                .iter()
                .zip(&taken)
                .filter(|((e, y, _), t)| !*y && !t[i] && e.as_ref().is_some_and(|e| in_b(&e)))
                .filter_map(|((e, _, _), _)| e.as_ref())
                .filter(|e| e.rank >= dev[b].gate && e.pre >= dev[b].pregate)
                .map(|e| e.x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            neg.sort_by(f64::total_cmp);
            pairs.push((dev[b].pos.clone(), neg));
            caps.push(
                BUCKET_CONCENTRATION
                    * GATE2_RISE[i]
                    * targets.scale.max(1.0)
                    * bucket_neg[b] as f64,
            );
        }
        let t = allocate_with(&pairs, budget, Some(&caps), true);
        let mut new_fp = 0usize;
        for (b, tb) in t.iter().enumerate() {
            thresholds[b][i] = *tb;
            new_fp += pairs[b].1.iter().filter(|&&z| z >= *tb).count();
        }
        // Split-half: allocate on one half of the negatives, count on the other
        let mut half_fp = 0usize;
        for h in 0..2 {
            let halves: Vec<(Vec<f64>, Vec<f64>)> = pairs
                .iter()
                .map(|(pos, neg)| {
                    (
                        pos.clone(),
                        neg.iter().copied().skip(h).step_by(2).collect(),
                    )
                })
                .collect();
            let th = allocate_with(
                &halves,
                budget / 2.0,
                Some(&caps.iter().map(|c| c / 2.0).collect::<Vec<_>>()),
                true,
            );
            for (b, tb) in th.iter().enumerate() {
                half_fp += pairs[b]
                    .1
                    .iter()
                    .skip(1 - h)
                    .step_by(2)
                    .filter(|&&z| z >= *tb)
                    .count();
            }
        }
        let accepted: usize = pairs
            .iter()
            .zip(&t)
            .map(|((pos, _), tb)| pos.iter().filter(|&&z| z >= *tb).count())
            .sum();
        let total: usize = pairs.iter().map(|(pos, _)| pos.len()).sum();
        lines.push(format!(
            "budget {}: {new_fp} new dev false positives for a budget of {budget:.1} (split-half {half_fp}); dev recall of the pack {:.3}",
            ["High", "Medium", "Low"][i],
            ratio(accepted, total),
        ));
    }
}

/// Budget mode's extra gates, from the train and dev positives (`examples`), given the
/// calibration's gate: the pre-gate (below every positive of 12+ letters' pre-rank, by
/// [`PREGATE_MARGIN`]), and a gate per length bucket with [`BUCKET_GATE_MIN`] positives
/// (below its lowest positive's rank, by [`GATE_MARGIN`]; `-inf` where that isn't above
/// `gate`).
pub fn budget_gates(
    tr: &[(Option<Example>, bool, bool)],
    dv: &[(Option<Example>, bool, bool)],
    gate: f32,
) -> (f32, Vec<f32>) {
    let positives = || {
        tr.iter()
            .chain(dv.iter())
            .filter_map(|(e, _, _)| e.as_ref())
            .filter(|e| e.y)
    };
    let bucket_gates = (0..BUCKETS.len())
        .map(|b| {
            let ranks: Vec<f32> = positives()
                .filter(|e| bucket_of(e.chars) == b)
                .map(|e| e.rank)
                .collect();
            if ranks.len() < BUCKET_GATE_MIN {
                return f32::NEG_INFINITY;
            }
            let low = ranks.iter().copied().fold(f32::INFINITY, f32::min) - GATE_MARGIN;
            if low > gate {
                low
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let long_pre: Vec<f32> = positives()
        .filter(|e| e.pre.is_finite())
        .map(|e| e.pre)
        .collect();
    let pregate = if long_pre.len() >= BUCKET_GATE_MIN {
        long_pre.iter().copied().fold(f32::INFINITY, f32::min) - PREGATE_MARGIN
    } else {
        f32::NEG_INFINITY
    };
    (pregate, bucket_gates)
}

/// `cal` (a calibration's text) with budget mode's gates added: `pregate`, and each bucket
/// line's own gate.
pub fn with_gates(cal: &str, pregate: f32, bucket_gates: &[f32]) -> String {
    let mut out = String::new();
    let mut b = 0usize;
    for line in cal.lines() {
        out.push_str(line);
        if line.starts_with("bucket ") {
            if let Some(g) = bucket_gates.get(b).filter(|g| g.is_finite()) {
                out.push_str(&format!(" | {g}"));
            }
            b += 1;
        }
        out.push('\n');
        if line.starts_with("gate ") && pregate.is_finite() {
            out.push_str(&format!("pregate {pregate}\n"));
        }
    }
    out
}

/// The calibration's gate (below every train and dev positive's rank, by `GATE_MARGIN`).
pub fn base_gate(
    tr: &[(Option<Example>, bool, bool)],
    dv: &[(Option<Example>, bool, bool)],
) -> f32 {
    let min_pos = tr
        .iter()
        .chain(dv.iter())
        .filter_map(|(e, _, _)| e.as_ref())
        .filter(|e| e.y)
        .map(|e| e.rank)
        .fold(f32::INFINITY, f32::min);
    if min_pos.is_finite() {
        min_pos - GATE_MARGIN
    } else {
        f32::NEG_INFINITY
    }
}

pub fn fit(
    pack: Pack,
    train: &[Sample],
    dev_samples: &[Sample],
    targets: Targets,
    l2: f64,
) -> Result<FitReport, String> {
    let lang = pack.lang();
    let names = Detector::feature_names();
    let detector = Detector::empty().with_pack(pack);
    let tr = examples(&detector, lang, train);
    let dv = examples(&detector, lang, dev_samples);
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
        "# written by `gib-bench fit --lang {}`: weights from the train split (l2 {l2}), anchors from the dev split, mode {:?} x{}\n",
        lang.code(),
        targets.mode,
        targets.scale
    ));
    // The gate: below every train and dev positive's rank, with a margin
    let gate = base_gate(&tr, &dv);
    let gated_neg = dv
        .iter()
        .filter_map(|(e, _, _)| e.as_ref())
        .filter(|e| !e.y && e.rank < gate)
        .count();
    let dev_neg = dv.iter().filter(|(_, y, _)| !y).count();
    lines.push(format!(
        "gate {gate:.3}: rejects {gated_neg} of {dev_neg} dev negatives without the word model"
    ));
    cal.push_str(&format!("gate {gate}\n"));
    let (pregate, bucket_gates) = if targets.mode == Mode::Budget {
        budget_gates(&tr, &dv, gate)
    } else {
        (f32::NEG_INFINITY, vec![f32::NEG_INFINITY; BUCKETS.len()])
    };
    if pregate.is_finite() {
        cal.push_str(&format!("pregate {pregate}\n"));
        let n = dv
            .iter()
            .filter_map(|(e, _, _)| e.as_ref())
            .filter(|e| !e.y && e.pre < pregate)
            .count();
        lines.push(format!(
            "pregate {pregate:.3}: rejects {n} dev negatives before the full n-gram pass"
        ));
    }
    if bucket_gates.iter().any(|g| g.is_finite()) {
        let more = dv
            .iter()
            .filter_map(|(e, _, _)| e.as_ref())
            .filter(|e| !e.y && e.rank >= gate && e.rank < bucket_gates[bucket_of(e.chars)])
            .count();
        lines.push(format!(
            "bucket gates: reject {more} more dev negatives without the word model"
        ));
    }
    // Pass 1: weights per group of buckets, and every bucket's dev logits
    let mut dev: Vec<Dev> = Vec::new();
    for (b, &lo) in BUCKETS.iter().enumerate() {
        let in_b = |e: &Example| bucket_of(e.chars) == b;
        let group = group_of(lo);
        let train_g: Vec<&Example> = tr
            .iter()
            .filter_map(|(e, _, _)| e.as_ref())
            .filter(|e| group_of(e.chars) == group)
            .collect();
        let w = if train_g.is_empty() {
            vec![0.0; names.len()]
        } else {
            logistic(&train_g, l2)
        };
        let logit = |e: &Example| -> f64 { e.x.iter().zip(&w).map(|(a, b)| a * b).sum() };
        let mut d = Dev {
            pos: Vec::new(),
            neg: Vec::new(),
            own: Vec::new(),
            neg_total: 0,
            own_total: 0,
            halves: [Vec::new(), Vec::new()],
            train_pos: tr
                .iter()
                .filter_map(|(e, _, _)| e.as_ref())
                .filter(|e| in_b(e) && e.y)
                .count(),
            train_neg: tr
                .iter()
                .filter_map(|(e, _, _)| e.as_ref())
                .filter(|e| in_b(e) && !e.y)
                .count(),
            w: Vec::new(),
            gate: gate.max(bucket_gates[b]),
            pregate,
        };
        // Negatives the pack doesn't apply to (no letters in its scripts, vetoed) are
        // rejected by it whatever the threshold, and counted in no bucket.
        let mut nth = 0usize;
        for (e, y, is_own) in &dv {
            let Some(e) = e else { continue };
            if !in_b(e) {
                continue;
            }
            let z = if e.rank < d.gate || e.pre < d.pregate {
                f64::NEG_INFINITY
            } else {
                logit(e)
            };
            if *y {
                if z.is_finite() {
                    d.pos.push(z);
                }
                continue;
            }
            d.neg_total += 1;
            d.own_total += usize::from(*is_own);
            d.halves[nth % 2].push(z);
            nth += 1;
            if z.is_finite() {
                d.neg.push(z);
                if *is_own {
                    d.own.push(z);
                }
            }
        }
        d.pos.sort_by(f64::total_cmp);
        d.neg.sort_by(f64::total_cmp);
        d.own.sort_by(f64::total_cmp);
        d.w = w;
        dev.push(d);
    }

    let mut thresholds = vec![[f64::NEG_INFINITY; 3]; BUCKETS.len()];
    let short: Vec<usize> = (0..BUCKETS.len())
        .filter(|&b| BUCKETS[b] < SHORT_CHARS)
        .collect();
    match targets.mode {
        Mode::Targets => {
            targets_thresholds(&dev, &short, targets, &mut thresholds, &mut lines);
        }
        Mode::Budget => {
            budget_thresholds(
                &detector_others(lang),
                lang,
                dev_samples,
                &dv,
                &dev,
                targets,
                &mut thresholds,
                &mut lines,
            );
        }
    }

    // Anchors, the split-half check of the other buckets, and the report
    let mut prev_anchors: Option<[f64; 5]> = None;
    for (b, &lo) in BUCKETS.iter().enumerate() {
        let hi = BUCKETS.get(b + 1).map(|n| n - 1);
        let d = &dev[b];
        let mut a = [0.0f64; 5];
        a[1..4].copy_from_slice(&thresholds[b]);
        // Without enough junk in the bucket, fall back to the previous bucket's anchors
        // relative position, or to the positives' spread
        if !a[1..4].iter().all(|v| v.is_finite()) {
            let pos_med = d.pos.get(d.pos.len() / 2).copied().unwrap_or(0.0);
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
        let neg_med = d.neg.get(d.neg.len() / 2).copied().unwrap_or(a[1] - spread);
        a[0] = (a[1] - spread).min(neg_med).min(a[1] - 0.5);
        let pos_hi = d
            .pos
            .get(d.pos.len() * 9 / 10)
            .copied()
            .unwrap_or(a[3] + spread);
        a[4] = (a[3] + spread).max(pos_hi).max(a[3] + 0.5);
        prev_anchors = Some(a);
        if !short.contains(&b) {
            let mut half_fp = [0usize; 3];
            let mut half_n = 0usize;
            for h in 0..2 {
                let mut fit_half: Vec<f64> = d.halves[h]
                    .iter()
                    .copied()
                    .filter(|z| z.is_finite())
                    .collect();
                fit_half.sort_by(f64::total_cmp);
                let other = &d.halves[1 - h];
                half_n += other.len();
                for (i, f) in self::targets(lo).iter().enumerate() {
                    let t = threshold(&fit_half, d.halves[h].len(), f * targets.scale);
                    half_fp[2 - i] += other.iter().filter(|&&z| z >= t).count();
                }
            }
            lines.push(format!(
                "bucket {lo}-{}: split-half FPR L/M/H {:.4}/{:.4}/{:.4} (targets {:.4}/{:.4}/{:.4})",
                hi.map_or("inf".into(), |h| h.to_string()),
                ratio(half_fp[0], half_n),
                ratio(half_fp[1], half_n),
                ratio(half_fp[2], half_n),
                self::targets(lo)[2] * targets.scale,
                self::targets(lo)[1] * targets.scale,
                self::targets(lo)[0] * targets.scale,
            ));
        }
        let recall = |t: f64| ratio(d.pos.iter().filter(|&&z| z >= t).count(), d.pos.len());
        let fpr = |t: f64| ratio(d.neg.iter().filter(|&&z| z >= t).count(), d.neg_total);
        lines.push(format!(
            "bucket {lo}-{}: train {} pos / {} neg; dev {} pos / {} neg; dev recall H/M/L {:.3}/{:.3}/{:.3}, FPR {:.4}/{:.4}/{:.4}",
            hi.map_or("inf".into(), |h| h.to_string()),
            d.train_pos,
            d.train_neg,
            d.pos.len(),
            d.neg_total,
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
        for v in &d.w {
            cal.push_str(&format!(" {}", *v as f32));
        }
        cal.push_str(" |");
        for v in a {
            cal.push_str(&format!(" {}", v as f32));
        }
        if bucket_gates[b].is_finite() {
            cal.push_str(&format!(" | {}", bucket_gates[b]));
        }
        cal.push('\n');
    }
    Ok(FitReport {
        calibration: cal,
        lines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 100 negatives from -10.0 to -0.1: enough that the tail model prices every threshold
    fn many_negatives() -> Vec<f64> {
        (1..=100).map(|i| -0.1 * i as f64).rev().collect()
    }

    #[test]
    fn a_cap_under_one_still_takes_the_positives_above_every_negative() {
        let buckets = vec![(vec![5.0, 6.0], many_negatives())];
        // The tail model prices t = 5 above 0, but no dev negative is that high
        assert!(cost(&buckets[0].1, 5.0, true) > 0.0);
        let t = allocate_with(&buckets, 1.0, Some(&[0.0]), true);
        assert!(t[0] <= 5.0, "threshold {} rejects the positives", t[0]);
    }

    #[test]
    fn caps_count_the_false_positives_seen_on_dev() {
        // Positives at 0.5 and 1.5; negatives at 1.0 and 2.0 (costs are counts: few negatives)
        let buckets = vec![(vec![0.5, 1.5, 2.5], vec![1.0, 2.0])];
        let t = allocate_with(&buckets, 10.0, Some(&[1.0]), true);
        let seen = buckets[0].1.iter().filter(|&&z| z >= t[0]).count();
        assert_eq!(seen, 1);
        assert!(t[0] <= 1.5 && t[0] > 1.0);
    }

    #[test]
    fn a_dear_step_does_not_stop_cheaper_ones() {
        // Bucket 0: 10 positives for 5 false positives (the best ratio, over the budget);
        // bucket 1: 1 positive for 1 false positive
        let buckets = vec![(vec![0.5; 10], vec![1.0; 5]), (vec![0.5], vec![1.0])];
        let t = allocate_with(&buckets, 3.0, None, false);
        assert!(t[0] > 1.0, "bucket 0 can't be afforded");
        assert!(t[1] <= 0.5, "bucket 1 fits the budget: threshold {}", t[1]);
    }

    #[test]
    fn ceilings_grow_with_each_further_pack() {
        for lo in BUCKETS {
            for i in 0..3 {
                let base = FPR_CEILING_FACTOR * targets(lo)[i] + FPR_CEILING_SLACK;
                assert_eq!(fpr_ceiling(lo, i, 1), base);
                let more = fpr_ceiling(lo, i, 3) - base;
                assert!((more - 2.0 * BUCKET_CONCENTRATION * GATE2_RISE[i]).abs() < 1e-12);
            }
        }
    }
}
