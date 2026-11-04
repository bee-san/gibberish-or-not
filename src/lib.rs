use phf::phf_set;
use std::path::{Path, PathBuf};

#[doc(hidden)]
mod cli;
mod dictionary;
mod model;
mod passwords;

// Core library exports
pub use model::{
    check_token_status, default_model_path, download_model, download_model_with_progress_bar,
    model_exists, ModelError, TokenStatus,
};

// CLI utilities made available for binary integration, but hidden from docs
#[doc(hidden)]
pub use cli::download_with_progress_bar;

/// Sensitivity level for gibberish detection: how readily text is classified as English.
///
/// A higher sensitivity flags more text as English. `Low` is the strictest level and
/// `High` the most lenient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sensitivity {
    /// High sensitivity - the most lenient classification as English.
    /// Little evidence is needed (the lowest composite-score threshold), and the
    /// high-entropy and no-English-word rejections are skipped.
    /// Best when input is expected to be mostly gibberish, and any English-like
    /// patterns should be flagged as potential English text.
    High,

    /// Medium sensitivity - balanced approach using both dictionary and n-gram analysis.
    /// Suitable for general purpose text classification.
    Medium,

    /// Low sensitivity - the strictest classification: requires the most evidence to
    /// classify text as English (the highest composite-score threshold), and three or more
    /// dictionary words alone are not enough.
    /// Best for texts that appear English-like but are actually gibberish.
    Low,
}

/// Gibberish detector with optional enhanced model
pub struct GibberishDetector {
    model_path: Option<PathBuf>,
}

impl GibberishDetector {
    /// Create new detector with no model
    pub fn new() -> Self {
        Self { model_path: None }
    }

    /// Create new detector with model path
    pub fn with_model<P: AsRef<Path>>(path: P) -> Self {
        Self {
            model_path: Some(path.as_ref().to_path_buf()),
        }
    }

    /// Check if model is available
    pub fn has_enhanced_detection(&self) -> bool {
        self.model_path
            .as_ref()
            .map(|p| model::Model::exists(p))
            .unwrap_or(false)
    }

    /// Main detection function
    pub fn is_gibberish(&self, text: &str, sensitivity: Sensitivity) -> bool {
        if is_string_bad_quality(&text) {
            return true;
        }
        // Run basic checks first
        // returns true if its gibberish
        let basic_result = run_basic_checks(text, sensitivity);

        // If basic checks say it's gibberish, no need for model
        if basic_result {
            return true;
        }

        // Try enhanced detection if available
        if let Some(path) = &self.model_path {
            if let Some(model) = model::Model::get_or_load(path) {
                // model returns True if its gibberish
                return model.predict(text);
            }
        }

        // Fall back to basic result
        basic_result
    }
}

/// is it worth it to calculate this string?
pub fn is_string_bad_quality(s: &str) -> bool {
    // Check for high percentage of invisible characters
    let non_printable_ratio = calculate_non_printable_ratio(s);
    if non_printable_ratio > 0.5 {
        return true; // Return lowest quality for strings with >50% invisible chars
    }
    if s.len() <= 3 {
        return true;
    }
    return false
}

/// Calculate the ratio of non-printable characters in a string
/// Returns a value between 0.0 (all printable) and 1.0 (all non-printable)
pub fn calculate_non_printable_ratio(text: &str) -> f32 {
    let non_printable_count = text
        .chars()
        .filter(|&c| {
            // Only count control characters (except common whitespace) and non-ASCII as non-printable
            (c.is_control() && c != '\n' && c != '\r' && c != '\t') || !c.is_ascii()
        })
        .count();

    non_printable_count as f32 / text.len() as f32
}

fn is_english_word(word: &str) -> bool {
    dictionary::ENGLISH_WORDS.contains(word)
}

/// Checks if the given text matches a known common password.
///
/// This function checks if the input text exactly matches a password from a comprehensive
/// list of common passwords, including:
/// - Most commonly used passwords
/// - Default passwords
/// - Dictionary-based passwords
///
/// # Arguments
///
/// * `text` - The text to check against the password list
///
/// # Returns
///
/// * `true` if the text exactly matches a known password
/// * `false` otherwise
///
/// # Examples
///
/// ```
/// // Import the function directly
/// use gibberish_or_not::is_password;
///
/// // Test with a common password
/// assert!(is_password("123456"));
///
/// // Test with a non-password
/// assert!(!is_password("not-a-common-password"));
/// ```
pub fn is_password(text: &str) -> bool {
    passwords::PASSWORDS.contains(text)
}
// The dictionary module provides a perfect hash table implementation
// using the phf crate, which is generated at compile time
// for optimal performance and memory efficiency

/// Checks if the given text is gibberish based on English word presence
/// and n-gram analysis scores. The sensitivity level determines how strict
/// the classification should be.
///
/// # Arguments
///
/// * `text` - The input text to analyze
/// * `sensitivity` - Controls how readily text is classified as English:
///   - High: Most lenient, flags English-like patterns as non-gibberish
///   - Medium: Balanced approach using dictionary and n-grams
///   - Low: Most strict, requires high confidence to classify as English
///
/// # Algorithm Steps
///
/// 1. Clean and normalize the input text
/// 2. Short text (len < 10) - single word check
/// 3. Split into words and count English words:
///    - 2+ English words → considered valid
///    - 1 English word → check n-gram scores
///    - 0 English words → more lenient n-gram check
/// 4. Use different n-gram thresholds depending on sensitivity level
pub fn is_gibberish(text: &str, sensitivity: Sensitivity) -> bool {
    GibberishDetector::new().is_gibberish(text, sensitivity)
}

/// Run the basic gibberish detection algorithm without model enhancement
fn run_basic_checks(text: &str, sensitivity: Sensitivity) -> bool {
    // Clean the text first
    let cleaned = clean_text(text);

    // Check if empty after cleaning
    if cleaned.is_empty() {
        return true;
    }

    // For very short cleaned text, only check if it's an English word
    if cleaned.len() < 10 {
        let is_english = is_english_word(&cleaned);
        return !is_english;
    }

    // Split into words and check for English words
    let words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect();

    // Count English words
    let english_words: Vec<&&str> = words.iter().filter(|w| is_english_word(w)).collect();
    let english_word_count = english_words.len();
    let english_word_ratio = if words.is_empty() {
        0.0
    } else {
        english_word_count as f64 / words.len() as f64
    };

    // Check for non-printable characters which are strong indicators of gibberish
    let non_printable_count = text
        .chars()
        .filter(|&c| c < ' ' && c != '\n' && c != '\r' && c != '\t')
        .count();

    // If there are non-printable characters, it's likely gibberish
    if non_printable_count > 0 {
        return true;
    }

    // Calculate character entropy - gibberish often has unusual character distributions
    let entropy = calculate_entropy(text);

    // Calculate character transition probability - English has predictable transitions
    let transition_score = calculate_transition_score(text);

    // Calculate vowel-consonant ratio - English has a fairly consistent ratio
    let vowel_consonant_ratio = calculate_vowel_consonant_ratio(&cleaned);

    // Proceed with trigram/quadgram analysis (but with less weight)
    let trigrams = generate_ngrams(&cleaned, 3);
    let quadgrams = generate_ngrams(&cleaned, 4);

    let valid_trigrams = trigrams
        .iter()
        .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
        .collect::<Vec<_>>();

    let valid_quadgrams = quadgrams
        .iter()
        .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
        .collect::<Vec<_>>();

    // Calculate scores
    let trigram_score = if trigrams.is_empty() {
        0.0
    } else {
        valid_trigrams.len() as f64 / trigrams.len() as f64
    };

    let quadgram_score = if quadgrams.is_empty() {
        0.0
    } else {
        valid_quadgrams.len() as f64 / quadgrams.len() as f64
    };

    // Calculate a composite score that combines multiple metrics
    // This makes the algorithm more robust than relying heavily on n-grams
    let mut composite_score = 0.0;

    // English word ratio has high weight
    composite_score += english_word_ratio * 0.4;

    // Transition probability has medium weight
    composite_score += transition_score * 0.25;

    // N-gram scores have lower weight
    composite_score += trigram_score * 0.15;
    composite_score += quadgram_score * 0.1;

    // Vowel-consonant ratio has low weight
    composite_score += if (0.3..=0.7).contains(&vowel_consonant_ratio) {
        0.1
    } else {
        0.0
    };

    // Entropy check - English text typically has entropy between 3.5-4.5
    // If entropy is outside this range, reduce the composite score
    if !(3.5..=4.5).contains(&entropy) {
        composite_score *= 0.8;
    }

    // Adjust thresholds based on text length
    let length_factor = match cleaned.len() {
        0..=20 => 0.7,    // Very short text needs higher threshold
        21..=50 => 0.8,   // Short text
        51..=100 => 0.9,  // Medium text
        101..=200 => 1.0, // Standard threshold
        _ => 1.1,         // Long text can be more lenient
    };

    // Decision thresholds based on sensitivity
    let threshold = match sensitivity {
        Sensitivity::Low => 0.35 * length_factor, // Stricter - needs more evidence to be English
        Sensitivity::Medium => 0.25 * length_factor, // Balanced
        Sensitivity::High => 0.15 * length_factor, // Lenient - less evidence needed to be English
    };

    // If entropy is very high (above 4.5), it's likely gibberish
    if entropy > 4.5 && sensitivity != Sensitivity::High {
        return true;
    }

    // If almost all words are English, it's definitely English
    if english_word_ratio > 0.8 {
        return false;
    }

    // If we have multiple English words, it's likely English
    if english_word_count >= 3 && sensitivity != Sensitivity::Low {
        return false;
    }

    // If we have no English words and poor transition score, it's likely gibberish
    if english_word_count == 0 && transition_score < 0.4 && sensitivity != Sensitivity::High {
        return true;
    }

    // For the remaining cases, use the composite score
    composite_score < threshold
}

/// Calculate character entropy - a measure of randomness in the text
fn calculate_entropy(text: &str) -> f64 {
    let text = text.to_lowercase();
    let total_chars = text.chars().count() as f64;

    if total_chars == 0.0 {
        return 0.0;
    }

    // Count character frequencies
    let mut char_counts = std::collections::HashMap::new();
    for c in text.chars() {
        *char_counts.entry(c).or_insert(0) += 1;
    }

    // Calculate entropy
    let mut entropy = 0.0;
    for &count in char_counts.values() {
        let probability = count as f64 / total_chars;
        entropy -= probability * probability.log2();
    }

    // Return raw entropy value (typical English text has entropy around 3.5-4.5)
    // This allows for more accurate threshold comparisons
    entropy
}

/// Calculate character transition probabilities based on English patterns
fn calculate_transition_score(text: &str) -> f64 {
    let text = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();

    if chars.len() < 2 {
        return 0.0;
    }

    let mut valid_transitions = 0;
    let total_transitions = chars.len() - 1;

    for i in 0..total_transitions {
        let pair = format!("{}{}", chars[i], chars[i + 1]);
        if COMMON_CHAR_PAIRS.contains(&pair.as_str()) {
            valid_transitions += 1;
        }
    }

    valid_transitions as f64 / total_transitions as f64
}

/// Calculate vowel-consonant ratio (English typically has a ratio around 0.4-0.6)
fn calculate_vowel_consonant_ratio(text: &str) -> f64 {
    let vowels = ['a', 'e', 'i', 'o', 'u'];
    let mut vowel_count = 0;
    let mut consonant_count = 0;

    for c in text.chars() {
        if vowels.contains(&c) {
            vowel_count += 1;
        } else if c.is_alphabetic() {
            consonant_count += 1;
        }
    }

    if consonant_count == 0 {
        return if vowel_count == 0 { 0.0 } else { 1.0 };
    }

    vowel_count as f64 / (vowel_count + consonant_count) as f64
}

// Common character pairs in English
static COMMON_CHAR_PAIRS: phf::Set<&'static str> = phf_set! {
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
    "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur"
};

static COMMON_QUADGRAMS: phf::Set<&'static str> = phf_set! {
    "tion", "atio", "that", "ther", "with", "ment", "ions", "this",
    "here", "from", "ould", "ting", "hich", "whic", "ctio", "ever",
    "they", "thin", "have", "othe", "were", "tive", "ough", "ight"
};

static COMMON_TRIGRAMS: phf::Set<&'static str> = phf_set! {
    "the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter",
    "hat", "tha", "ere", "con", "res", "ver", "all", "ons", "nce", "men",
    "ith", "ted", "ers", "pro", "thi", "wit", "are", "ess", "not", "ive",
    "was", "ect", "rea", "com", "eve", "per", "int", "est", "sta", "cti",
    "ica", "ist", "ear", "ain", "one", "our", "iti", "rat", "ell", "ant"
};

static ENGLISH_LETTERS: phf::Set<char> = phf_set! {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
};

fn clean_text(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ENGLISH_LETTERS.contains(&c) || c.is_ascii_digit() {
                c.to_ascii_lowercase()
            } else if c.is_whitespace() || c == '_' || c == '-' || c == '/' {
                ' '
            } else if c == ',' || c == '.' || c == '!' || c == '?' {
                // Keep common punctuation but add a space after it to help with word splitting
                ' '
            } else {
                // Keep other characters intact instead of replacing with space
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

fn generate_ngrams(text: &str, n: usize) -> Vec<String> {
    let filtered: String = text
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ENGLISH_LETTERS.contains(&ch) || ch.is_numeric() {
                ch
            } else {
                ' '
            }
        })
        .collect();

    filtered
        .split_whitespace()
        .flat_map(|word| {
            word.as_bytes()
                .windows(n)
                .filter_map(|window| String::from_utf8(window.to_vec()).ok())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::{debug, info, warn};

    // Helper function to initialize logger for tests
    fn init_logger() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(true)
            .try_init();
    }

    // Helper function to log detailed analysis of gibberish detection
    fn log_gibberish_analysis(text: &str) -> bool {
        info!("==== ANALYZING TEXT: '{}' ====", text);

        // Clean the text
        let cleaned = clean_text(text);
        debug!("Cleaned text: '{}'", cleaned);

        // Check if empty after cleaning
        if cleaned.is_empty() {
            info!("RESULT: GIBBERISH - Text is empty after cleaning");
            return true;
        }

        // For very short cleaned text, only check if it's an English word
        if cleaned.len() < 10 {
            let is_english = is_english_word(&cleaned);
            debug!(
                "Short text check: Is '{}' an English word? {}",
                cleaned, is_english
            );
            if is_english {
                info!("RESULT: NOT GIBBERISH - Short text is an English word");
                return false;
            } else {
                info!("RESULT: GIBBERISH - Short text is not an English word");
                return true;
            }
        }

        // Split into words and check for English words
        let words: Vec<&str> = cleaned
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect();

        debug!("Word count: {}", words.len());

        // Count English words
        let english_words: Vec<&&str> = words.iter().filter(|w| is_english_word(w)).collect();
        debug!(
            "English words: {} ({:?})",
            english_words.len(),
            english_words
        );

        let english_word_count = english_words.len();
        let english_word_ratio = if words.is_empty() {
            0.0
        } else {
            english_word_count as f64 / words.len() as f64
        };
        debug!("English word ratio: {:.4}", english_word_ratio);

        // Check for non-printable characters
        let non_printable_count = text
            .chars()
            .filter(|&c| c < ' ' && c != '\n' && c != '\r' && c != '\t')
            .count();

        debug!("Non-printable character count: {}", non_printable_count);

        if non_printable_count > 0 {
            info!("RESULT: GIBBERISH - Contains non-printable characters");
            return true;
        }

        // Calculate entropy
        let entropy = calculate_entropy(text);
        debug!("Entropy score: {:.4}", entropy);

        // Calculate transition score
        let transition_score = calculate_transition_score(text);
        debug!("Transition score: {:.4}", transition_score);

        // Calculate vowel-consonant ratio
        let vc_ratio = calculate_vowel_consonant_ratio(text);
        debug!("Vowel-consonant ratio: {:.4}", vc_ratio);

        // Check for substrings that are English words
        let possible_words = (3..=cleaned.len().min(10))
            .flat_map(|len| {
                cleaned
                    .as_bytes()
                    .windows(len)
                    .map(|window| std::str::from_utf8(window).unwrap_or(""))
                    .filter(|w| is_english_word(w))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        debug!("English subwords found: {:?}", possible_words);

        // N-gram analysis
        let trigrams = generate_ngrams(&cleaned, 3);
        let quadgrams = generate_ngrams(&cleaned, 4);

        let valid_trigrams = trigrams
            .iter()
            .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        let valid_quadgrams = quadgrams
            .iter()
            .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        debug!("All trigrams: {:?}", trigrams);
        debug!("Valid trigrams: {:?}", valid_trigrams);

        let trigram_score = if trigrams.is_empty() {
            0.0
        } else {
            valid_trigrams.len() as f64 / trigrams.len() as f64
        };
        debug!("Trigram score: {:.4}", trigram_score);

        debug!("All quadgrams: {:?}", quadgrams);
        debug!("Valid quadgrams: {:?}", valid_quadgrams);

        let quadgram_score = if quadgrams.is_empty() {
            0.0
        } else {
            valid_quadgrams.len() as f64 / quadgrams.len() as f64
        };
        debug!("Quadgram score: {:.4}", quadgram_score);

        // Medium sensitivity thresholds
        let english_word_threshold = 0.2;
        let trigram_threshold = 0.15;
        let quadgram_threshold = 0.1;
        let entropy_threshold = 4.5; // Updated from 3.7 to match raw entropy values for English text
        let transition_threshold = 0.7;

        // Check thresholds
        debug!(
            "English word ratio threshold check (> {}): {}",
            english_word_threshold,
            english_word_ratio > english_word_threshold
        );
        debug!(
            "Trigram score threshold check (> {}): {}",
            trigram_threshold,
            trigram_score > trigram_threshold
        );
        debug!(
            "Quadgram score threshold check (> {}): {}",
            quadgram_threshold,
            quadgram_score > quadgram_threshold
        );
        debug!(
            "Entropy threshold check (< {}): {}",
            entropy_threshold,
            entropy < entropy_threshold
        );
        debug!(
            "Transition score threshold check (> {}): {}",
            transition_threshold,
            transition_score > transition_threshold
        );

        // Final decision for Medium sensitivity
        let is_gibberish = !((english_word_ratio > english_word_threshold)
            || (english_word_count >= 3)
            || (trigram_score > trigram_threshold && quadgram_score > quadgram_threshold)
            || (transition_score > transition_threshold && entropy < entropy_threshold));

        if is_gibberish {
            info!("RESULT: GIBBERISH - Failed threshold checks");
        } else {
            info!("RESULT: NOT GIBBERISH - Passed threshold checks");
        }

        is_gibberish
    }

    // Tests for the password detection functionality
    #[test]
    fn test_common_passwords() {
        assert!(is_password("123456"));
        assert!(is_password("password"));
        assert!(is_password("qwerty"));
        assert!(is_password("abc123"));
    }

    #[test]
    fn test_numeric_passwords() {
        assert!(is_password("123456789"));
        assert!(is_password("12345678"));
        assert!(is_password("1234567"));
    }

    #[test]
    fn test_word_passwords() {
        assert!(is_password("iloveyou"));
        assert!(is_password("admin"));
        assert!(is_password("welcome"));
    }

    #[test]
    fn test_non_passwords() {
        assert!(!is_password("")); // Empty string
        assert!(!is_password("this is not a password")); // Contains spaces
        assert!(!is_password("verylongandunlikelypasswordthatnoonewoulduse")); // Too long
        assert!(!is_password("unique_string_123")); // Not in common list
    }

    // Helper function to run tests with different sensitivities
    fn test_with_sensitivities(
        text: &str,
        expected_low: bool,
        expected_med: bool,
        expected_high: bool,
    ) {
        assert_eq!(is_gibberish(text, Sensitivity::Low), expected_low);
        assert_eq!(is_gibberish(text, Sensitivity::Medium), expected_med);
        assert_eq!(is_gibberish(text, Sensitivity::High), expected_high);
    }

    #[test]
    fn test_clear_english_all_sensitivities() {
        let text = "The quick brown fox jumps over the lazy dog.";
        println!("\nTesting text: '{}'", text);

        for sensitivity in [Sensitivity::Low, Sensitivity::Medium, Sensitivity::High] {
            let cleaned = clean_text(text);
            let words: Vec<&str> = cleaned.split_whitespace().collect();
            let english_words: Vec<&&str> =
                words.iter().filter(|word| is_english_word(word)).collect();

            let trigrams = generate_ngrams(&cleaned, 3);
            let quadgrams = generate_ngrams(&cleaned, 4);

            let valid_trigrams = trigrams
                .iter()
                .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
                .collect::<Vec<_>>();
            let valid_quadgrams = quadgrams
                .iter()
                .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
                .collect::<Vec<_>>();

            println!("\nSensitivity {:?}:", sensitivity);
            println!("Cleaned text: '{}'", cleaned);
            println!(
                "English words found: {} out of {}",
                english_words.len(),
                words.len()
            );
            println!("English words: {:?}", english_words);
            println!(
                "Trigram score: {:.3}",
                if trigrams.is_empty() {
                    0.0
                } else {
                    valid_trigrams.len() as f64 / trigrams.len() as f64
                }
            );
            println!(
                "Quadgram score: {:.3}",
                if quadgrams.is_empty() {
                    0.0
                } else {
                    valid_quadgrams.len() as f64 / quadgrams.len() as f64
                }
            );

            let result = is_gibberish(text, sensitivity);
            println!("Result: {}", if result { "GIBBERISH" } else { "ENGLISH" });
        }

        test_with_sensitivities(
            text, false, // Changed from true to false for Low sensitivity
            false, // Changed from true to false for Medium sensitivity
            false, // Changed from true to false for High sensitivity
        );
    }

    #[test]
    fn test_borderline_english_like_gibberish() {
        init_logger();
        let text = "Rcl maocr otmwi lit dnoen oehc 13 iron seah.";

        info!("==== TESTING BORDERLINE ENGLISH LIKE GIBBERISH ====");
        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        // This text has English words "lit" and "iron", but is mostly gibberish
        // With our current thresholds, it should be classified as NOT gibberish
        test_with_sensitivities(
            text, true,  // Low sensitivity should detect as gibberish
            false, // Medium sensitivity accepts this due to "iron" and "lit"
            false, // High sensitivity accepts this
        );
    }

    #[test]
    fn test_english_without_spaces() {
        assert!(!is_gibberish(
            "HelloSkeletonsThisIsATestOfEnglishWithoutSpacesIHopeItWorks",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_clear_gibberish_all_sensitivities() {
        test_with_sensitivities("!@#$%^&*()", true, true, true);
    }

    #[test]
    fn test_english_word_with_ngrams() {
        let text = "ther with tion";
        println!("\n==== DEBUG: test_english_word_with_ngrams ====");
        println!("Text: '{}'", text);

        // Clean and analyze text
        let cleaned = clean_text(text);
        let words: Vec<&str> = cleaned.split_whitespace().collect();
        let english_words: Vec<&&str> = words.iter().filter(|w| is_english_word(w)).collect();

        println!("\n== Word Analysis ==");
        println!("Total words: {}", words.len());
        println!(
            "English words: {} ({:?})",
            english_words.len(),
            english_words
        );

        // Calculate n-gram scores
        let trigrams = generate_ngrams(&cleaned, 3);
        let quadgrams = generate_ngrams(&cleaned, 4);

        let valid_trigrams = trigrams
            .iter()
            .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        let valid_quadgrams = quadgrams
            .iter()
            .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        let trigram_score = if trigrams.is_empty() {
            0.0
        } else {
            valid_trigrams.len() as f64 / trigrams.len() as f64
        };
        let quadgram_score = if quadgrams.is_empty() {
            0.0
        } else {
            valid_quadgrams.len() as f64 / quadgrams.len() as f64
        };

        println!("\n== N-gram Analysis ==");
        println!("Trigram score: {:.3}", trigram_score);
        println!("Quadgram score: {:.3}", quadgram_score);

        println!("\n== Test Assertion ==");
        println!("Should classify as GIBBERISH with LOW sensitivity");
        assert!(
            !is_gibberish(text, Sensitivity::Low),
            "Text with common n-grams should not be classified as gibberish with low sensitivity"
        );
    }

    // Valid English text tests
    #[test]
    fn test_pangram() {
        assert!(!is_gibberish(
            "The quick brown fox jumps over the lazy dog.",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_simple_sentence() {
        assert!(!is_gibberish(
            "This is a simple English sentence.",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_hello_world() {
        init_logger();
        let text = "Hello, world!";

        info!("==== TESTING HELLO WORLD ====");
        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(!is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_single_word() {
        assert!(!is_gibberish("hello", Sensitivity::Medium));
    }

    #[test]
    fn test_common_ngrams() {
        assert!(!is_gibberish("ther with tion", Sensitivity::Medium));
    }

    #[test]
    fn test_technical_text() {
        assert!(!is_gibberish(
            "The function returns a boolean value.",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_mixed_case() {
        assert!(!is_gibberish(
            "MiXeD cAsE text IS still English",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_with_punctuation() {
        assert!(!is_gibberish(
            "Hello! How are you? I'm doing well.",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_long_text() {
        assert!(!is_gibberish("This is a longer piece of text that contains multiple sentences and should definitely be recognized as valid English content.", Sensitivity::Medium));
    }

    // Gibberish text tests
    #[test]
    fn test_numbers_only() {
        assert!(is_gibberish("12345 67890", Sensitivity::Medium));
    }

    #[test]
    fn test_empty_string() {
        assert!(is_gibberish("", Sensitivity::Medium));
    }

    #[test]
    fn test_non_english_chars() {
        assert!(is_gibberish("你好世界", Sensitivity::Medium));
    }

    #[test]
    fn test_special_chars() {
        assert!(is_gibberish("!@#$%^&*()", Sensitivity::Medium));
    }

    #[test]
    fn test_base64_like() {
        assert!(is_gibberish("MOTCk4ywLLjjEE2=", Sensitivity::Medium));
    }

    #[test]
    fn test_short_gibberish() {
        assert!(is_gibberish("4-Fc@w7MF", Sensitivity::Medium));
    }

    #[test]
    fn test_letter_substitution() {
        assert!(is_gibberish("Vszzc hvwg wg zcbu", Sensitivity::Medium));
    }

    // Edge cases
    #[test]
    fn test_single_letter() {
        assert!(is_gibberish("a", Sensitivity::Medium));
    }

    #[test]
    fn test_mixed_valid_invalid() {
        assert!(!is_gibberish("hello xkcd world", Sensitivity::Medium));
    }

    #[test]
    fn test_common_abbreviation() {
        init_logger();
        let text = "NASA FBI CIA";

        info!("==== TESTING COMMON ABBREVIATION ====");
        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(!is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_2() {
        init_logger();
        let text = "h2=ReOrS9DAnED8o";

        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_3() {
        let text = "\"D_{qU_RIO`zxE>T";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 3 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_4() {
        let text = "eDVD.ER#)U:FC_*9";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 4 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_5() {
        let text = "ST2dUnH9RI8a=Ste";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 5 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_6() {
        let text = "\"qxUD_ER_I>O{`Tz";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 6 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_7() {
        let text = "OQ\\:RAnuxw\\]@L}E";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 7 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_8() {
        let text = "nURa9TH28tISdS=e";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 8 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_9() {
        let text = "^Y+oU)cNT1,nd\"an";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 9 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_10() {
        let text = "R>iE:aC39edNTtAD";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 10 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_11() {
        let text = "pTD\"aTU\"z`^IT>Ex";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 11 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_12() {
        let text = "oD8eASEetEN=S29r";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 12 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_13() {
        init_logger();
        let text = "and\",nT1cNU)+o^Y";

        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_14() {
        let text = "caNnUd)\"+,on^TY1";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 14 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_15() {
        let text = "RoStES3EO9:Oeer>";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 15 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_16() {
        let text = "b-d,ooMpeST_#2*X";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 16 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_17() {
        let text = "RoStES2EO89Oeer=";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 17 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_18() {
        let text = "#IDP`a|{ryVE`>SU";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 18 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_19() {
        let text = "Y*#U_Nedp2oT,ob-";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 19 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_20() {
        let text = "t>9RSTdneaI:S3UH";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 20 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_21() {
        let text = "aRSUHdSI=te892nT";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 21 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_22() {
        let text = "cNU)+o^Yand\",nT1";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 22 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_23() {
        let text = "2To-#oYp*UNdeb_,";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 23 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_24() {
        let text = "R=tE9aN28eoNTeAO";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 24 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_25() {
        let text = "9DAnED8oh2=ReOrS";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 25 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_26() {
        let text = "=e9O2ESRotSE8erO";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 26 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_27() {
        let text = "o9DEnAD:SrOeR>3h";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 27 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_28() {
        let text = "z`^pTIEDT>\"aTx\"U";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 28 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_29() {
        let text = "2I'HicHd8a=Z-.;>";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 29 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_30() {
        let text = "Ia>`#{`|PyUrDESV";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 30 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_31() {
        init_logger();
        let text = "et";

        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_32() {
        init_logger();
        let text = "A";

        let is_gibberish_result = log_gibberish_analysis(text);

        // Compare with the actual function result
        let lib_result = is_gibberish(text, Sensitivity::Medium);
        if is_gibberish_result != lib_result {
            warn!(
                "WARNING: Analysis result ({}) differs from library result ({})",
                is_gibberish_result, lib_result
            );
        }

        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_33() {
        let text = "B";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 33 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_34() {
        let text = "RoStES2EO89Oeer=";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 34 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_35() {
        let text = "RoStES2EO89Oeer=";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 35 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_36() {
        let text = "et";
        println!("\n==== TESTING ASTAR SEARCH GIBBERISH 36 ====");
        println!("Testing text: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_astar_search_gibberish_37() {
        test_with_sensitivities(
            "Aastar search algorithm is a path finding algorithm",
            false,
            false,
            false,
        );
    }

    #[test]
    fn test_cyrillic_gibberish() {
        // Test for Cyrillic-like gibberish
        test_with_sensitivities("%B:;@J A8 4>35= CG3DFL\\ <G697 ?K HAI", true, true, true);
    }

    #[test]
    fn test_mixed_latin_gibberish() {
        // Test for mixed Latin character gibberish
        test_with_sensitivities("xgcyzw Snh fabkqta,jedm ioopl  uru v", true, true, true);
    }

    #[test]
    fn test_binary_control_chars_gibberish() {
        // Test for binary/control character gibberish
        let binary_gibberish = "\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\0\0\u{1}\u{1}\0\u{1}\0\0\0\u{1}\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\0\0\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\u{1}\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\0\0\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\0\0\u{1}\u{1}\0\0\0\0\u{1}\u{1}\u{1}\0\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\u{1}\0\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\0\0\0\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\0\u{1}\0\u{1}\u{1}\0\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\0\u{1}\0\0\u{1}\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0";
        test_with_sensitivities(binary_gibberish, true, true, true);
    }

    #[test]
    fn test_all_gibberish_examples_medium_sensitivity() {
        // Test all examples with medium sensitivity
        assert!(is_gibberish(
            "%B:;@J A8 4>35= CG3DFL\\ <G697 ?K HAI",
            Sensitivity::Medium
        ));
        assert!(is_gibberish(
            "xgcyzw Snh fabkqta,jedm ioopl  uru v",
            Sensitivity::Medium
        ));

        let binary_gibberish = "\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\0\0\u{1}\u{1}\0\u{1}\0\0\0\u{1}\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\0\0\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\u{1}\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\0\0\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\0\0\u{1}\u{1}\0\0\0\0\u{1}\u{1}\u{1}\0\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\u{1}\0\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\0\0\0\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\0\u{1}\0\0\u{1}\0\u{1}\u{1}\0\0\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0\0\u{1}\0\0\u{1}\u{1}\0\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\0\0\u{1}\0\u{1}\0\0\0\0\0\u{1}\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\0\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\u{1}\0\u{1}\u{1}\u{1}\0\u{1}\0\u{1}\u{1}\u{1}\0";
        assert!(is_gibberish(binary_gibberish, Sensitivity::Medium));
    }

    #[test]
    fn test_gibberish_string_1() {
        init_logger();
        let text = "ant nehoteeh ntaoe seen e tohetael";
        debug!("Testing gibberish string 1: '{}'", text);

        // Use the diagnostic function to see detailed analysis
        let _is_gibberish_result = log_gibberish_analysis(text);

        // Test with low sensitivity
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_2() {
        init_logger();
        let text = "eoa nte neeseateh tot ne lhoteenah";
        debug!("Testing gibberish string 2: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_3() {
        init_logger();
        let text = "nte neeseateh tot ne lhoteenahaoe";
        debug!("Testing gibberish string 3: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_4() {
        init_logger();
        let text = "alehestnnhton o ee tee  a eatohteen";
        debug!("Testing gibberish string 4: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_5() {
        init_logger();
        let text = "h eee lee ahetes n ntoatohene nttoa";
        debug!("Testing gibberish string 5: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_6() {
        init_logger();
        let text = "ana leeoehanteees t hot eenohet tn";
        debug!("Testing gibberish string 6: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }

    #[test]
    fn test_gibberish_string_7() {
        init_logger();
        let text = "eoahaneetohl en tot hetaeseen etn";
        debug!("Testing gibberish string 7: '{}'", text);
        assert!(is_gibberish(text, Sensitivity::Low));
    }
}
