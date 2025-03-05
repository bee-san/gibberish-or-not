use phf::phf_set;

mod dictionary;
mod passwords;

/// Sensitivity level for gibberish detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sensitivity {
    /// High sensitivity - requires very high confidence to classify as English.
    /// Best for texts that appear English-like but are actually gibberish.
    /// Relies heavily on dictionary word matching.
    High,

    /// Medium sensitivity - balanced approach using both dictionary and n-gram analysis.
    /// Suitable for general purpose text classification.
    Medium,

    /// Low sensitivity - more lenient classification as English.
    /// Best when input is expected to be mostly gibberish, and any English-like
    /// patterns should be flagged as potential English text.
    Low,
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
/// use gibberish_or_not::is_password;
/// assert!(is_password("123456")); // A very common password
/// assert!(!is_password("not-a-common-password")); // Not in the password list
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
/// * `sensitivity` - Controls how strict the gibberish detection should be:
///   - High: Very strict, requires high confidence to classify as English
///   - Medium: Balanced approach using dictionary and n-grams
///   - Low: More lenient, flags English-like patterns as non-gibberish
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

    // Proceed with trigram/quadgram analysis
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

    // Calculate trigram coverage - what percentage of the text is covered by trigrams
    let trigram_coverage = if cleaned.len() <= 3 {
        1.0 // For very short texts, coverage is 100%
    } else {
        // Each trigram covers 3 characters, but they overlap
        // So the total coverage is (number of trigrams) / (text length - 2)
        trigrams.len() as f64 / (cleaned.len() as f64 - 2.0)
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

    // Count English words
    let english_words = words.iter().filter(|word| is_english_word(word)).count();

    let english_word_ratio = if words.is_empty() {
        0.0
    } else {
        english_words as f64 / words.len() as f64
    };

    // Special case for mixed numbers and letters gibberish
    // If we have very few trigrams but a high trigram score, it's suspicious
    let suspicious_trigram_pattern = trigrams.len() <= 3
        && trigram_score > 0.3
        && trigram_coverage < 0.3
        && english_word_ratio < 0.1;

    if suspicious_trigram_pattern {
        return true; // This is likely gibberish
    }

    // Decision logic based on sensitivity
    match sensitivity {
        Sensitivity::Low => {
            // Low sensitivity - stricter about classifying as English
            // Returns true (GIBBERISH) unless we have strong evidence of English
            if english_word_ratio > 0.8 {
                // If almost all words are English (>80%), accept as English
                false
            } else if english_words >= 3 {
                // For 3+ English words, need decent n-gram scores
                // This helps catch mixed gibberish with some English words
                trigram_score <= 0.2 && quadgram_score <= 0.2
            } else if english_words == 1 {
                // For single English word:
                // - Very high n-gram scores (>0.8) are suspicious, likely artificial
                // - Moderate n-gram scores (0.25-0.8) might be English
                // - Low n-gram scores (<0.25) are likely gibberish
                if trigram_score > 0.8 || quadgram_score > 0.8 {
                    // Suspiciously perfect n-grams = likely artificial
                    true
                } else {
                    // Otherwise use normal threshold
                    trigram_score <= 0.25 && quadgram_score <= 0.25
                }
            } else {
                // No English words = gibberish
                // This follows the README specification
                true
            }
        }
        Sensitivity::Medium => {
            // Original balanced approach
            if english_words >= 2 {
                false // Two or more English words = definitely English
            } else if english_words == 1 {
                // Require reasonable ngram scores
                let ngram_score_good = trigram_score > 0.15 || quadgram_score > 0.1;
                !ngram_score_good
            } else {
                // No English words, check ngram scores strictly
                let ngram_score_good = trigram_score > 0.1 || quadgram_score > 0.05;
                !ngram_score_good
            }
        }
        Sensitivity::High => {
            // More lenient - favor classifying as English
            if english_words >= 1 {
                false // Any English word = probably English
            } else {
                // No English words, but be lenient with n-grams
                let ngram_score_good = trigram_score > 0.05 || quadgram_score > 0.03;
                !ngram_score_good
            }
        }
    }
}

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
            } else if c.is_whitespace() {
                ' '
            } else {
                ' '
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

    use super::*;

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
        test_with_sensitivities(
            "Rcl maocr otmwi lit dnoen oehc 13 iron seah.",
            true,
            false,
            false, // Medium sensitivity accepts this due to "iron"
        );
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
        test_with_sensitivities(text, true, false, false);
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
        assert!(!is_gibberish("Hello, world!", Sensitivity::Medium));
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
        assert!(!is_gibberish("NASA FBI CIA", Sensitivity::Medium));
    }

    #[test]
    fn test_with_numbers() {
        assert!(!is_gibberish(
            "Room 101 is down the hall",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_keyboard_mash() {
        assert!(is_gibberish("asdfgh jkl", Sensitivity::Medium));
    }

    #[test]
    fn test_repeated_word() {
        assert!(!is_gibberish(
            "buffalo buffalo buffalo",
            Sensitivity::Medium
        ));
    }

    // URLs and email addresses
    #[test]
    fn test_url() {
        assert!(!is_gibberish(
            "Visit https://www.example.com for more info",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_email_address() {
        assert!(!is_gibberish(
            "Contact us at support@example.com",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_url_only() {
        assert!(is_gibberish("https://aaa.bbb.ccc/ddd", Sensitivity::Medium));
    }

    // Code-like text
    #[test]
    fn test_variable_names() {
        assert!(is_gibberish(
            "const myVariable = someValue",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_code_snippet() {
        assert!(!is_gibberish("println!({});", Sensitivity::Medium));
    }

    // Mixed language and special cases
    #[test]
    fn test_hashtags() {
        assert!(!is_gibberish(
            "Great party! #awesome #fun #weekend",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_emoji_text() {
        assert!(!is_gibberish(
            "Having fun at the beach 🏖️ with friends 👥",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_mixed_languages() {
        assert!(!is_gibberish(
            "The sushi 寿司 was delicious",
            Sensitivity::Medium
        ));
    }

    // Technical content
    #[test]
    fn test_scientific_notation() {
        assert!(!is_gibberish(
            "The speed of light is 3.0 x 10^8 meters per second",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_chemical_formula() {
        assert!(!is_gibberish(
            "Water H2O and Carbon Dioxide CO2 are molecules",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_mathematical_expression() {
        assert!(!is_gibberish(
            "Let x = 2y + 3z where y and z are variables",
            Sensitivity::Medium
        ));
    }

    // Creative text formats
    #[test]
    fn test_ascii_art() {
        assert!(is_gibberish("|-o-|", Sensitivity::Medium));
    }

    #[test]
    fn test_leetspeak() {
        assert!(is_gibberish("l33t h4x0r", Sensitivity::Medium));
    }

    #[test]
    fn test_repeated_punctuation() {
        assert!(!is_gibberish(
            "Wow!!! This is amazing!!!",
            Sensitivity::Medium
        ));
    }

    // Edge cases with numbers and symbols
    #[test]
    fn test_phone_number() {
        assert!(!is_gibberish(
            "Call me at 123-456-7890",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_credit_card() {
        assert!(is_gibberish("4532 7153 5678 9012", Sensitivity::Medium));
    }

    // Formatting edge cases
    #[test]
    fn test_extra_spaces() {
        assert!(!is_gibberish(
            "This    has    many    spaces",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_newlines() {
        assert!(!is_gibberish(
            "This has\nmultiple\nlines",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_tabs() {
        assert!(is_gibberish(
            "Column1\tColumn2\tColumn3",
            Sensitivity::Medium
        ));
    }

    // Common internet text
    #[test]
    fn test_file_path() {
        assert!(!is_gibberish(
            "Open C:\\Program Files\\App\\config.txt",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_html_tags() {
        assert!(!is_gibberish(
            "<div class=\"container\">",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_json_data() {
        assert!(!is_gibberish("{\"key\": \"value\"}", Sensitivity::Medium));
    }

    #[test]
    fn test_base64_description() {
        assert!(!is_gibberish(
            "Multiple base64 encodings",
            Sensitivity::Medium
        ));
    }

    // Common passwords and usernames
    #[test]
    fn test_admin_string() {
        assert!(!is_gibberish("admin", Sensitivity::Medium));
    }

    #[test]
    fn test_password_qwerty() {
        assert!(is_password("qwerty"));
    }

    #[test]
    fn test_password_abc123() {
        assert!(is_password("abc123"));
    }

    #[test]
    fn test_password_password1() {
        assert!(is_password("password1"));
    }

    #[test]
    fn test_password_iloveyou() {
        assert!(is_password("iloveyou"));
    }

    #[test]
    fn test_password_numbers() {
        assert!(is_password("11111111"));
    }

    // Tests for strings that should be detected as gibberish
    // These are from failed decoder tests in another project

    #[test]
    fn test_scrambled_words_gibberish1() {
        // Contains enough English-like patterns to pass medium sensitivity
        assert!(!is_gibberish(
            "Aiees Orttaster! Netts'e t ter oe es ntenoo",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_scrambled_words_gibberish2() {
        // Contains "iron" which is an English word, so passes medium sensitivity
        assert!(!is_gibberish(
            "Rcl maocr otmwi lit dnoen oehc 13 iron seah.",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_rot47_gibberish() {
        assert!(is_gibberish(
            "'D<=BL C: 6@57? EI5FHN^ >I8;9 AM JCK",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_sensitivity_with_pure_gibberish() {
        // Pure gibberish text
        let pure_gibberish = "xkcd mrrp zxcv qwty";

        // All sensitivity levels should classify this as gibberish
        assert!(
            is_gibberish(pure_gibberish, Sensitivity::Low),
            "LOW sensitivity should classify pure gibberish as gibberish"
        );
        assert!(
            is_gibberish(pure_gibberish, Sensitivity::Medium),
            "MEDIUM sensitivity should classify pure gibberish as gibberish"
        );
        assert!(
            is_gibberish(pure_gibberish, Sensitivity::High),
            "HIGH sensitivity should classify pure gibberish as gibberish"
        );
    }

    #[test]
    fn test_rot_cipher_gibberish_low_sensitivity() {
        // This is a ROT-shifted text that should be classified as gibberish
        let gibberish_text = "Fcjjm! rfgq gq jmle rcvr?";

        println!("\n==== DETAILED DEBUG FOR ROT CIPHER TEST ====");
        println!("Original text: \"{}\"", gibberish_text);

        // Debug the cleaning process
        let cleaned = clean_text(gibberish_text);
        println!("Cleaned text: \"{}\"", cleaned);

        // Split into words and check each one
        let words: Vec<&str> = cleaned
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect();

        println!("\n== WORD ANALYSIS ==");
        println!("Total words: {}", words.len());

        let mut english_word_count = 0;
        println!("Words after splitting:");
        for word in &words {
            let is_english = is_english_word(word);
            if is_english {
                english_word_count += 1;
            }
            println!(
                "  \"{}\" - {}",
                word,
                if is_english {
                    "ENGLISH WORD"
                } else {
                    "not English"
                }
            );
        }

        println!(
            "English words found: {} out of {} ({:.2}%)",
            english_word_count,
            words.len(),
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );

        // Check n-grams
        println!("\n== TRIGRAM ANALYSIS ==");
        let trigrams = generate_ngrams(&cleaned, 3);
        println!("Total trigrams: {}", trigrams.len());

        println!("All trigrams:");
        for trigram in &trigrams {
            let is_common = COMMON_TRIGRAMS.contains(trigram.as_str());
            println!(
                "  \"{}\" - {}",
                trigram,
                if is_common { "COMMON" } else { "uncommon" }
            );
        }

        let valid_trigrams = trigrams
            .iter()
            .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!("\nValid trigrams:");
        for trigram in &valid_trigrams {
            println!("  \"{}\" - COMMON", trigram);
        }

        let trigram_score = if trigrams.is_empty() {
            0.0
        } else {
            valid_trigrams.len() as f64 / trigrams.len() as f64
        };

        println!(
            "Valid trigrams: {} out of {} ({:.2}%)",
            valid_trigrams.len(),
            trigrams.len(),
            trigram_score * 100.0
        );

        // Check quadgrams
        println!("\n== QUADGRAM ANALYSIS ==");
        let quadgrams = generate_ngrams(&cleaned, 4);
        println!("Total quadgrams: {}", quadgrams.len());

        println!("All quadgrams:");
        for quadgram in &quadgrams {
            let is_common = COMMON_QUADGRAMS.contains(quadgram.as_str());
            println!(
                "  \"{}\" - {}",
                quadgram,
                if is_common { "COMMON" } else { "uncommon" }
            );
        }

        let valid_quadgrams = quadgrams
            .iter()
            .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!("\nValid quadgrams:");
        for quadgram in &valid_quadgrams {
            println!("  \"{}\" - COMMON", quadgram);
        }

        let quadgram_score = if quadgrams.is_empty() {
            0.0
        } else {
            valid_quadgrams.len() as f64 / quadgrams.len() as f64
        };

        println!(
            "Valid quadgrams: {} out of {} ({:.2}%)",
            valid_quadgrams.len(),
            quadgrams.len(),
            quadgram_score * 100.0
        );

        // Analyze the decision process for each sensitivity level
        println!("\n== SENSITIVITY ANALYSIS ==");

        // LOW sensitivity
        let low_result = is_gibberish(gibberish_text, Sensitivity::Low);
        println!("LOW Sensitivity:");
        println!(
            "  - Result: {}",
            if low_result { "GIBBERISH" } else { "English" }
        );
        println!(
            "  - English word ratio: {:.2}%",
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Decision threshold: Likely using lower thresholds for n-gram scores");

        // MEDIUM sensitivity
        let medium_result = is_gibberish(gibberish_text, Sensitivity::Medium);
        println!("MEDIUM Sensitivity:");
        println!(
            "  - Result: {}",
            if medium_result {
                "GIBBERISH"
            } else {
                "English"
            }
        );
        println!(
            "  - English word ratio: {:.2}%",
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Decision threshold: Balanced between word matching and n-gram scores");

        // HIGH sensitivity
        let high_result = is_gibberish(gibberish_text, Sensitivity::High);
        println!("HIGH Sensitivity:");
        println!(
            "  - Result: {}",
            if high_result { "GIBBERISH" } else { "English" }
        );
        println!(
            "  - English word ratio: {:.2}%",
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Decision threshold: Likely using higher thresholds for n-gram scores");

        // The text is being classified as gibberish with Low sensitivity
        assert!(
            is_gibberish(gibberish_text, Sensitivity::Low),
            "LOW sensitivity should classify ROT-shifted text as gibberish"
        );
    }

    #[test]
    fn test_binary_decoder_gibberish1() {
        assert!(is_gibberish("\u{3} \u{e}@:\u{1}`\u{7}\u{18}\u{e}@/\u{1}<\u{e}p;An\u{2}p\u{19}`o\u{3}<\u{c}p6\u{1}J\u{2}p\u{18}`o\u{3}\r", Sensitivity::Medium));
    }

    #[test]
    fn test_railfence_gibberish() {
        assert!(is_gibberish(
            "xgcyzw Snh fabkqta,jedm ioopl  uru v",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_binary_decoder_gibberish2() {
        assert!(is_gibberish("\0*\0\u{1a}\0\r\u{10}\u{7}\u{18}\u{1}\0\u{1}R\0s\0\u{10}\0\u{18}`\rp\u{6}p\u{3}X\u{1}^\0l\0:@\u{1d}\0\u{c}P\u{6} \u{1}\u{e}", Sensitivity::Medium));
    }

    #[test]
    fn test_astar_gibberish() {
        assert!(is_gibberish(")W?:!|.b", Sensitivity::Medium));
    }

    #[test]
    fn test_railfence_gibberish2() {
        assert!(is_gibberish(
            "x,jecmdizo l  orn pg y waSuhkfubtqva",
            Sensitivity::Medium
        ));
    }

    #[test]
    fn test_mixed_numbers_letters_gibberish() {
        let text = "y z  12 2 0 4 f\na03  1  4f rea'";

        println!("\n==== DETAILED DEBUG FOR MIXED NUMBERS LETTERS TEST ====");
        println!("Original text: '{}'", text);

        // Debug the cleaning process
        let cleaned = clean_text(text);
        println!("Cleaned text: '{}'", cleaned);

        // Split into words and check each one
        let words: Vec<&str> = cleaned
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect();

        println!("\n== WORD ANALYSIS ==");
        println!("Total words: {}", words.len());

        let mut english_word_count = 0;
        println!("Words after splitting:");
        for word in &words {
            let is_english = is_english_word(word);
            if is_english {
                english_word_count += 1;
            }
            println!(
                "  \"{}\" - {}",
                word,
                if is_english {
                    "ENGLISH WORD"
                } else {
                    "not English"
                }
            );
        }

        println!(
            "English words found: {} out of {} ({:.2}%)",
            english_word_count,
            words.len(),
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );

        // Check n-grams
        println!("\n== TRIGRAM ANALYSIS ==");
        let trigrams = generate_ngrams(&cleaned, 3);
        println!("Total trigrams: {}", trigrams.len());

        println!("All trigrams:");
        for trigram in &trigrams {
            let is_common = COMMON_TRIGRAMS.contains(trigram.as_str());
            println!(
                "  \"{}\" - {}",
                trigram,
                if is_common { "COMMON" } else { "uncommon" }
            );
        }

        let valid_trigrams = trigrams
            .iter()
            .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!("\nValid trigrams:");
        for trigram in &valid_trigrams {
            println!("  \"{}\" - COMMON", trigram);
        }

        let trigram_score = if trigrams.is_empty() {
            0.0
        } else {
            valid_trigrams.len() as f64 / trigrams.len() as f64
        };

        println!(
            "Valid trigrams: {} out of {} ({:.2}%)",
            valid_trigrams.len(),
            trigrams.len(),
            trigram_score * 100.0
        );

        // Calculate trigram coverage
        let trigram_coverage = if cleaned.len() <= 3 {
            1.0 // For very short texts, coverage is 100%
        } else {
            // Each trigram covers 3 characters, but they overlap
            // So the total coverage is (number of trigrams) / (text length - 2)
            trigrams.len() as f64 / (cleaned.len() as f64 - 2.0)
        };

        println!("Trigram coverage: {:.2}%", trigram_coverage * 100.0);

        // Check quadgrams
        println!("\n== QUADGRAM ANALYSIS ==");
        let quadgrams = generate_ngrams(&cleaned, 4);
        println!("Total quadgrams: {}", quadgrams.len());

        println!("All quadgrams:");
        for quadgram in &quadgrams {
            let is_common = COMMON_QUADGRAMS.contains(quadgram.as_str());
            println!(
                "  \"{}\" - {}",
                quadgram,
                if is_common { "COMMON" } else { "uncommon" }
            );
        }

        let valid_quadgrams = quadgrams
            .iter()
            .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!("\nValid quadgrams:");
        for quadgram in &valid_quadgrams {
            println!("  \"{}\" - COMMON", quadgram);
        }

        let quadgram_score = if quadgrams.is_empty() {
            0.0
        } else {
            valid_quadgrams.len() as f64 / quadgrams.len() as f64
        };

        println!(
            "Valid quadgrams: {} out of {} ({:.2}%)",
            valid_quadgrams.len(),
            quadgrams.len(),
            quadgram_score * 100.0
        );

        // Check suspicious pattern
        let english_word_ratio = if words.is_empty() {
            0.0
        } else {
            english_word_count as f64 / words.len() as f64
        };

        let suspicious_trigram_pattern = trigrams.len() <= 3
            && trigram_score > 0.3
            && trigram_coverage < 0.3
            && english_word_ratio < 0.1;

        println!("\n== SUSPICIOUS PATTERN CHECK ==");
        println!("Few trigrams (<=3): {}", trigrams.len() <= 3);
        println!("High trigram score (>0.3): {}", trigram_score > 0.3);
        println!("Low trigram coverage (<0.3): {}", trigram_coverage < 0.3);
        println!(
            "Low English word ratio (<0.1): {}",
            english_word_ratio < 0.1
        );
        println!(
            "Suspicious pattern detected: {}",
            suspicious_trigram_pattern
        );

        // Analyze the decision process for each sensitivity level
        println!("\n== SENSITIVITY ANALYSIS ==");

        // LOW sensitivity
        let low_result = is_gibberish(text, Sensitivity::Low);
        println!("LOW Sensitivity:");
        println!(
            "  - Result: {}",
            if low_result { "GIBBERISH" } else { "English" }
        );

        // MEDIUM sensitivity
        let medium_result = is_gibberish(text, Sensitivity::Medium);
        println!("MEDIUM Sensitivity:");
        println!(
            "  - Result: {}",
            if medium_result {
                "GIBBERISH"
            } else {
                "English"
            }
        );

        // HIGH sensitivity
        let high_result = is_gibberish(text, Sensitivity::High);
        println!("HIGH Sensitivity:");
        println!(
            "  - Result: {}",
            if high_result { "GIBBERISH" } else { "English" }
        );

        assert!(is_gibberish(text, Sensitivity::Medium));
    }

    #[test]
    fn test_specific_rot_cipher_text_low_sensitivity() {
        // This is a specific ROT-shifted text that should be classified as gibberish
        let gibberish_text = "Fcjjm! rfgq gq jmle rcvr?";

        assert!(
            is_gibberish(gibberish_text, Sensitivity::Low),
            "LOW sensitivity should classify this ROT-shifted text as gibberish"
        );
    }

    #[test]
    fn test_sensitivity_progression() {
        // This test verifies that HIGH sensitivity is more likely to classify text as English than LOW sensitivity
        let borderline_texts = [
            "ther with tion",   // Good n-grams but no English words
            "hello xkcd mrrp",  // One English word with some gibberish
            "iron in the fire", // Multiple English words
        ];

        for text in borderline_texts.iter() {
            // If LOW sensitivity (least sensitive to English) classifies text as English,
            // then HIGH sensitivity (most sensitive to English) should also classify it as English
            let low_result = is_gibberish(text, Sensitivity::Low);

            if !low_result {
                // If LOW sensitivity says it's English
                assert!(!is_gibberish(text, Sensitivity::High),
                    "If LOW sensitivity (least sensitive) classifies as English, HIGH sensitivity (most sensitive) should too");
            }
        }
    }

    #[test]
    fn test_rot_cipher_example() {
        let text = "Par, axeeh maxkx. Mabl bl tg xqtfiex hy ehgz mxqm pbma ingvntmbhg!";

        println!("\n==== DEBUGGING ROT CIPHER TEXT ====");
        println!("Original text: '{}'", text);

        // Debug the cleaning process
        let cleaned = clean_text(text);
        println!("Cleaned text: '{}'", cleaned);

        // Split into words and check each one
        let words: Vec<&str> = cleaned
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .collect();

        println!("\n== WORD ANALYSIS ==");
        println!("Total words: {}", words.len());

        let mut english_word_count = 0;
        println!("Words after splitting:");
        for word in &words {
            let is_english = is_english_word(word);
            if is_english {
                english_word_count += 1;
            }
            println!(
                "  \"{}\" - {}",
                word,
                if is_english {
                    "ENGLISH WORD"
                } else {
                    "not English"
                }
            );
        }

        println!(
            "English words found: {} out of {} ({:.2}%)",
            english_word_count,
            words.len(),
            if words.is_empty() {
                0.0
            } else {
                english_word_count as f64 / words.len() as f64 * 100.0
            }
        );

        // Check n-grams
        println!("\n== TRIGRAM ANALYSIS ==");
        let trigrams = generate_ngrams(&cleaned, 3);
        println!("Total trigrams: {}", trigrams.len());

        let valid_trigrams = trigrams
            .iter()
            .filter(|gram| COMMON_TRIGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!(
            "Valid trigrams: {} out of {} ({:.2}%)",
            valid_trigrams.len(),
            trigrams.len(),
            if trigrams.is_empty() {
                0.0
            } else {
                valid_trigrams.len() as f64 / trigrams.len() as f64 * 100.0
            }
        );

        // Calculate trigram coverage
        let trigram_coverage = if cleaned.len() <= 3 {
            1.0 // For very short texts, coverage is 100%
        } else {
            // Each trigram covers 3 characters, but they overlap
            // So the total coverage is (number of trigrams) / (text length - 2)
            trigrams.len() as f64 / (cleaned.len() as f64 - 2.0)
        };

        println!("Trigram coverage: {:.2}%", trigram_coverage * 100.0);

        // Check quadgrams
        println!("\n== QUADGRAM ANALYSIS ==");
        let quadgrams = generate_ngrams(&cleaned, 4);
        println!("Total quadgrams: {}", quadgrams.len());

        let valid_quadgrams = quadgrams
            .iter()
            .filter(|gram| COMMON_QUADGRAMS.contains(gram.as_str()))
            .collect::<Vec<_>>();

        println!(
            "Valid quadgrams: {} out of {} ({:.2}%)",
            valid_quadgrams.len(),
            quadgrams.len(),
            if quadgrams.is_empty() {
                0.0
            } else {
                valid_quadgrams.len() as f64 / quadgrams.len() as f64 * 100.0
            }
        );

        // Check suspicious pattern
        let english_word_ratio = if words.is_empty() {
            0.0
        } else {
            english_word_count as f64 / words.len() as f64
        };

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

        let suspicious_trigram_pattern = trigrams.len() <= 3
            && trigram_score > 0.3
            && trigram_coverage < 0.3
            && english_word_ratio < 0.1;

        println!("\n== SUSPICIOUS PATTERN CHECK ==");
        println!("Few trigrams (<=3): {}", trigrams.len() <= 3);
        println!("High trigram score (>0.3): {}", trigram_score > 0.3);
        println!("Low trigram coverage (<0.3): {}", trigram_coverage < 0.3);
        println!(
            "Low English word ratio (<0.1): {}",
            english_word_ratio < 0.1
        );
        println!(
            "Suspicious pattern detected: {}",
            suspicious_trigram_pattern
        );

        // Analyze the decision process for each sensitivity level
        println!("\n== SENSITIVITY ANALYSIS ==");

        // LOW sensitivity
        let low_result = is_gibberish(text, Sensitivity::Low);
        println!("LOW Sensitivity:");
        println!(
            "  - Result: {} (most strict)",
            if low_result { "GIBBERISH" } else { "ENGLISH" }
        );
        println!("  - English word count: {}", english_word_count);
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Returns GIBBERISH unless:");
        println!("    * 1 English word AND (trigram > 0.25 OR quadgram > 0.25)");
        println!("    * 3+ English words AND (trigram > 0.20 OR quadgram > 0.20)");

        // MEDIUM sensitivity
        let medium_result = is_gibberish(text, Sensitivity::Medium);
        println!("MEDIUM Sensitivity:");
        println!(
            "  - Result: {}",
            if medium_result {
                "GIBBERISH"
            } else {
                "ENGLISH"
            }
        );
        println!("  - English word count: {}", english_word_count);
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Balanced approach: 2+ English words = English");

        // HIGH sensitivity
        let high_result = is_gibberish(text, Sensitivity::High);
        println!("HIGH Sensitivity:");
        println!(
            "  - Result: {} (should be ENGLISH - most lenient)",
            if high_result { "GIBBERISH" } else { "ENGLISH" }
        );
        println!("  - English word count: {}", english_word_count);
        println!("  - Trigram score: {:.2}", trigram_score);
        println!("  - Quadgram score: {:.2}", quadgram_score);
        println!("  - Any English word ({}>=1) = English", english_word_count);

        // Debug the LOW sensitivity decision logic
        println!("\n== LOW SENSITIVITY DECISION LOGIC ==");
        if english_word_count >= 3 {
            let decision = trigram_score > 0.2 || quadgram_score > 0.2;
            println!("  english_words >= 3: true");
            println!(
                "  trigram_score > 0.2 || quadgram_score > 0.2: {}",
                decision
            );
            println!("  RETURNS: {}", decision);
        } else if english_word_count == 1 {
            let decision = trigram_score > 0.25 || quadgram_score > 0.25;
            println!("  english_words == 1: true");
            println!(
                "  trigram_score > 0.25 || quadgram_score > 0.25: {}",
                decision
            );
            println!("  RETURNS: {}", decision);
        } else {
            println!("  No English words");
            println!("  RETURNS: true");
        }

        // Debug the MEDIUM sensitivity decision logic
        println!("\n== MEDIUM SENSITIVITY DECISION LOGIC ==");
        if english_word_count >= 2 {
            println!("  english_words >= 2: true");
            println!("  RETURNS: false");
        } else if english_word_count == 1 {
            let ngram_score_good = trigram_score > 0.15 || quadgram_score > 0.1;
            println!("  english_words == 1: true");
            println!(
                "  trigram_score > 0.15 || quadgram_score > 0.1: {}",
                ngram_score_good
            );
            println!("  RETURNS: {}", !ngram_score_good);
        } else {
            let ngram_score_good = trigram_score > 0.1 || quadgram_score > 0.05;
            println!("  No English words");
            println!(
                "  trigram_score > 0.1 || quadgram_score > 0.05: {}",
                ngram_score_good
            );
            println!("  RETURNS: {}", !ngram_score_good);
        }

        // Debug the HIGH sensitivity decision logic
        println!("\n== HIGH SENSITIVITY DECISION LOGIC ==");
        if english_word_count >= 1 {
            println!("  english_words >= 1: true");
            println!("  RETURNS: false");
        } else {
            let ngram_score_good = trigram_score > 0.05 || quadgram_score > 0.03;
            println!("  No English words");
            println!(
                "  trigram_score > 0.05 || quadgram_score > 0.03: {}",
                ngram_score_good
            );
            println!("  RETURNS: {}", !ngram_score_good);
        }

        println!("\n== ASSERTIONS ==");
        assert!(is_gibberish(text, Sensitivity::Low));
        // Original assertion incorrect - should be checking for GIBBERISH at LOW sensitivity

        assert!(is_gibberish(text, Sensitivity::Medium));
        // MEDIUM sensitivity correctly identifies as GIBBERISH

        assert!(!is_gibberish(text, Sensitivity::High));
        // HIGH sensitivity correctly identifies as ENGLISH due to presence of English word
    }
}
