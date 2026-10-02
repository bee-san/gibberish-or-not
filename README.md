<div align="center">

# 🔍 Gibberish Detection Tool

**Instantly detect if text is English or nonsense with 99% accuracy**

[![Crates.io](https://img.shields.io/crates/v/gibberish-or-not.svg)](https://crates.io/crates/gibberish-or-not)
[![Documentation](https://docs.rs/gibberish-or-not/badge.svg)](https://docs.rs/gibberish-or-not)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Documentation](https://docs.rs/gibberish-or-not) |
[Examples](#examples) |
[Contributing](#contributing)

</div>

## ⚡ Quick Install

```bash
# As a CLI tool
cargo install gibberish-or-not

# As a library in Cargo.toml
gibberish-or-not = "4.1.1"
```

<<<<<<< HEAD
=======
<<<<<<< Updated upstream
## 🎯 Examples
=======
>>>>>>> main
## 🤖 Enhanced Detection with BERT

The library offers enhanced detection using a BERT model for more accurate results on borderline cases. To use enhanced detection:

1. Set up HuggingFace authentication (one of two methods):

   **Method 1: Environment Variable**
   ```bash
   # Set the token in your environment
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

   **Method 2: Direct Token**
   ```rust
   use gibberish_or_not::{download_model_with_progress_bar, default_model_path};
   
   // Pass token directly to the download function
   download_model_with_progress_bar(default_model_path(), Some("your_token_here"))?;
   ```

   Get your token by:
   1. Creating an account at https://huggingface.co
   2. Generating a token at https://huggingface.co/settings/tokens

2. Download the model (choose one method):
   ```bash
   # Using the CLI (uses environment variable)
   cargo run --bin download_model

   # Or in your code (using direct token)
   use gibberish_or_not::{download_model_with_progress_bar, default_model_path};
   download_model_with_progress_bar(default_model_path(), Some("your_token_here"))?;
   ```

3. Use enhanced detection in your code:
   ```rust
<<<<<<< HEAD
   use gibberish_or_not::{GibberishDetector, Sensitivity, default_model_path};
   
   // Create detector with model
   let detector = GibberishDetector::with_model(default_model_path());
   
   // Check if enhanced detection is available
=======
   // Basic usage - automatically uses enhanced detection if model exists
   use gibberish_or_not::{is_gibberish, Sensitivity};
   
   // The function automatically checks for model at default_model_path()
   let result = is_gibberish("Your text here", Sensitivity::Medium);
   
   // Optional: Explicit model control
   use gibberish_or_not::{GibberishDetector, default_model_path};
   
   // Create detector with model - useful if you want to:
   // - Use a custom model path
   // - Check if enhanced detection is available
   // - Reuse the same model instance
   let detector = GibberishDetector::with_model(default_model_path());
>>>>>>> main
   if detector.has_enhanced_detection() {
       let result = detector.is_gibberish("Your text here", Sensitivity::Medium);
   }
   ```

<<<<<<< HEAD
=======
Note: The basic detection algorithm will be used as a fallback if the model is not available. The model is automatically loaded from the default path (`default_model_path()`) when using the simple `is_gibberish` function.

>>>>>>> main
You can also check the token status programmatically:
```rust
use gibberish_or_not::{check_token_status, TokenStatus, default_model_path};

match check_token_status(default_model_path()) {
    TokenStatus::Required => println!("HuggingFace token needed"),
    TokenStatus::Available => println!("Token found, ready to download"),
    TokenStatus::NotRequired => println!("Model exists, no token needed"),
}
```

<<<<<<< HEAD
Note: The basic detection algorithm will be used as a fallback if the model is not available.

## �� Examples
=======
##  Examples
>>>>>>> Stashed changes
>>>>>>> main

```rust
use gibberish_or_not::{is_gibberish, is_password, Sensitivity};

// Password Detection
assert!(is_password("123456"));  // Detects common passwords

// Valid English
assert!(!is_gibberish("The quick brown fox jumps over the lazy dog", Sensitivity::Medium));
assert!(!is_gibberish("Hello, world!", Sensitivity::Medium));

// Gibberish
assert!(is_gibberish("asdf jkl qwerty", Sensitivity::Medium));
assert!(is_gibberish("xkcd vwpq mntb", Sensitivity::Medium));
assert!(is_gibberish("println!({});", Sensitivity::Medium)); // Code snippets are classified as gibberish
```

## 🔬 How It Works

Our advanced detection algorithm uses multiple components:

### 1. 📚 Dictionary Analysis
- **370,000+ English words** compiled into the binary
- Perfect hash table for O(1) lookups
- Zero runtime loading overhead
- Includes technical terms and proper nouns

### 2. 🧮 N-gram Analysis
- **Trigrams** (3-letter sequences)
- **Quadgrams** (4-letter sequences)
- Trained on massive English text corpus
- Weighted scoring system

### 3. 🎯 Smart Classification
- Composite scoring system combining:
  - English word ratio (40% weight)
  - Character transition probability (25% weight)
  - Trigram analysis (15% weight)
  - Quadgram analysis (10% weight)
  - Vowel-consonant ratio (10% weight)
- Length-based threshold adjustment
- Special case handling for:
  - Very short text (<10 chars)
  - Non-printable characters
  - Code snippets
  - URLs and technical content

## 🎚️ Sensitivity Levels

The library provides three sensitivity levels:

### High Sensitivity
- Most lenient classification
- Easily accepts text as English
- Best for minimizing false negatives (English classified as gibberish)
- Use when: You want to catch anything remotely English-like

### Medium Sensitivity (Default)
- Balanced approach
- Suitable for general text classification
- Reliable for most use cases
- Use when: You want general-purpose gibberish detection

### Low Sensitivity
- Most strict classification
- Requires strong evidence of English
- Best for security applications
- Use when: False positives are costly

## 🔑 Password Detection

Built-in detection of common passwords:

```rust
use gibberish_or_not::is_password;

assert!(is_password("123456"));     // Common password
assert!(is_password("password"));   // Common password
assert!(!is_password("unique_and_secure_passphrase")); // Not in common list
```

## 🎯 Special Cases

The library handles various special cases:

- Code snippets are classified as gibberish
- URLs in text are preserved for analysis
- Technical terms and abbreviations are recognized
- Mixed-language content is supported
- ASCII art is detected as gibberish
- Common internet text patterns are recognized

## 🧮 Algorithm Deep Dive

The gibberish detection algorithm combines multiple scoring components into a weighted composite score. Here's a detailed look at each component:

### Composite Score Formula

The final classification uses a weighted sum:

$S = 0.4E + 0.25T + 0.15G_3 + 0.1G_4 + 0.1V$

Where:
- $E$ = English word ratio
- $T$ = Character transition probability
- $G_3$ = Trigram score
- $G_4$ = Quadgram score
- $V$ = Vowel-consonant ratio (binary: 1 if in range [0.3, 0.7], 0 otherwise)

### Length-Based Threshold Adjustment

The threshold is dynamically adjusted based on text length:

```rust
let threshold = match text_length {
    0..=20  => 0.7,  // Very short text needs higher threshold
    21..=50 => 0.8,  // Short text
    51..=100 => 0.9, // Medium text
    101..=200 => 1.0,// Standard threshold
    _ => 1.1,        // Long text can be more lenient
} * sensitivity_factor;
```

### Character Entropy

We calculate Shannon entropy to measure randomness:

$H = -\sum_{i} p_i \log_2(p_i)$

Where $p_i$ is the probability of character $i$ occurring in the text.

```rust
let entropy = char_frequencies.iter()
    .map(|p| -p * p.log2())
    .sum::<f64>();
```

### N-gram Analysis

Trigrams and quadgrams are scored using frequency analysis:

$G_n = \frac{\text{valid n-grams}}{\text{total n-grams}}$

```rust
let trigram_score = valid_trigrams.len() as f64 / total_trigrams.len() as f64;
let quadgram_score = valid_quadgrams.len() as f64 / total_quadgrams.len() as f64;
```

### Character Transition Probability

We analyze character pair frequencies against known English patterns:

$T = \frac{\text{valid transitions}}{\text{total transitions}}$

The transition matrix is pre-computed from a large English corpus and stored as a perfect hash table.

### Sensitivity Levels

The final threshold varies by sensitivity:
- Low: $0.35 \times \text{length\_factor}$
- Medium: $0.25 \times \text{length\_factor}$
- High: $0.15 \times \text{length\_factor}$

### Special Case Overrides

The algorithm includes fast-path decisions:

1. If English word ratio > 0.8: Not gibberish
2. If ≥ 3 English words (Medium/High sensitivity): Not gibberish
3. If no English words AND transition score < 0.4 (Low/Medium): Gibberish

### Why These Weights?

- **Word Ratio (40%)**: Strong indicator of English text
- **Transitions (25%)**: Captures natural language patterns
- **Trigrams (15%)**: Common subword patterns
- **Quadgrams (10%)**: Longer patterns, but noisier
- **Vowel Ratio (10%)**: Basic language structure

This weighting balances accuracy with computational efficiency, prioritizing stronger indicators while still considering multiple aspects of language structure.

## ⚡ Performance

The library is optimized for speed, with benchmarks showing excellent performance across different text types:

### Basic Detection Speed (without BERT)

| Text Length | Processing Time |
|------------|----------------|
| Short (10-20 chars) | 2.3-2.7 μs |
| Medium (20-50 chars) | 4-7 μs |
| Long (50-100 chars) | 7-15 μs |
| Very Long (200+ chars) | ~50 μs |

### Enhanced Detection Speed (with BERT)

| Text Length | First Run* | Subsequent Runs |
|------------|------------|-----------------|
| Short (10-20 chars) | ~100ms | 5-10ms |
| Medium (20-50 chars) | ~100ms | 5-15ms |
| Long (50-100 chars) | ~100ms | 10-20ms |
| Very Long (200+ chars) | ~100ms | 15-30ms |

*First run includes model loading time. The model is cached after first use.

### Sensitivity Level Impact (Basic Detection)

| Sensitivity | Processing Time |
|------------|----------------|
| Low | ~7.3 μs |
| Medium | ~6.7 μs |
| High | ~7.9 μs |

These benchmarks were run on a modern CPU using the Criterion benchmarking framework. The library achieves this performance through:

- Perfect hash tables for O(1) dictionary lookups
- Pre-computed n-gram tables
- Optimized character transition matrices
- Early-exit optimizations for clear cases
- Zero runtime loading overhead
- Memory-mapped BERT model loading
- Model result caching

### Memory Usage

- Basic Detection: < 1MB
- Enhanced Detection: ~400-500MB (BERT model, memory-mapped)

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and request features
- Improve documentation
- Submit pull requests
- Add test cases

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

