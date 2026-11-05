//! `gibberish`: is this text natural language? (feature `cli`)
//!
//! ```text
//! gibberish [--sensitivity low|medium|high] [--lang en,fr,...] [TEXT ...]
//! ```
//!
//! Prints `language <code> <score>` or `gibberish <score>` for each TEXT, or for each line
//! of standard input when no TEXT is given. Exits 0 if every input was language, 1 if any
//! was gibberish, 2 on a usage error.

use std::io::BufRead;
use std::process::ExitCode;

use gibberish_or_not::{Detector, Lang, Sensitivity};

fn usage() -> ExitCode {
    eprintln!("usage: gibberish [--sensitivity low|medium|high] [--lang en,fr,...] [TEXT ...]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut detector = Detector::new();
    let mut texts = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-s" | "--sensitivity" => {
                let s = match args.next().as_deref() {
                    Some("low") => Sensitivity::Low,
                    Some("medium") => Sensitivity::Medium,
                    Some("high") => Sensitivity::High,
                    _ => return usage(),
                };
                detector = detector.sensitivity(s);
            }
            "-l" | "--lang" => {
                let Some(list) = args.next() else {
                    return usage();
                };
                let langs: Option<Vec<Lang>> = list.split(',').map(Lang::from_code).collect();
                let Some(langs) = langs else { return usage() };
                detector = detector.languages(&langs);
            }
            "-h" | "--help" => {
                let _ = usage();
                return ExitCode::SUCCESS;
            }
            _ => texts.push(arg),
        }
    }
    if texts.is_empty() {
        texts = std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .collect();
    }
    let mut all_language = true;
    for text in &texts {
        let d = detector.detect(text);
        match d.language {
            Some(lang) if d.is_language => println!("language {lang} {:.3}", d.score),
            _ => {
                all_language = false;
                println!("gibberish {:.3}", d.score);
            }
        }
    }
    if all_language {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
