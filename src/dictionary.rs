use phf::phf_set;

pub static ENGLISH_WORDS: phf::Set<&'static str> = phf_set! {
    // This will be populated by the dictionary generator
};
