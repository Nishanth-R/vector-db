//! Token generation and verification.
//!
//! Tokens are machine-generated, high-entropy secrets rather than
//! user-chosen passwords, so a fast keyed hash (`blake3`) with
//! constant-time comparison is the right choice — Argon2's work factor
//! buys nothing against a 256-bit random preimage and would add latency to
//! every request.

use rand::Rng;

pub const TOKEN_PREFIX: &str = "mara_pat_";
const SUFFIX_LEN: usize = 43;
const BASE62_ALPHABET: &[u8] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Generates a new plaintext token of the form `mara_pat_<43 base62 chars>`,
/// carrying `43 * log2(62) ≈ 256.0` bits of entropy. The plaintext is meant
/// to be displayed exactly once, at creation, and never stored.
pub fn generate_token() -> String {
    let mut rng = rand::thread_rng();
    let suffix: String = (0..SUFFIX_LEN)
        .map(|_| BASE62_ALPHABET[rng.gen_range(0..BASE62_ALPHABET.len())] as char)
        .collect();
    format!("{TOKEN_PREFIX}{suffix}")
}

/// Blake3 hash of a token's bytes — what's actually persisted.
pub fn hash_token(token: &str) -> [u8; 32] {
    blake3::hash(token.as_bytes()).into()
}

/// Constant-time comparison of a candidate token against a stored hash.
pub fn verify_token(token: &str, stored_hash: &[u8; 32]) -> bool {
    let candidate = hash_token(token);
    constant_time_eq(&candidate, stored_hash)
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// The masked form shown in audit entries and logs — `mara_pat_ab12…` — so
/// the full secret never appears outside the moment of creation.
pub fn display_prefix(token: &str) -> String {
    let after_prefix = token.strip_prefix(TOKEN_PREFIX).unwrap_or(token);
    let visible = after_prefix.chars().take(4).collect::<String>();
    format!("{TOKEN_PREFIX}{visible}\u{2026}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_token_has_expected_shape() {
        let t = generate_token();
        assert!(t.starts_with(TOKEN_PREFIX));
        assert_eq!(t.len(), TOKEN_PREFIX.len() + SUFFIX_LEN);
        assert!(t[TOKEN_PREFIX.len()..]
            .bytes()
            .all(|b| BASE62_ALPHABET.contains(&b)));
    }

    #[test]
    fn tokens_are_not_repeated() {
        let a = generate_token();
        let b = generate_token();
        assert_ne!(a, b);
    }

    #[test]
    fn verify_accepts_correct_and_rejects_wrong_token() {
        let token = generate_token();
        let hash = hash_token(&token);
        assert!(verify_token(&token, &hash));
        assert!(!verify_token(&generate_token(), &hash));
    }

    #[test]
    fn display_prefix_never_leaks_the_full_token() {
        let token = generate_token();
        let shown = display_prefix(&token);
        assert!(shown.starts_with(TOKEN_PREFIX));
        assert!(!shown.contains(&token[TOKEN_PREFIX.len() + 4..]));
        assert!(shown.len() < token.len());
    }
}
