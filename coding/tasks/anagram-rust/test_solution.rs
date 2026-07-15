#[path = "/workspace/solution.rs"]
mod solution;

use solution::is_anagram;

#[test]
fn detects_anagrams() {
    assert!(is_anagram("Listen", "Silent"));
    assert!(!is_anagram("hello", "world"));
    assert!(is_anagram("a", "a"));
}
