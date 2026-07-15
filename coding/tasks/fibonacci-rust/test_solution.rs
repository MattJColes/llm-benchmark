#[path = "/workspace/solution.rs"]
mod solution;

use solution::fib;

#[test]
fn fib_sequence() {
    assert_eq!(fib(0), 0);
    assert_eq!(fib(1), 1);
    assert_eq!(fib(2), 1);
    assert_eq!(fib(10), 55);
}
