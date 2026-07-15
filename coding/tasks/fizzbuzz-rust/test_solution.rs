#[path = "/workspace/solution.rs"]
mod solution;

use solution::fizzbuzz;

#[test]
fn fizzbuzz_sequence() {
    assert_eq!(
        fizzbuzz(3),
        vec!["1".to_string(), "2".to_string(), "Fizz".to_string()]
    );
    assert_eq!(fizzbuzz(5)[4], "Buzz");
    assert_eq!(fizzbuzz(15)[14], "FizzBuzz");
}
