const { test } = require("node:test");
const assert = require("node:assert");
const { dedupe } = require("/workspace/solution.js");

test("removes duplicates preserving order", () => {
  assert.deepStrictEqual(dedupe([1, 2, 2, 3]), [1, 2, 3]);
  assert.deepStrictEqual(dedupe([]), []);
  assert.deepStrictEqual(dedupe(["a", "a", "b"]), ["a", "b"]);
});
