const { test } = require("node:test");
const assert = require("node:assert");
const { isValidBrackets } = require("/workspace/solution.js");

test("validates bracket nesting", () => {
  assert.strictEqual(isValidBrackets("([{}])"), true);
  assert.strictEqual(isValidBrackets("([)]"), false);
  assert.strictEqual(isValidBrackets(""), true);
});
