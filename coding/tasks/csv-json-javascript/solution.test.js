const { test } = require("node:test");
const assert = require("node:assert");
const { csvToJson } = require("/workspace/solution.js");

test("parses csv into objects keyed by header", () => {
  assert.deepStrictEqual(csvToJson("a,b\n1,2"), [{ a: "1", b: "2" }]);
  assert.deepStrictEqual(csvToJson("a,b\n1,2\n3,4"), [
    { a: "1", b: "2" },
    { a: "3", b: "4" },
  ]);
});
