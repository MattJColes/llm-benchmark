# Tool-calling scenario approval

Approved for structured-agent sweeps.

- Typed tools: `search_files`, `read_file`, `calculator`, `search_web`
- Intentional no-call case: present
- Single-tool selection: present
- Exact argument extraction: present
- Similar-name selection (`search_files` vs `search_web`): present
- Multi-step chain with supplied first-call result: present
- Manifest references and IDs: schema validated
- Exact call/order/argument scoring: covered by tests

Validation: `tests/test_structured.py` — 5 passed.
