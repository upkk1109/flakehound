@AGENTS.md

# Claude-specific

- Tiering: Fable/Opus plan & review; Sonnet implements rules/tests; Haiku verifies claims
  (e.g. "does ruff already cover this?" checks). Design starts high, execution starts low.
- This is a PUBLIC repo: never commit secrets, tokens, absolute local paths, or references to
  the owner's other private projects. Reference research by citation (paper/venue), not by
  internal note paths.
- Reference material (read-only, never write there): `~/Projects/AlgoTrading` test-isolation
  history and `kimi-colab-context/data/context/projects/devtools/RESEARCH-RESULT-flakehound.md`.
- GitHub via `gh` CLI. CI is GitHub Actions (free for public repos — sanctioned for this repo).
