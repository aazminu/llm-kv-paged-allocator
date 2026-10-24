# Changelog — `llm-kv-paged-allocator`

All notable engineering changes to `llm-kv-paged-allocator` (`kv_paged`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m kv_paged --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/kv_paged/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/kv_paged/__main__.py`.
- Exported diagnostic helpers in `kv_paged.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: PagedAttention-style LLM KV-cache block allocator with Radix-tree prefix caching, reference-counted Copy-on-Write (CoW), and water-level preemption.
