# Changelog

All significant changes to this project are documented in this release file.

| Legend      |                                       |
| ----------- | ------------------------------------- |
| Feature     | New feature                           |
| Enhancement | Improvement in existing functionality |
| API Change  | Changes in the API                    |
| Fix         | Bug fix                               |

## Released (2026-08-16, v0.3.0)

This release raises the minimum Python version and the minimum versions of several runtime dependencies. Both are breaking for anyone on an older environment; everything else is additive or internal.

- Feature Python 3.14 is now supported and covered by CI on Linux, macOS, and Windows. Installs on 3.14 already worked, since the runtime requirements are lower bounds, but the version was neither tested nor declared.
- API Change The minimum supported Python version is now 3.11 (previously 3.9). Python 3.9 reached end of life in October 2025, and NumPy and scikit-learn have already dropped support for it under SPEC 0 — installs on 3.9 and 3.10 silently resolved to noticeably older versions of the scientific stack.
- API Change Minimum versions of `numba`, `numpy`, and `scikit-learn` were raised to 0.63.0, 2.3.2, and 1.7.2. The previous bounds claimed support for versions that publish no wheels for Python 3.13 or 3.14, so the declared floor could not actually be installed across the supported Python range. A scheduled CI job now resolves against these bounds to keep them honest.
- Enhancement Type annotations were modernized to PEP 604 syntax (`X | Y`, `X | None`) now that the minimum version allows it.
- Enhancement All `zip()` calls over collections that are equal-length by construction now pass `strict=True`, so a length mismatch raises instead of silently truncating.
- Fix Spelling errors were corrected in user-facing documentation, including IKTOD's class docstring and a variable name in the IKGOD usage example.
- Fix Locked development dependencies were upgraded, clearing all open security advisories. These were confined to the docs and dev dependency groups and never reached the published package, whose runtime requirements are unaffected.

## Released (2026-03-22, v0.2.4 Hotfix)

- Fix Converted hard-coded site links in the docs home page to version-safe relative links, so navigation works under `latest`, `dev`, and versioned routes.

## Released (2026-03-22)

- Feature IDKC now supports optional `force_assign_unassigned` to force final assignment of remaining `-1` samples.
- Enhancement CI/CD workflows were unified around `uv` with reusable pipelines for lint, tests, and docs checks.
- Enhancement Installation guidance was standardized to recommend `uv pip install`, with classic `pip` retained as fallback.
- Enhancement A top-level CI entry workflow was introduced to run lint, tests, and docs validation in a consistent matrix-driven flow.
- Enhancement Docs deployment was refactored to support branch/tag version publishing using `mike` with `dev` and `latest` aliases.
- Enhancement PyPI release workflow was modernized with `uv build`, OIDC publishing, and GitHub Release artifact upload.
- Enhancement Shared cluster bookkeeping was consolidated in `KCluster` (`build_labels`, `total_points`) and reused by IDKC/PSKC.
- Enhancement IDKC and PSKC internals were refactored into clearer helper methods without changing their core algorithmic intent.
- Enhancement Project formatting and linting gates were strengthened by integrating `black` with existing `ruff` checks.
- Fix Notebook web rendering reliability was improved by enabling `mkdocs-jupyter` execution and validating notebook build in strict docs checks.
- Fix Documentation build failures caused by notebook runtime dependencies were resolved by adding docs-time plotting dependencies.
- Fix User guide notebook example import paths were corrected to match actual public API symbols.
- Fix Multiple Ruff and warning issues were resolved across anomaly, kernel, stream, graph, and utility modules.
- Fix Pytest warning noise was reduced via dependency/config updates (`pytest-asyncio`, `pytest-timeout`) and timeout tuning.
- Fix IDKC post-processing logic for unassigned points was corrected to prevent invalid cluster transitions.
- Fix Graph test sparse matrix construction was optimized to avoid efficiency warnings (`lil` construction then `csr` conversion).
- Fix Time-series test regex matching for exception messages was fixed using escaped expected patterns.
- Fix FAQ content was rewritten for IKPyKit and reconnected to site navigation.
- Fix Multiple documentation links and navigation targets were corrected to pass `mkdocs --strict`.
