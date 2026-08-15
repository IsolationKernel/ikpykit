# Changelog

All significant changes to this project are documented in this release file.

| Legend                                                     |                                       |
| :--------------------------------------------------------- | :------------------------------------ |
| <span class="badge text-bg-feature">Feature</span>         | New feature                           |
| <span class="badge text-bg-enhancement">Enhancement</span> | Improvement in existing functionality |
| <span class="badge text-bg-api-change">API Change</span>   | Changes in the API                    |
| <span class="badge text-bg-danger">Fix</span>              | Bug fix                               |

## Unreleased
- <span class="badge text-bg-api-change">API Change</span> Minimum versions of `numba`, `numpy`, and `scikit-learn` were raised to 0.63.0, 2.3.2, and 1.7.2. The previous bounds claimed support for versions that publish no wheels for Python 3.13 or 3.14, so the declared floor could not actually be installed across the supported Python range. A scheduled CI job now resolves against these bounds to keep them honest.
- <span class="badge text-bg-feature">Feature</span> Python 3.14 is now supported and covered by CI on Linux, macOS, and Windows. Installs on 3.14 already worked, since the runtime requirements are lower bounds, but the version was neither tested nor declared.
- <span class="badge text-bg-api-change">API Change</span> The minimum supported Python version is now 3.11 (previously 3.9). Python 3.9 reached end of life in October 2025, and NumPy and scikit-learn have already dropped support for it under SPEC 0 — installs on 3.9 and 3.10 silently resolved to noticeably older versions of the scientific stack.
- <span class="badge text-bg-enhancement">Enhancement</span> Type annotations were modernized to PEP 604 syntax (`X | Y`, `X | None`) now that the minimum version allows it.
- <span class="badge text-bg-enhancement">Enhancement</span> All `zip()` calls over collections that are equal-length by construction now pass `strict=True`, so a length mismatch raises instead of silently truncating.
- <span class="badge text-bg-danger">Fix</span> Locked development dependencies were upgraded to clear 76 security advisories. These were confined to the docs and dev dependency groups and never reached the published package, whose runtime requirements are unchanged.

## Released (2026-03-22, v0.2.4 Hotfix)
- <span class="badge text-bg-danger">Fix</span> Converted hard-coded site links in the docs home page to version-safe relative links, so navigation works under `latest`, `dev`, and versioned routes.

## Released (2026-03-22)
- <span class="badge text-bg-feature">Feature</span> IDKC now supports optional `force_assign_unassigned` to force final assignment of remaining `-1` samples.
- <span class="badge text-bg-enhancement">Enhancement</span> CI/CD workflows were unified around `uv` with reusable pipelines for lint, tests, and docs checks.
- <span class="badge text-bg-enhancement">Enhancement</span> Installation guidance was standardized to recommend `uv pip install`, with classic `pip` retained as fallback.
- <span class="badge text-bg-enhancement">Enhancement</span> A top-level CI entry workflow was introduced to run lint, tests, and docs validation in a consistent matrix-driven flow.
- <span class="badge text-bg-enhancement">Enhancement</span> Docs deployment was refactored to support branch/tag version publishing using `mike` with `dev` and `latest` aliases.
- <span class="badge text-bg-enhancement">Enhancement</span> PyPI release workflow was modernized with `uv build`, OIDC publishing, and GitHub Release artifact upload.
- <span class="badge text-bg-enhancement">Enhancement</span> Shared cluster bookkeeping was consolidated in `KCluster` (`build_labels`, `total_points`) and reused by IDKC/PSKC.
- <span class="badge text-bg-enhancement">Enhancement</span> IDKC and PSKC internals were refactored into clearer helper methods without changing their core algorithmic intent.
- <span class="badge text-bg-enhancement">Enhancement</span> Project formatting and linting gates were strengthened by integrating `black` with existing `ruff` checks.
- <span class="badge text-bg-danger">Fix</span> Notebook web rendering reliability was improved by enabling `mkdocs-jupyter` execution and validating notebook build in strict docs checks.
- <span class="badge text-bg-danger">Fix</span> Documentation build failures caused by notebook runtime dependencies were resolved by adding docs-time plotting dependencies.
- <span class="badge text-bg-danger">Fix</span> User guide notebook example import paths were corrected to match actual public API symbols.

- <span class="badge text-bg-danger">Fix</span> Multiple Ruff and warning issues were resolved across anomaly, kernel, stream, graph, and utility modules.
- <span class="badge text-bg-danger">Fix</span> Pytest warning noise was reduced via dependency/config updates (`pytest-asyncio`, `pytest-timeout`) and timeout tuning.

- <span class="badge text-bg-danger">Fix</span> IDKC post-processing logic for unassigned points was corrected to prevent invalid cluster transitions.

- <span class="badge text-bg-danger">Fix</span> Graph test sparse matrix construction was optimized to avoid efficiency warnings (`lil` construction then `csr` conversion).
- <span class="badge text-bg-danger">Fix</span> Time-series test regex matching for exception messages was fixed using escaped expected patterns.

- <span class="badge text-bg-danger">Fix</span> FAQ content was rewritten for IKPyKit and reconnected to site navigation.
- <span class="badge text-bg-danger">Fix</span> Multiple documentation links and navigation targets were corrected to pass `mkdocs --strict`.
