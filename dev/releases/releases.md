# Changelog

All significant changes to this project are documented in this release file.

| Legend                                                     |                                       |
| :--------------------------------------------------------- | :------------------------------------ |
| <span class="badge text-bg-feature">Feature</span>         | New feature                           |
| <span class="badge text-bg-enhancement">Enhancement</span> | Improvement in existing functionality |
| <span class="badge text-bg-api-change">API Change</span>   | Changes in the API                    |
| <span class="badge text-bg-danger">Fix</span>              | Bug fix                               |

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
