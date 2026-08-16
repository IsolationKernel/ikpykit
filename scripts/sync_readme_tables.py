"""Write the estimator tables into both READMEs, or check that they are current.

The root README is read on GitHub and never passes through MkDocs, so a plugin
cannot reach it. Instead both files carry the tables between markers and this
script fills them in from scripts/estimators.py. Run with --check by pre-commit,
which fails if either file has fallen behind.

    python scripts/sync_readme_tables.py           # rewrite both READMEs
    python scripts/sync_readme_tables.py --check   # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from estimators import API_ROOT, check_complete, page_path, render_tables  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED ALGORITHMS -->"
END = "<!-- END GENERATED ALGORITHMS -->"

SITE = "https://isolationkernel.github.io/ikpykit/latest"


def site_link(module: str, name: str) -> str:
    """Absolute link for the root README, which is read outside the site."""
    return f"{SITE}/{API_ROOT}/{page_path(module, name)}".replace(".md", ".html")


def docs_link(module: str, name: str) -> str:
    """Relative link for docs/README.md, which MkDocs resolves and checks."""
    return f"./{API_ROOT}/{page_path(module, name)}"


TARGETS = {
    pathlib.Path("README.md"): site_link,
    pathlib.Path("docs/README.md"): docs_link,
}


def bounds(text: str, path: pathlib.Path) -> tuple[int, int]:
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1:
        raise SystemExit(f"{path}: missing the {BEGIN} / {END} markers.")
    return start, end


def normalize(block: str) -> str:
    """Collapse the whitespace used to align table pipes.

    Editors and Markdown formatters pad table cells so the pipes line up, which
    is how these tables were kept before. Comparing normalized forms means a
    padded table still counts as up to date, rather than this script and a
    formatter overwriting each other on every commit.
    """
    return "\n".join(
        " ".join(line.split()) for line in block.splitlines() if line.strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if a README is out of date",
    )
    args = parser.parse_args()

    check_complete()

    stale = []
    for path, link in TARGETS.items():
        current = path.read_text()
        start, end = bounds(current, path)
        body = render_tables(link)
        if normalize(current[start + len(BEGIN) : end]) == normalize(body):
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_text(f"{current[:start]}{BEGIN}\n\n{body}\n\n{current[end:]}")
            print(f"updated {path}")

    if stale:
        names = ", ".join(str(p) for p in stale)
        print(
            f"{names} out of date. Run `python {__file__}` to regenerate.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
