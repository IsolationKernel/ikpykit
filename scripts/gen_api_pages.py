"""Generate the API reference pages, their navigation and an overview index.

Every public estimator used to need a one-line stub under ``docs/api/`` and a
matching entry in the ``nav`` of ``mkdocs.yml``. Both were easy to forget, and a
missing one failed silently: the class simply had no page. What is public, and
how each class describes itself, is read from the package by estimators.py.

Run by mkdocs-gen-files during the build; the pages exist only in the built site.
"""

from __future__ import annotations

import json
import pathlib
import sys
import textwrap

import mkdocs_gen_files
from mkdocs.structure.files import InclusionLevel

# mkdocs-gen-files runs this through runpy, which does not put the script's own
# directory on the path the way running it as a program would.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from estimators import (  # noqa: E402
    API_ROOT,
    ESTIMATORS,
    SECTIONS,
    Section,
    check_complete,
    collect,
    page_path,
    summarize_text,
)


def write_page(module: str, name: str, description: str) -> None:
    """Write the stub that mkdocstrings expands into the rendered class page."""
    path = page_path(module, name)
    with mkdocs_gen_files.open(f"{API_ROOT}/{path}", "w") as page:
        # Drives the meta description, the social card and the llms.txt entry.
        # Quoting through JSON keeps a description containing ": " or a quote
        # from breaking the front matter.
        page.write(f"---\ndescription: {json.dumps(description)}\n---\n\n")
        page.write(f"::: {module}.{name}\n")


def render_nav(section: Section, depth: int = 0) -> list[str]:
    """Render one section of the literate-nav SUMMARY, subsections first."""
    indent = "    " * depth
    lines = [f"{indent}* {section.title}"]
    for subsection in section.subsections:
        lines += render_nav(subsection, depth + 1)
    for module, name, _ in collect(section):
        lines.append(f"{indent}    * [{name}]({page_path(module, name)})")
    return lines


def render_overview(section: Section, level: int = 2) -> list[str]:
    """Render one section of the overview page as a table of its estimators.

    The publication column only exists for estimators the READMEs list, so
    dataset loaders get the two-column form.
    """
    entries = collect(section)
    described = all(name in ESTIMATORS for _, name, _ in entries)
    header = ["Estimator", "Description"] + (["Publication"] if described else [])
    lines = [
        f"{'#' * level} {section.title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for module, name, description in entries:
        row = [f"[{name}]({page_path(module, name)})", description]
        if described:
            row.append(", ".join(ESTIMATORS[name].publications))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for subsection in section.subsections:
        lines += render_overview(subsection, level + 1)
    return lines


def main() -> None:
    check_complete()

    for section in SECTIONS:
        for subsection in (section, *section.subsections):
            for module, name, description in collect(subsection):
                write_page(module, name, description)

    with mkdocs_gen_files.open(f"{API_ROOT}/SUMMARY.md", "w") as summary:
        summary.write("* [Overview](index.md)\n")
        for section in SECTIONS:
            summary.write("\n".join(render_nav(section)) + "\n")

    # literate-nav reads this file and then only marks it as not-in-nav, so it
    # would still be built, land in the sitemap and turn up in site search as a
    # bare list of links. It is an input to the build, not a page.
    editor = mkdocs_gen_files.FilesEditor.current()
    nav_file = editor.files.get_file_from_path(f"{API_ROOT}/SUMMARY.md")
    nav_file.inclusion = InclusionLevel.EXCLUDED

    intro = (
        "Every estimator IKPyKit exports, grouped by the kind of data it works "
        "on. They all follow the scikit-learn API, so each one is constructed "
        "with its parameters and then fitted."
    )
    overview = [
        "---",
        f"description: {json.dumps(summarize_text(intro))}",
        "---",
        "",
        "# API Reference",
        "",
        textwrap.fill(intro, width=79),
        "",
    ]
    for section in SECTIONS:
        overview += render_overview(section)

    with mkdocs_gen_files.open(f"{API_ROOT}/index.md", "w") as index:
        index.write("\n".join(overview))


main()
