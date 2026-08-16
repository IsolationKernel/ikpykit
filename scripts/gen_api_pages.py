"""Generate the API reference pages, their navigation and an overview index.

Every public estimator used to need a one-line stub under ``docs/api/`` and a
matching entry in the ``nav`` of ``mkdocs.yml``. Both were easy to forget, and a
missing one failed silently: the class simply had no page. The package already
states what is public, through the ``__all__`` of each module, so that is what
this script reads.

The only thing it cannot derive is the section names below, which are editorial,
and their order, which is meaningful. Adding an estimator to a module that is
already listed needs no change here.

Run by mkdocs-gen-files during the build; the pages exist only in the built site.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import textwrap
from dataclasses import dataclass, field

import mkdocs_gen_files
from mkdocs.structure.files import InclusionLevel

# Longest description to put in front matter. Social cards and search engines
# both cut off around here, so a longer one is only truncated elsewhere.
DESCRIPTION_LIMIT = 160

API_ROOT = "api"


@dataclass(frozen=True)
class Section:
    """A group of estimators shown under one heading in the navigation."""

    title: str
    module: str
    subsections: tuple[Section, ...] = field(default_factory=tuple)


SECTIONS = (
    Section("Isolation Kernel", "ikpykit.kernel"),
    Section("Point Anomaly Detection", "ikpykit.anomaly"),
    Section("Point Clustering", "ikpykit.cluster"),
    Section("Graph Mining", "ikpykit.graph"),
    Section("Group Mining", "ikpykit.group"),
    Section("Stream Mining", "ikpykit.stream"),
    Section(
        "Trajectory Mining",
        "ikpykit.trajectory",
        subsections=(Section("DataLoader", "ikpykit.trajectory.dataloader"),),
    ),
    Section("Time Series Mining", "ikpykit.timeseries"),
)


def summarize_text(text: str) -> str:
    """Collapse text to a single line, keeping whole sentences within the limit.

    Text longer than the limit is cut at the last sentence that still fits,
    rather than mid-word.
    """
    line = " ".join(text.split())
    if len(line) <= DESCRIPTION_LIMIT:
        return line

    kept = ""
    for sentence in re.findall(r"[^.]*\.(?:\s|$)", line):
        if len(kept) + len(sentence) > DESCRIPTION_LIMIT:
            break
        kept += sentence
    # A first sentence that is itself over the limit leaves nothing to keep.
    return kept.strip() or line[:DESCRIPTION_LIMIT].rstrip()


def summarize(obj: object) -> str:
    """Return the first paragraph of an object's docstring as a single line."""
    doc = inspect.getdoc(obj) or ""
    return summarize_text(doc.split("\n\n")[0])


def page_path(module: str, name: str) -> str:
    """Return the doc path for a class, mirroring its location in the package."""
    package = module.removeprefix("ikpykit.").replace(".", "/")
    return f"{package}/{name.lower()}.md"


def write_page(module: str, name: str, description: str) -> None:
    """Write the stub that mkdocstrings expands into the rendered class page."""
    path = page_path(module, name)
    with mkdocs_gen_files.open(f"{API_ROOT}/{path}", "w") as page:
        # Drives the meta description, the social card and the llms.txt entry.
        # Quoting through JSON keeps a description containing ": " or a quote
        # from breaking the front matter.
        page.write(f"---\ndescription: {json.dumps(description)}\n---\n\n")
        page.write(f"::: {module}.{name}\n")


def collect(section: Section) -> list[tuple[str, str, str]]:
    """Return (module, name, description) for every public class in a section."""
    module = importlib.import_module(section.module)
    entries = []
    for name in module.__all__:
        obj = getattr(module, name)
        entries.append((section.module, name, summarize(obj)))
    return entries


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
    """Render one section of the overview page as a table of its estimators."""
    lines = [
        f"{'#' * level} {section.title}",
        "",
        "| Estimator | Description |",
        "| --- | --- |",
    ]
    for module, name, description in collect(section):
        lines.append(f"| [{name}]({page_path(module, name)}) | {description} |")
    lines.append("")
    for subsection in section.subsections:
        lines += render_overview(subsection, level + 1)
    return lines


def main() -> None:
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
