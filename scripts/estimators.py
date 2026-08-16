"""The single description of what IKPyKit implements.

The estimator tables existed in three places that had drifted apart: the root
``README.md``, ``docs/README.md`` and, since the API reference was generated,
``api/index.md``. This module holds the parts that cannot be read off the code
and renders every one of those tables from them, so adding an estimator means
editing one entry rather than three tables.

What the code already states is not repeated here. The class list comes from the
``__all__`` of each module, and the one-line description from the first
paragraph of its docstring.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass

# Longest description to put in a page's front matter. Social cards and search
# engines both cut off around here, so a longer one is only truncated elsewhere.
DESCRIPTION_LIMIT = 160

API_ROOT = "api"


@dataclass(frozen=True)
class Section:
    """A group of estimators shown under one heading."""

    title: str
    module: str
    subsections: tuple[Section, ...] = ()
    # Dataset loaders belong in the API reference but not in a table of
    # algorithms, so a section can opt out of the README tables.
    in_readme: bool = True


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
        subsections=(
            Section("DataLoader", "ikpykit.trajectory.dataloader", in_readme=False),
        ),
    ),
    Section("Time Series Mining", "ikpykit.timeseries"),
)


@dataclass(frozen=True)
class Estimator:
    """What the tables say about an estimator that the code does not.

    ``data_type`` and ``capability`` place it in the summary matrix; they are the
    row and the column it appears in.
    """

    algorithm: str
    application: str
    publications: tuple[str, ...]
    data_type: str
    capability: str


# Every public class outside a section marked `in_readme=False` needs an entry
# here. `check_complete` enforces that, so a new estimator cannot quietly end up
# missing from the tables the way it used to.
ESTIMATORS = {
    "IsoKernel": Estimator(
        algorithm="Isolation Kernel",
        application="IK feature mapping and similarity calculating",
        publications=("AAAI2019", "SIGKDD2018"),
        data_type="Point Data",
        capability="Kernel Similarity",
    ),
    "IsoDisKernel": Estimator(
        algorithm="Isolation Distribution Kernel",
        application="Distribution similarity calculating",
        publications=("SIGKDD2020",),
        data_type="Group Data",
        capability="Kernel Similarity",
    ),
    "IForest": Estimator(
        algorithm="Isolation forest",
        application="Anomaly Detection",
        publications=("ICDM2008", "TKDD2012"),
        data_type="Point Data",
        capability="Anomaly Detection",
    ),
    "INNE": Estimator(
        algorithm="Isolation-based anomaly detection using nearest-neighbor ensembles",
        application="Anomaly Detection",
        publications=("CIJ2018",),
        data_type="Point Data",
        capability="Anomaly Detection",
    ),
    "IDKD": Estimator(
        algorithm="Isolation Distributional Kernel for point anomaly detections",
        application="Anomaly Detection",
        publications=("TKDE2022",),
        data_type="Point Data",
        capability="Anomaly Detection",
    ),
    "IDKC": Estimator(
        algorithm="Kernel-based Clustering via Isolation Distributional Kernel",
        application="Point Clustering",
        publications=("IS2023",),
        data_type="Point Data",
        capability="Clustering",
    ),
    "PSKC": Estimator(
        algorithm="Point-set Kernel Clustering",
        application="Point Clustering",
        publications=("TKDE2023",),
        data_type="Point Data",
        capability="Clustering",
    ),
    "IKAHC": Estimator(
        algorithm="Isolation Kernel for Agglomerative Hierarchical Clustering",
        application="Hierarchical Clustering",
        publications=("PR2023",),
        data_type="Point Data",
        capability="Clustering",
    ),
    "IKGOD": Estimator(
        algorithm="Subgraph Centralization: A Necessary Step for Graph Anomaly Detection",
        application="Graph Anomaly Detection",
        publications=("SIAM2023",),
        data_type="Graph Data",
        capability="Anomaly Detection",
    ),
    "IsoGraphKernel": Estimator(
        algorithm="Isolation Graph Kernel",
        application="Graph IK embedding and similarity calculating",
        publications=("AAAI2021",),
        data_type="Graph Data",
        capability="Kernel Similarity",
    ),
    "IKGAD": Estimator(
        algorithm="Isolation Distributional Kernel for group anomaly detections",
        application="Group Anomaly Detection",
        publications=("TKDE2022",),
        data_type="Group Data",
        capability="Anomaly Detection",
    ),
    "ICID": Estimator(
        algorithm="Detecting change intervals with isolation distributional kernel",
        application="Change Intervals Detection",
        publications=("JAIR2024",),
        data_type="Stream Data",
        capability="Change Detection",
    ),
    "STREAMKHC": Estimator(
        algorithm="Streaming Hierarchical Clustering Based on Point-Set Kernel",
        application="Online Hierarchical Clustering",
        publications=("SIGKDD2022",),
        data_type="Stream Data",
        capability="Clustering",
    ),
    "IKAT": Estimator(
        algorithm="Isolation Distribution Kernel for Trajectory Anomaly Detections",
        application="Trajectory Anomaly Detection",
        publications=("JAIR2024",),
        data_type="Trajectory Data",
        capability="Anomaly Detection",
    ),
    "TIDKC": Estimator(
        algorithm="Distribution-based Trajectory Clustering",
        application="Trajectory Clustering",
        publications=("ICDM2023",),
        data_type="Trajectory Data",
        capability="Clustering",
    ),
    "IKTOD": Estimator(
        algorithm="Isolation distribution kernel for Time Series Anomaly Detection",
        application="Anomaly Detection",
        publications=("VLDB2022",),
        data_type="Time Series",
        capability="Anomaly Detection",
    ),
}

MATRIX_ROWS = (
    "Point Data",
    "Graph Data",
    "Group Data",
    "Stream Data",
    "Time Series",
    "Trajectory Data",
)
MATRIX_COLUMNS = (
    "Kernel Similarity",
    "Anomaly Detection",
    "Clustering",
    "Change Detection",
)

ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")


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


def collect(section: Section) -> list[tuple[str, str, str]]:
    """Return (module, name, description) for every public class in a section."""
    module = importlib.import_module(section.module)
    return [
        (section.module, name, summarize(getattr(module, name)))
        for name in module.__all__
    ]


def readme_sections() -> list[tuple[Section, list[tuple[str, str, str]]]]:
    """Return the sections that appear in the README tables, with their classes."""
    return [(s, collect(s)) for s in SECTIONS if s.in_readme]


def check_complete() -> None:
    """Fail if an estimator has no table entry, or an entry has no estimator."""
    public = {name for _, entries in readme_sections() for _, name, _ in entries}
    missing = sorted(public - ESTIMATORS.keys())
    if missing:
        raise SystemExit(
            f"No entry in ESTIMATORS for {', '.join(missing)}. Add one in "
            f"{__file__} so the tables include it."
        )
    extra = sorted(ESTIMATORS.keys() - public)
    if extra:
        raise SystemExit(
            f"ESTIMATORS has entries for {', '.join(extra)}, which no module "
            f"exports. Remove them from {__file__}."
        )


def short_publication(publication: str) -> str:
    """Shorten a publication for the summary matrix: AAAI2019 -> AAAI'19."""
    return re.sub(r"^(\D+)\d{2}(\d{2})$", r"\1'\2", publication)


def render_summary_matrix() -> list[str]:
    """Render the matrix of which estimators cover which task on which data."""
    cells: dict[tuple[str, str], list[str]] = {}
    for name, meta in ESTIMATORS.items():
        short = ", ".join(short_publication(p) for p in meta.publications)
        cells.setdefault((meta.data_type, meta.capability), []).append(
            f"{name} ({short})"
        )

    lines = [
        "| Algorithms | " + " | ".join(MATRIX_COLUMNS) + " |",
        "| --- | " + " | ".join("---" for _ in MATRIX_COLUMNS) + " |",
    ]
    for row in MATRIX_ROWS:
        # Several estimators can share a cell; <br> keeps them in one row rather
        # than spilling into blank rows underneath.
        filled = ["<br>".join(cells.get((row, col), [])) for col in MATRIX_COLUMNS]
        lines.append(f"| {row} | " + " | ".join(filled) + " |")
    return lines


def render_tables(link: Callable[[str, str], str]) -> str:
    """Render the whole generated block, linking each class through `link`.

    `link` takes the module and the class name and returns a URL, which differs
    between the two READMEs: one is read on GitHub, the other inside the site.
    """
    lines = ["#### Summary", "", *render_summary_matrix(), ""]
    for number, (section, entries) in zip(ROMAN, readme_sections(), strict=False):
        lines += [
            f"**({number}) {section.title}**:",
            "",
            "| Abbr | Algorithm | Application | Publication |",
            "| --- | --- | --- | --- |",
        ]
        for module, name, _ in entries:
            meta = ESTIMATORS[name]
            lines.append(
                f"| [{name}]({link(module, name)}) | {meta.algorithm} | "
                f"{meta.application} | {', '.join(meta.publications)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
