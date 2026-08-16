"""Draw the IKPyKit logo by running Isolation Kernel on a small point set.

The mark is an "ik" monogram over a Voronoi partition, and the partition is not
decoration: it is the output of one aNNE estimator, the same partition the
kernel uses to measure similarity. Cells are shaded by area, and the largest --
the one whose point sits furthest from any neighbour, and so is easiest to
isolate -- is the orange one.

The palette is scikit-learn's blue and orange. IKPyKit implements the
scikit-learn API, and that pair is how the surrounding ecosystem signals it;
the form is a monogram rather than scikit-learn's blobs, so the mark reads as a
sibling rather than as something official.

Nothing is hand-drawn, so the mark regenerates at any size and the palette or
the cell count can be changed by editing the constants below.

    python scripts/gen_logo.py            # write the assets into docs/img/
    python scripts/gen_logo.py --preview  # contact sheet of other candidates
"""

from __future__ import annotations

import argparse
import pathlib

import cairosvg
import numpy as np
from scipy.spatial import Voronoi

from ikpykit.kernel import IsoKernel

OUT_DIR = pathlib.Path("docs/img")
SIZE = 512

# scikit-learn's blue, lightened into a four step ramp. Their blue is a mid
# tone, so the ramp deliberately stops short of navy -- a dark step reads as a
# different family and weighs one corner of the mark down.
BLUE = ("#9ed2ee", "#6bb8e4", "#3499cd", "#2a7fae")
ORANGE = "#f89939"
WHITE = "#ffffff"

# Chosen from the candidates --preview renders, on composition: balanced cells,
# no slivers, and the orange cell clear of the monogram's counters.
CELLS = 4
RANDOM_STATE = 3

# The monogram, drawn as constant weight strokes. The k's arms are one path
# whose vertex sits on the stem's centre line, and the stem is painted after
# them, so the join between arms is hidden rather than spiking out to the left.
STROKE = 52
ARMS = "M 401 178 L 245 290 L 401 392"
K_STEM = "M 245 120 L 245 392"
I_STEM = "M 139 200 L 139 392"
DOT = (139, 150, 31)


def sample_points(random_state: int, n: int = 160) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    centres = rng.uniform(0.2, 0.8, size=(4, 2))
    points = np.vstack([c + rng.randn(n // 4, 2) * 0.13 for c in centres])
    return np.clip(points, 0.06, 0.94)


def partition(n_cells: int, random_state: int) -> list[np.ndarray]:
    """Return the Voronoi cells one aNNE estimator lays over the unit square.

    Mirroring the centres across each edge makes every cell finite and exactly
    bounded by the square, which avoids clipping infinite ridges by hand.
    """
    centres = (
        IsoKernel(
            method="anne",
            n_estimators=1,
            max_samples=n_cells,
            random_state=random_state,
        )
        .fit(sample_points(random_state))
        .iso_kernel_.center_data
    )
    blocks = [centres]
    for axis, value in ((0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)):
        mirror = centres.copy()
        mirror[:, axis] = 2 * value - mirror[:, axis]
        blocks.append(mirror)
    diagram = Voronoi(np.vstack(blocks))
    return [
        diagram.vertices[diagram.regions[diagram.point_region[i]]]
        for i in range(len(centres))
    ]


def area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fatness(poly: np.ndarray) -> float:
    """1 for a circle, near 0 for a sliver. Keeps thin wedges out of the mark."""
    perimeter = np.hypot(*(np.roll(poly, -1, 0) - poly).T).sum()
    return 4 * np.pi * area(poly) / perimeter**2 if perimeter else 0.0


def composition_score(cells: list[np.ndarray]) -> float:
    """Rank candidates so --preview shows the ones worth looking at first."""
    areas = np.array([area(c) for c in cells])
    return min(fatness(c) for c in cells) * 2 + (
        1 - abs(areas.max() / areas.sum() - 0.4) * 4
    )


def render(
    n_cells: int = CELLS, random_state: int = RANDOM_STATE, size: int = SIZE
) -> str:
    cells = partition(n_cells, random_state)
    areas = np.array([area(c) for c in cells])
    rank = np.argsort(np.argsort(-areas))
    isolated = int(np.argmax(areas))

    polygons = []
    for i, cell in enumerate(cells):
        points = " ".join(f"{x * size:.1f},{(1 - y) * size:.1f}" for x, y in cell)
        fill = ORANGE if i == isolated else BLUE[min(rank[i], len(BLUE) - 1)]
        polygons.append(f'<polygon points="{points}" fill="{fill}"/>')

    cx, cy, r = DOT
    joined = "\n        ".join(polygons)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" \
role="img" aria-label="IKPyKit">
  <defs>
    <clipPath id="disc"><circle cx="256" cy="256" r="230"/></clipPath>
  </defs>
  <g id="partition" clip-path="url(#disc)">
    <g stroke="{WHITE}" stroke-width="8" stroke-linejoin="round">
        {joined}
    </g>
  </g>
  <!-- The mark is mostly blue and the site header is too, so without this ring
       the disc dissolves into it. On a light background the ring is invisible. -->
  <circle cx="256" cy="256" r="230" fill="none" stroke="{WHITE}" stroke-width="16"/>
  <g id="monogram">
    <g stroke="{WHITE}" stroke-width="{STROKE}" fill="none" stroke-linejoin="round">
      <path d="{ARMS}"/>
      <path d="{K_STEM}"/>
      <path d="{I_STEM}"/>
    </g>
    <!-- The ring keeps the dot visible when it falls on the orange cell. -->
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="{ORANGE}" stroke="{WHITE}" stroke-width="11"/>
  </g>
</svg>
"""


def write_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mark = render()
    written = ["logo.svg"]
    (OUT_DIR / "logo.svg").write_text(mark)
    # PNG fallbacks: 512 for social cards and PyPI, 180 for iOS home screens,
    # 32 for browsers that still ask for a raster favicon.
    for name, width in (
        ("logo-512.png", 512),
        ("logo-180.png", 180),
        ("favicon-32.png", 32),
    ):
        cairosvg.svg2png(
            bytestring=mark.encode(), write_to=str(OUT_DIR / name), output_width=width
        )
        written.append(name)
    print("\n".join(f"wrote {OUT_DIR / name}" for name in written))


def write_preview(n_cells: int, count: int) -> None:
    import io

    from PIL import Image

    ranked = sorted(
        ((composition_score(partition(n_cells, s)), s) for s in range(40)),
        key=lambda pair: -pair[0],
    )[:count]

    sheet = Image.new("RGBA", (220 * count, 290), (255, 255, 255, 255))
    for i, (score, state) in enumerate(ranked):
        png = cairosvg.svg2png(
            bytestring=render(n_cells, state).encode(), output_width=210
        )
        big = Image.open(io.BytesIO(png)).convert("RGBA")
        flat = Image.new("RGBA", big.size, (255, 255, 255, 255))
        flat.alpha_composite(big)
        sheet.alpha_composite(flat, (220 * i, 0))
        for j, px in enumerate((16, 24, 32)):
            sheet.alpha_composite(
                flat.resize((px, px), Image.LANCZOS), (220 * i + 10 + j * 60, 225)
            )
        print(f"random_state={state:<3} score={score:.3f}")
    path = pathlib.Path(f"/tmp/ikpykit-logo-{n_cells}-cells.png")
    sheet.save(path)
    print(f"contact sheet: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true", help="render other candidates"
    )
    parser.add_argument("--cells", type=int, default=CELLS)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    if args.preview:
        write_preview(args.cells, args.count)
    else:
        write_assets()


if __name__ == "__main__":
    main()
