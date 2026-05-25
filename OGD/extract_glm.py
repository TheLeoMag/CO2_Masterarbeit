from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GML = SCRIPT_DIR / "pd_popreg_100m_7767c33f-302c-11e3-beb4-0000c1ab0db6.gml"
DEFAULT_OUTPUT = DEFAULT_GML.with_suffix(".geoparquet")
CRS = "EPSG:3035"

NS = {
    "pd": "http://inspire.ec.europa.eu/schemas/pd/4.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

CELL_ID_RE = re.compile(r"AT_CRS3035RES(?P<size>\d+)mN(?P<northing>\d+)E(?P<easting>\d+)$")


def parse_grid_cell_href(href: str) -> tuple[int, int, int]:
    """Return easting, northing, and cell size from an INSPIRE grid-cell href."""
    match = CELL_ID_RE.search(href)
    if match is None:
        raise ValueError(f"Unsupported grid cell reference: {href}")

    return (
        int(match.group("easting")),
        int(match.group("northing")),
        int(match.group("size")),
    )


def iter_population_cells(gml_path: Path):
    """Stream population grid cells from the nested StatisticalValue elements."""
    for _, element in ET.iterparse(gml_path, events=("end",)):
        if element.tag != f"{{{NS['pd']}}}StatisticalValue":
            continue

        value_element = element.find("pd:value", NS)
        spatial_element = element.find(".//pd:spatial", NS)

        if value_element is None or spatial_element is None:
            element.clear()
            continue

        href = spatial_element.attrib.get(f"{{{NS['xlink']}}}href")
        if not href or value_element.text is None:
            element.clear()
            continue

        easting, northing, cell_size = parse_grid_cell_href(href)
        yield {
            "population": int(value_element.text),
            "cell_id": href.rsplit("/", 1)[-1],
            "easting": easting,
            "northing": northing,
            "cell_size_m": cell_size,
            "geometry": box(easting, northing, easting + cell_size, northing + cell_size),
        }

        element.clear()


def gml_to_geodataframe(gml_path: Path) -> gpd.GeoDataFrame:
    records = list(iter_population_cells(gml_path))
    if not records:
        raise ValueError(f"No population grid cells found in {gml_path}")

    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract INSPIRE population grid cells from a GML file to GeoParquet."
    )
    parser.add_argument(
        "gml",
        nargs="?",
        type=Path,
        default=DEFAULT_GML,
        help=f"Input GML file. Defaults to {DEFAULT_GML.name}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output GeoParquet file. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gml_path = args.gml.resolve()
    output_path = args.output.resolve()

    if not gml_path.exists():
        raise FileNotFoundError(f"Input GML file does not exist: {gml_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = gml_to_geodataframe(gml_path)
    grid.to_parquet(output_path, index=False)

    print(f"Wrote {len(grid):,} grid cells to {output_path}")


if __name__ == "__main__":
    main()
