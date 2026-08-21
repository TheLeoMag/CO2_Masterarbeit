"""Small schema helpers shared by routing feature generation and validation."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq


FACHGRUPPE_LAG_PATTERN = re.compile(r"^fachgruppe_(.+)_active_firms_tminus1$")
ACCESS_MINUTES = (5, 10, 15, 30)


def fachgruppe_ids(panel_path: Path) -> list[str]:
    """Return the Fachgruppe IDs encoded in the lagged panel columns."""
    ids = []
    for name in pq.ParquetFile(panel_path).schema_arrow.names:
        match = FACHGRUPPE_LAG_PATTERN.match(name)
        if match:
            ids.append(match.group(1))
    return ids


def fachgruppe_stock_columns(ids: list[str]) -> list[str]:
    return [f"fachgruppe_{fachgruppe_id}_active_firms_tminus1" for fachgruppe_id in ids]


def fachgruppe_access_columns(ids: list[str]) -> list[str]:
    return [
        f"fachgruppe_{fachgruppe_id}_access_{minutes}min"
        for fachgruppe_id in ids
        for minutes in ACCESS_MINUTES
    ]


def main_access_columns() -> list[str]:
    return [
        column
        for minutes in ACCESS_MINUTES
        for column in (f"pop_access_{minutes}min", f"existing_firms_access_{minutes}min")
    ]
