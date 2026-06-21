from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int
    subject: str
    fall_start: Optional[float]
    fall_end: Optional[float]

    @property
    def key(self) -> str:
        return self.path.as_posix()


_BRACKET_RE = re.compile(r"\[([^]]+)\]")
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _fall_interval(classes: str, length: float) -> tuple[float, float]:
    segments = [segment.strip() for segment in classes.split(";") if segment.strip()]
    fall_segments = [segment for segment in segments if "fall" in segment.lower()]
    if not fall_segments:
        raise ValueError(f"Missing fall annotation: {classes!r}")

    for segment in fall_segments:
        match = _BRACKET_RE.search(segment)
        if match:
            numbers = [float(value) for value in _NUMBER_RE.findall(match.group(1))]
            if len(numbers) >= 2:
                return numbers[0], numbers[1]

    # One source row omits the fall interval but gives the preceding sitting
    # interval. Treat its end as the annotated fall onset.
    preceding_ends = []
    for segment in segments:
        if "fall" in segment.lower():
            continue
        for match in _BRACKET_RE.finditer(segment):
            numbers = [float(value) for value in _NUMBER_RE.findall(match.group(1))]
            if len(numbers) >= 2:
                preceding_ends.append(numbers[1])
    if preceding_ends:
        return max(preceding_ends), length

    raise ValueError(f"Could not parse fall interval: {classes!r}")


def load_records(dataset_root: Path) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    for subject_dir in sorted(dataset_root.glob("Subject *")):
        for category, label in (("ADL", 0), ("Fall", 1)):
            csv_path = subject_dir / f"{category}.csv"
            if not csv_path.exists():
                continue
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    filename = row["File Name"].strip()
                    path = subject_dir / category / filename
                    if not path.exists():
                        raise FileNotFoundError(path)
                    length = float(row["Length (seconds)"].strip())
                    if label:
                        start, end = _fall_interval(row.get(" Classes", ""), length)
                    else:
                        start = end = None
                    records.append(
                        VideoRecord(path, label, subject_dir.name, start, end)
                    )
    return records


def cache_path(cache_root: Path, record: VideoRecord) -> Path:
    subject = record.path.parent.parent.name.replace(" ", "_")
    category = record.path.parent.name.lower()
    return cache_root / subject / category / f"{record.path.stem}.npz"
