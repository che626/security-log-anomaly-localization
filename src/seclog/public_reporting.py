"""Local result records and dependency-free static summaries for public runs."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .public_protocol import PublicPrediction, PublicProtocolError


@dataclass(frozen=True)
class PublicResultRecord:
    experiment_id: str
    dataset: str
    profile: str
    split_strategy: str
    model: str
    manifest_sha256: str
    metrics: dict[str, float | int | None]
    metadata: dict[str, Any]

    def validate(self) -> None:
        required = (
            self.experiment_id,
            self.dataset,
            self.profile,
            self.split_strategy,
            self.model,
            self.manifest_sha256,
        )
        if any(not value.strip() for value in required):
            raise PublicProtocolError("public result record identity fields cannot be empty")
        if len(self.manifest_sha256) != 64:
            raise PublicProtocolError("public result record requires a SHA256 manifest hash")
        if not self.metrics:
            raise PublicProtocolError("public result record requires metrics")


def write_result_record(path: Path, record: PublicResultRecord) -> None:
    record.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_result_record(path: Path) -> PublicResultRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = PublicResultRecord(**payload)
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublicProtocolError(f"invalid public result record: {path}") from exc
    record.validate()
    return record


def write_predictions(path: Path, predictions: Iterable[PublicPrediction]) -> None:
    """Write local detail records without raw log lines or source identifiers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    {
                        "sid": prediction.sid,
                        "score": prediction.score,
                        "has_anomaly": prediction.has_anomaly,
                        "spans": [asdict(span) for span in prediction.spans],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _compatible(records: tuple[PublicResultRecord, ...]) -> None:
    if not records:
        raise PublicProtocolError("cannot aggregate an empty public result collection")
    fields = ("dataset", "profile", "split_strategy", "manifest_sha256")
    for field in fields:
        values = {getattr(record, field) for record in records}
        if len(values) != 1:
            raise PublicProtocolError(f"cannot aggregate public results with mixed {field}")


def write_aggregate_report(output_dir: Path, records: Iterable[PublicResultRecord]) -> dict[str, Path]:
    """Write compact report inputs without including raw logs or detailed predictions."""

    items = tuple(records)
    _compatible(items)
    if len({record.model for record in items}) != len(items):
        raise PublicProtocolError("aggregate report requires one result row per model")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for record in items for key in record.metrics})
    table_path = output_dir / "summary.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("experiment_id", "model", *metric_keys, "metadata_json"),
        )
        writer.writeheader()
        for record in sorted(items, key=lambda item: item.model):
            writer.writerow(
                {
                    "experiment_id": record.experiment_id,
                    "model": record.model,
                    **record.metrics,
                    "metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                }
            )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": items[0].dataset,
                "profile": items[0].profile,
                "split_strategy": items[0].split_strategy,
                "manifest_sha256": items[0].manifest_sha256,
                "records": [asdict(record) for record in sorted(items, key=lambda item: item.model)],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = output_dir / "f1-comparison.svg"
    _write_f1_svg(
        figure_path,
        items,
        title=f"{items[0].dataset} / {items[0].split_strategy} F1 comparison",
    )
    return {"table": table_path, "summary": summary_path, "figure": figure_path}


def _write_f1_svg(path: Path, records: tuple[PublicResultRecord, ...], *, title: str) -> None:
    values = [(record.model, float(record.metrics.get("f1", record.metrics.get("span_f1", 0.0)) or 0.0)) for record in records]
    width = 760
    row_height = 44
    height = 92 + len(values) * row_height
    bars: list[str] = []
    for index, (label, value) in enumerate(sorted(values)):
        y = 56 + index * row_height
        normalized = max(0.0, min(1.0, value))
        bars.extend(
            (
                f'<text x="24" y="{y + 17}" font-size="14">{html.escape(label)}</text>',
                f'<rect x="240" y="{y}" width="460" height="24" rx="4" fill="#e5e7eb"/>',
                f'<rect x="240" y="{y}" width="{460 * normalized:.1f}" height="24" rx="4" fill="#2563eb"/>',
                f'<text x="710" y="{y + 17}" font-size="13" text-anchor="end">{value:.4f}</text>',
            )
        )
    path.write_text(
        "\n".join(
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="24" y="30" font-size="20" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>',
                *bars,
                "</svg>",
            )
        )
        + "\n",
        encoding="utf-8",
    )
