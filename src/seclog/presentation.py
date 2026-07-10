from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayLine:
    line_number: int
    text: str
    is_anomalous: bool


def annotate_lines(lines: list[str], start: int, end: int) -> list[DisplayLine]:
    return [
        DisplayLine(index, text, start <= index <= end)
        for index, text in enumerate(lines)
    ]
