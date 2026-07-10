from seclog.presentation import annotate_lines


def test_annotate_lines_marks_only_inclusive_span() -> None:
    result = annotate_lines(["a", "b", "c"], start=1, end=2)
    assert [item.is_anomalous for item in result] == [False, True, True]
    assert [item.line_number for item in result] == [0, 1, 2]
