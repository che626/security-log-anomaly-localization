import pytest

from seclog.public_data import (
    prepare_bgl,
    prepare_hdfs,
    prepare_openstack,
    prepare_thunderbird,
)
from seclog.public_protocol import PublicProtocolError, TaskProfile, read_manifest, read_prepared_dataset


def test_prepare_hdfs_groups_by_block_and_records_manifest(tmp_path) -> None:
    log = tmp_path / "HDFS.log"
    log.write_text(
        "081109 203518 INFO receive blk_-1 from 10.0.0.1\n"
        "081109 203519 INFO write blk_-1 to /tmp/a\n"
        "081109 203520 INFO receive blk_-2 from 10.0.0.2\n",
        encoding="utf-8",
    )
    labels = tmp_path / "anomaly_label.csv"
    labels.write_text("BlockId,Label\nblk_-1,Normal\nblk_-2,Anomaly\n", encoding="utf-8")
    paths = prepare_hdfs(log, labels, tmp_path / "out")
    samples = read_prepared_dataset(paths.dataset_path, TaskProfile.SEQUENCE_BINARY)
    assert [sample.has_anomaly for sample in samples] == [0, 1]
    assert len(samples[0].lines) == 2
    manifest = read_manifest(paths.manifest_path)
    assert manifest.dataset == "hdfs"
    assert manifest.source_line_count == 3


def test_prepare_hdfs_rejects_unlabelled_blocks(tmp_path) -> None:
    log = tmp_path / "HDFS.log"
    log.write_text("INFO blk_-1\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("BlockId,Label\nblk_-2,Normal\n", encoding="utf-8")
    with pytest.raises(PublicProtocolError, match="missing official labels"):
        prepare_hdfs(log, labels, tmp_path / "out")


def test_prepare_hdfs_uses_content_from_structured_csv(tmp_path) -> None:
    log = tmp_path / "HDFS_structured.csv"
    log.write_text(
        "LineId,Date,Time,Content,EventId\n1,081109,203518,receive blk_-1,E1\n2,081109,203519,write blk_-1,E2\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("BlockId,Label\nblk_-1,Anomaly\n", encoding="utf-8")
    paths = prepare_hdfs(log, labels, tmp_path / "out")
    sample = read_prepared_dataset(paths.dataset_path, TaskProfile.SEQUENCE_BINARY)[0]
    assert sample.lines == ("receive blk_-1", "write blk_-1")
    assert sample.timestamp == "081109 203518"
    assert read_manifest(paths.manifest_path).preparation["input_kind"] == "structured_csv"


def test_prepare_grouped_openstack_requires_explicit_grouped_csv(tmp_path) -> None:
    logs = tmp_path / "logs.csv"
    logs.write_text("instance_id,message\ni1,created\ni1,failed\ni2,created\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("instance_id,label\ni1,Anomaly\ni2,Normal\n", encoding="utf-8")
    paths = prepare_openstack(logs, labels, tmp_path / "out")
    samples = read_prepared_dataset(paths.dataset_path, TaskProfile.SEQUENCE_BINARY)
    assert [(sample.sid, sample.has_anomaly) for sample in samples] == [
        ("openstack:i1", 1),
        ("openstack:i2", 0),
    ]


def test_prepare_bgl_builds_nonoverlapping_binary_spans(tmp_path) -> None:
    log = tmp_path / "BGL.log"
    log.write_text(
        "- 1 boot ok\nALERT 2 disk failed\nALERT 3 retry failed\n- 4 recovered\n- 5 boot ok\n",
        encoding="utf-8",
    )
    paths = prepare_bgl(log, tmp_path / "out", window_size=4, stride=4)
    samples = read_prepared_dataset(paths.dataset_path, TaskProfile.SPAN_BINARY)
    assert samples[0].spans[0].start == 1
    assert samples[0].spans[0].end == 2
    assert samples[1].has_anomaly == 0
    with pytest.raises(PublicProtocolError, match="stride"):
        prepare_bgl(log, tmp_path / "bad", window_size=4, stride=1)


def test_thunderbird_requires_a_deterministic_source_line_limit(tmp_path) -> None:
    log = tmp_path / "tbird.log"
    log.write_text("- 1 ok\nALERT 2 bad\n- 3 ok\n", encoding="utf-8")
    paths = prepare_thunderbird(log, tmp_path / "out", window_size=2, stride=2, max_source_lines=2)
    manifest = read_manifest(paths.manifest_path)
    assert manifest.preparation["max_source_lines"] == 2
