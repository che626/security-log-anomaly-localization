import html
import json
import os
from pathlib import Path

import streamlit as st

from seclog.config import load_config
from seclog.inference import predict_text
from seclog.presentation import annotate_lines

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "synthetic_logs"
MAX_LINES = 400
MAX_CHARACTERS = 200_000


def load_examples() -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for path in sorted(EXAMPLE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples[str(payload["name"])] = [str(line) for line in payload["lines"]]
    return examples


def configured_checkpoints() -> list[Path]:
    raw = os.environ.get("SECLOG_CHECKPOINTS", "")
    return [Path(item) for item in raw.split(os.pathsep) if item.strip()]


def render_lines(lines: list[str], start: int, end: int) -> None:
    for item in annotate_lines(lines, start, end):
        background = "#fee2e2" if item.is_anomalous else "#f8fafc"
        border = "#ef4444" if item.is_anomalous else "#cbd5e1"
        safe_text = html.escape(item.text)
        st.markdown(
            (
                f'<div style="padding:.55rem .75rem;margin:.2rem 0;background:{background};'
                f'border-left:4px solid {border};border-radius:4px;font-family:monospace">'
                f'<span style="color:#64748b;margin-right:.8rem">{item.line_number:03d}</span>'
                f"{safe_text}</div>"
            ),
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Security Log Anomaly Localization", layout="wide")
st.title("Security Log Anomaly Localization")
st.warning(
    "作品集原型，仅用于演示模型流程，不是生产安全系统；输出可能出错，置信度未经校准。"
)

examples = load_examples()
source = st.selectbox("输入来源", [*examples, "粘贴自定义日志"])
default_text = "\n".join(examples.get(source, []))
text = st.text_area(
    "每行一条日志",
    value=default_text,
    height=220,
    placeholder="paste one log event per line",
)
lines = [line.strip() for line in text.splitlines() if line.strip()]

checkpoints = configured_checkpoints()
missing_checkpoints = [path for path in checkpoints if not path.is_file()]
if not checkpoints:
    st.info(
        "当前未配置私有模型。设置 SECLOG_CHECKPOINTS（多个路径用系统路径分隔符连接）后可运行推理。"
    )
elif missing_checkpoints:
    st.error("以下本地检查点不存在：" + ", ".join(str(path) for path in missing_checkpoints))
else:
    st.success(f"已配置 {len(checkpoints)} 个本地折模型。比赛检查点不会随仓库发布。")

if st.button("分析日志", type="primary", disabled=not checkpoints or bool(missing_checkpoints)):
    if len(text) > MAX_CHARACTERS:
        st.error(f"输入超过 {MAX_CHARACTERS:,} 个字符的限制。")
        st.stop()
    if not lines:
        st.error("请输入至少一条非空日志。")
        st.stop()
    if len(lines) > MAX_LINES:
        st.error(f"输入超过 {MAX_LINES} 行的限制。")
        st.stop()
    config_path = Path(os.environ.get("SECLOG_CONFIG", ROOT / "configs" / "final.yaml"))
    try:
        with st.spinner("正在执行本地推理……"):
            prediction = predict_text(lines, checkpoints, load_config(config_path))
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"推理不可用：{error}")
        st.stop()
    left, middle, right = st.columns(3)
    left.metric("是否异常", "是" if prediction["has_anomaly"] else "否")
    middle.metric("异常类型", str(prediction["primary_anomaly_type"]))
    right.metric("模型置信度（未校准）", f"{prediction['confidence']:.1%}")
    st.caption(
        f"预测区间：{prediction['primary_start_idx']}–{prediction['primary_end_idx']}（首尾均包含）"
    )
    render_lines(
        lines,
        int(prediction["primary_start_idx"]),
        int(prediction["primary_end_idx"]),
    )
