import re
import zlib

from .constants import HEX_RE, IP_RE, NUM_RE, PATH_RE, SEG_RE, TS_RE, WORD_RE


def hashed_bucket(text: str, mod: int) -> int:
    if mod < 2:
        raise ValueError("vocab_size must be at least 2")
    return (zlib.crc32(text.encode("utf-8")) % (mod - 1)) + 1


def clean_log_line(line: str) -> str:
    text = str(line).strip().lower()
    text = TS_RE.sub("", text)
    text = IP_RE.sub("<ip>", text)
    text = HEX_RE.sub("<hex>", text)
    text = PATH_RE.sub("<path>", text)
    text = SEG_RE.sub("<seg>", text)
    text = NUM_RE.sub("<num>", text)
    return re.sub(r"\s+", " ", text)


def encode_log_line(line: str, vocab_size: int, max_tokens: int = 128) -> list[int]:
    text = clean_log_line(line)
    tokens: list[str] = []
    words = WORD_RE.findall(text)
    tokens.extend("w:" + word for word in words[:48])
    for left, right in zip(words[:32], words[1:33]):
        tokens.append("bg:" + left + "_" + right)
    chars = re.sub(r"\s+", "_", text)
    if len(chars) > 3:
        step = 3 if len(chars) > 80 else 2
        for width in (3, 4):
            for index in range(0, max(0, len(chars) - width + 1), step):
                tokens.append(f"c{width}:" + chars[index : index + width])
                if len(tokens) >= max_tokens:
                    break
            if len(tokens) >= max_tokens:
                break
    if not tokens:
        tokens = ["<empty>"]
    return [hashed_bucket(token, vocab_size) for token in tokens[:max_tokens]]


def nonempty_log_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).split("\n") if line.strip()]
