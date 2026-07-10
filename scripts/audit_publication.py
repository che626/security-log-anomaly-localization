import argparse
import re
import subprocess
from pathlib import Path

MAX_FILE_SIZE = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".joblib", ".npy", ".npz"}
SKIP_DIRECTORIES = {
    ".git",
    ".private",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
}
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.I)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PARTICIPANT_ID = re.compile(r"\bISCC2026-XSTZ-[A-Z0-9_-]+\b", re.I)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\b\s*[:=]\s*"
    r"['\"]?[^\s'\"{}$]{8,}"
)
SENSITIVE_FILENAMES = {".env", "id_rsa", "id_ed25519", "secrets.toml"}
GENERATED_FILENAMES = {".coverage"}


def _is_allowed_fixture_dataset(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    return len(parts) >= 3 and parts[:2] == ("tests", "fixtures")


def audit_paths(root: Path) -> list[str]:
    root = root.resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if path.name in GENERATED_FILENAMES:
            continue
        display = relative.as_posix()
        suffix = path.suffix.lower()
        if path.stat().st_size > MAX_FILE_SIZE:
            findings.append(f"{display}: file exceeds 10 MiB")
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"{display}: model, cache, or array artifact is forbidden")
        if path.name.lower() in {"train.csv", "test.csv"} and not _is_allowed_fixture_dataset(
            relative
        ):
            findings.append(f"{display}: private dataset filename outside tests/fixtures")
        if path.name.lower() in SENSITIVE_FILENAMES:
            findings.append(f"{display}: sensitive filename is forbidden")
        if suffix not in TEXT_SUFFIXES and path.name not in {"README", "LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{display}: expected text file is not valid UTF-8")
            continue
        if WINDOWS_USER_PATH.search(text):
            findings.append(f"{display}: contains an absolute Windows user path")
        if PARTICIPANT_ID.search(text):
            findings.append(f"{display}: contains an ISCC participant identifier")
        if PRIVATE_KEY.search(text):
            findings.append(f"{display}: contains a private key marker")
        if CREDENTIAL_ASSIGNMENT.search(text):
            findings.append(f"{display}: contains a credential-like assignment")
        if EMAIL.search(text) and not display.startswith("docs/"):
            findings.append(f"{display}: contains an email address outside documentation")
    return findings


def audit_staged_ignored(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ["git index: unable to inspect staged files"]
    findings: list[str] = []
    for raw_path in result.stdout.decode("utf-8").split("\0"):
        if not raw_path:
            continue
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", raw_path],
            cwd=root,
            check=False,
        )
        if ignored.returncode == 0:
            findings.append(f"{raw_path}: staged file crosses the approved ignored boundary")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a repository before public release")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    findings = audit_paths(args.root) + audit_staged_ignored(args.root.resolve())
    if findings:
        print("publication audit: FAILED")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("publication audit: OK")


if __name__ == "__main__":
    main()
