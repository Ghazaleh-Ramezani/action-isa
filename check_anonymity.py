#!/usr/bin/env python3
"""Scan a directory (the repo you are about to upload, or the paper source)
for de-anonymising strings. Exit code 1 if anything is found.

    python check_anonymity.py path/to/repo --names-file ~/.anon_needles

Keep the names file OUTSIDE the repo (one string per line: surnames,
institutions, city, email fragments, GitHub handle). This script itself
contains no names, so it is safe to ship inside the bundle.
Always checked regardless of the names file: e-mail addresses, absolute
home paths that leak a username, and a .git directory.
"""
import argparse, re, sys
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "weights"}
SKIP_EXT = {".pdf", ".png", ".jpg", ".pt", ".bin", ".safetensors", ".npy", ".zip"}
HOME_RE = re.compile(r"/(?:home|Users)/([A-Za-z0-9_.-]+)/")
MAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--names-file", help="file with one identifying string per line, kept outside the repo")
    ap.add_argument("--names", nargs="*", default=[])
    a = ap.parse_args()
    needles = [n.lower() for n in a.names]
    if a.names_file:
        needles += [l.strip().lower() for l in Path(a.names_file).expanduser().read_text().splitlines() if l.strip()]
    hits = 0
    root = Path(a.root)
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file() or p.suffix in SKIP_EXT:
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        low = text.lower()
        for n in needles:
            if n in low:
                hits += 1; print(f"{p}: contains a listed identifier")
        for m in HOME_RE.finditer(text):
            hits += 1; print(f"{p}: absolute home path leaks username '{m.group(1)}'")
        for m in MAIL_RE.finditer(text):
            if not m.group(0).lower().endswith(("example.com", "anonymous.com")):
                hits += 1; print(f"{p}: e-mail address {m.group(0)}")
    if (root / ".git").exists():
        hits += 1; print(f"{root/'.git'}: remove before uploading (history carries author names and e-mails)")
    print(f"\n{hits} finding(s)." if hits else "\nclean.")
    sys.exit(1 if hits else 0)

if __name__ == "__main__":
    main()
