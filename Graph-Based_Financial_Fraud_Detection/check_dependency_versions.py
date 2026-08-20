"""
Strict dependency-version checker for CI/local reproducibility.

The app can only be deterministic across installs if the package set is also
stable. This script verifies that the installed environment exactly matches
requirements.txt for every pinned package that applies to the current platform.

Usage:
    python check_dependency_versions.py
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.markers import Marker
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)(?:\s*;\s*(.+))?$")


def canonical_name(name: str) -> str:
    return name.replace("_", "-").lower()


def marker_applies(marker_text: str | None) -> bool:
    if not marker_text:
        return True
    try:
        return Marker(marker_text).evaluate()
    except Exception as exc:
        raise RuntimeError(f"Invalid environment marker in {REQUIREMENTS_FILE.name}: {marker_text}") from exc


def read_expected_versions() -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = PIN_RE.match(line)
        if not match:
            continue
        package, version, marker_text = match.groups()
        if marker_applies(marker_text):
            expected[canonical_name(package)] = version
    return expected


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            versions[canonical_name(name)] = dist.version
    return versions


def versions_match(installed: str, expected: str) -> bool:
    # Normalise PEP 440 spelling while still requiring exact local builds, e.g.
    # 2.3.1+cpu is intentionally different from 2.3.1.
    try:
        return Version(installed) == Version(expected)
    except Exception:
        return installed == expected


def main() -> int:
    if not REQUIREMENTS_FILE.exists():
        print(f"FAIL: {REQUIREMENTS_FILE.name} is missing.", file=sys.stderr)
        return 1

    expected = read_expected_versions()
    installed = installed_versions()
    missing: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str, str]] = []

    for package, expected_version in sorted(expected.items()):
        actual_version = installed.get(package)
        if actual_version is None:
            missing.append((package, expected_version))
        elif not versions_match(actual_version, expected_version):
            mismatched.append((package, expected_version, actual_version))

    if missing or mismatched:
        print("FAIL: installed dependencies do not match requirements.txt.")
        if missing:
            print("\nMissing packages:")
            for package, expected_version in missing:
                print(f"  - {package}=={expected_version}")
        if mismatched:
            print("\nVersion mismatches:")
            for package, expected_version, actual_version in mismatched:
                print(f"  - {package}: expected {expected_version}, installed {actual_version}")
        return 1

    print(f"PASS: {len(expected)} applicable pinned dependencies match requirements.txt exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
