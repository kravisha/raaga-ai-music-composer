"""Shared, filesystem-real path containment check.

``posixpath.normpath`` (the previous implementation) only understands
forward slashes and pure string syntax: it does not resolve a Windows
backslash (``..\\``), a drive-relative path, or a symlink/junction, so a
maliciously or accidentally crafted path could pass a purely lexical check
while actually landing outside the intended root on this platform. This
module instead resolves both paths with ``os.path.realpath`` (which
understands the current OS's separators, collapses ``..`` against the real
filesystem, and follows any existing symlink/junction) and compares them
case-insensitively via ``os.path.normcase``, so the comparison is bound to
the same actually-resolved root on Windows.

This is still a best-effort, read-only check, not a race-free guarantee:
a symlink can be swapped in between this check and a later filesystem
access. Callers that actually write/export files must additionally create
their target exclusively (e.g. open with an exclusive-create flag) to
close that window; this module does not attempt to.
"""

from __future__ import annotations

import os


def resolve_path(path: str) -> str:
    """Canonicalize ``path`` for containment comparison purposes."""
    return os.path.normcase(os.path.realpath(path))


def is_within_root(root: str, path: str) -> bool:
    """True if the real, resolved ``path`` is ``root`` or under it."""
    root_resolved = resolve_path(root)
    path_resolved = resolve_path(path)
    if path_resolved == root_resolved:
        return True
    return path_resolved.startswith(root_resolved + os.sep)
