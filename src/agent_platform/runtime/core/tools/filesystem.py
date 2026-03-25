from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import fnmatch
import re

from python_ripgrep import search as rg_search  # ✅ ripgrep backend

from ...orch.state import AgentState
from ..context_store import ContextStore
from ..schema import ErrorCode

logger = logging.getLogger(__name__)


class FilesystemTools:
    """
    FilesystemTools provides a safe, cross-platform interface for agents.
    """

    MAX_TREE_DEPTH = 5  # ✅ constant

    def __init__(self, session_path: Path, context_store: Optional[ContextStore] = None):
        self.session_path = session_path
        self.context_store = context_store

    # ------------------------
    # INTERNAL HELPERS
    # ------------------------

    def _resolve_path(self, path_str: str, state: Optional[AgentState] = None) -> Optional[Path]:
        path = Path(path_str)
        return path.resolve() if path.is_absolute() else (self.session_path / path_str).resolve()

    def _normalize_extensions(self, extensions: Optional[List[str]]) -> Optional[List[str]]:
        if not extensions:
            return None
        return [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

    def _match_filters(self, p: Path, base: Path, glob: Optional[str], extensions: Optional[List[str]]) -> bool:
        if extensions and p.is_file():
            if p.suffix.lower() not in extensions:
                return False

        if glob:
            rel = str(p.relative_to(base)).replace("\\", "/")
            if not fnmatch.fnmatch(rel, glob):
                return False

        return True

    def _truncate_tree(self, lines: List[str]) -> List[str]:
        return lines + [f"Tree truncated to depth {self.MAX_TREE_DEPTH}"]

    def _regex_supported_by_ripgrep(self, pattern: str) -> bool:
        """
        Detect unsupported constructs in ripgrep (Rust regex engine).
        """
        unsupported_patterns = [
            r"\(\?<=",   # lookbehind
            r"\(\?<!",   # negative lookbehind
            r"\\\d+",    # backreferences like \1
        ]
        return not any(re.search(p, pattern) for p in unsupported_patterns)


    # ------------------------
    # FILE OPERATIONS
    # ------------------------

    def write_file(self, file_path: str, content: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {
                "success": False,
                "error": f"Invalid path: {file_path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        knowledge_dir = (self.session_path / "knowledge").resolve()
        try:
            if resolved_path == knowledge_dir or knowledge_dir in resolved_path.parents:
                return {
                    "success": False,
                    "error": "Direct writes to the 'knowledge/' directory are prohibited.",
                    "error_code": ErrorCode.PERMISSION_DENIED
                }
        except ValueError:
            pass

        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content)
            return {"success": True, "path": str(resolved_path)}
        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}

    def read_file(self, file_path: str, state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(file_path)
        if not resolved_path:
            return {
                "success": False,
                "error": f"Invalid path: {file_path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        try:
            content = resolved_path.read_text()
            return {"success": True, "content": content}
        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}

    def ls(self, path: str = ".", state: Optional[AgentState] = None) -> Dict[str, Any]:
        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.is_dir():
            return {
                "success": False,
                "error": f"Invalid directory: {path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        try:
            entries = []
            for e in resolved_path.iterdir():
                try:
                    stat = e.stat()
                    entries.append({
                        "name": e.name,
                        "type": "dir" if e.is_dir() else "file",
                        "size": stat.st_size if e.is_file() else None,
                    })
                except Exception:
                    entries.append({
                        "name": e.name,
                        "type": "unknown",
                        "size": None
                    })

            return {"success": True, "entries": entries}

        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}




    # ------------------------
    # TREE TOOL
    # ------------------------

    def tree(
        self,
        path: str = ".",
        glob: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        state: Optional[AgentState] = None
    ) -> Dict[str, Any]:

        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.is_dir():
            return {
                "success": False,
                "error": f"Invalid directory: {path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        max_depth = min(max_depth or self.MAX_TREE_DEPTH, self.MAX_TREE_DEPTH)
        truncated = False

        try:
            base = resolved_path
            lines: List[str] = []
            extensions = self._normalize_extensions(extensions)

            def walk(current: Path, prefix: str = "", depth: int = 0):
                nonlocal truncated

                if depth > max_depth:
                    truncated = True
                    return

                try:
                    entries = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                except PermissionError:
                    lines.append(f"{prefix}[Permission Denied]")
                    return

                for i, entry in enumerate(entries):
                    connector = "└── " if i == len(entries) - 1 else "├── "

                    if not self._match_filters(entry, base, glob, extensions) and not entry.is_dir():
                        continue

                    lines.append(f"{prefix}{connector}{entry.name}")

                    if entry.is_dir():
                        extension = "    " if i == len(entries) - 1 else "│   "
                        walk(entry, prefix + extension, depth + 1)

            lines.append(str(base))
            walk(base)

            if truncated:
                lines = self._truncate_tree(lines)

            return {"success": True, "tree": "\n".join(lines)}

        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}

    # ------------------------
    # MAX DEPTH TOOL
    # ------------------------

    def max_depth(
        self,
        path: str = ".",
        glob: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        state: Optional[AgentState] = None
    ) -> Dict[str, Any]:

        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.is_dir():
            return {
                "success": False,
                "error": f"Invalid directory: {path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        try:
            base = resolved_path
            extensions = self._normalize_extensions(extensions)

            max_depth_val = 0

            for p in base.rglob("*"):
                try:
                    if not self._match_filters(p, base, glob, extensions):
                        continue

                    rel = p.relative_to(base)
                    depth = len(rel.parts)

                    if depth > max_depth_val:
                        max_depth_val = depth

                except Exception:
                    continue

            return {
                "success": True,
                "max_depth": max_depth_val,
                "root": str(base)
            }

        except Exception as e:
            code = ErrorCode.PERMISSION_DENIED if isinstance(e, PermissionError) else ErrorCode.EXECUTION_ERROR
            return {"success": False, "error": str(e), "error_code": code}




    # ------------------------
    # GREP TOOL (ripgrep-backed)
    # ------------------------

    def grep(
        self,
        path: str,
        pattern: str,
        glob: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        before: int = 2,
        after: int = 2,
        state: Optional[AgentState] = None
    ) -> Dict[str, Any]:

        resolved_path = self._resolve_path(path)
        if not resolved_path or not resolved_path.exists():
            return {
                "success": False,
                "error": f"Invalid path: {path}",
                "error_code": ErrorCode.INVALID_ARGUMENTS
            }

        extensions = self._normalize_extensions(extensions)

        # ⚠️ Check regex compatibility
        use_ripgrep = self._regex_supported_by_ripgrep(pattern)

        try:
            # ------------------------
            # FAST PATH: ripgrep
            # ------------------------
            if use_ripgrep:
                results = rg_search(
                    patterns=[pattern],
                    paths=[str(resolved_path)],
                    globs=extensions if extensions else ([glob] if glob else None),
                )

                matches = []

                for r in results:
                    matches.append({
                        "file": r.path,
                        "line_number": r.line_number,
                        "match": r.line,
                        "context": []  # python-ripgrep may not support A/B context yet
                    })

                return {
                    "success": True,
                    "engine": "ripgrep",
                    "matches": matches,
                    "count": len(matches)
                }

            # ------------------------
            # FALLBACK: Python regex
            # ------------------------
            logger.info("Falling back to Python regex due to unsupported pattern")

            regex = re.compile(pattern)
            matches = []

            files = (
                [resolved_path]
                if resolved_path.is_file()
                else resolved_path.rglob("*")
            )

            for file in files:
                if not file.is_file():
                    continue

                try:
                    lines = file.read_text(errors="ignore").splitlines()

                    for i, line in enumerate(lines):
                        if regex.search(line):
                            start = max(0, i - before)
                            end = min(len(lines), i + after + 1)

                            matches.append({
                                "file": str(file),
                                "line_number": i + 1,
                                "match": line,
                                "context": lines[start:end]
                            })

                except Exception:
                    continue

            return {
                "success": True,
                "engine": "python-re",
                "matches": matches,
                "count": len(matches),
                "warning": "Used fallback regex engine (ripgrep unsupported pattern)"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": ErrorCode.EXECUTION_ERROR
            }