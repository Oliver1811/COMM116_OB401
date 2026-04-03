"""
sandbox.py — Restricted Python execution environment.

Executes agent-generated code with:
  • Module allow-list  : numpy, PIL, cv2, math, matplotlib, skimage + stdlib
                         data structures (json, re, collections, …)
  • Module block-list  : os, sys, subprocess, socket, requests, shutil, …
  • File-write guard   : writes only permitted inside ./outputs/
  • Timeout            : 3 s wall-clock via a daemon thread
  • Stdout capture     : print() output is captured and returned

Public API
----------
execute(code, context=None, timeout=3.0) -> dict
    Returns::
        {
          "success": bool,
          "stdout":  str,   # everything printed by the code
          "stderr":  str,   # currently always ""  (stderr not separately routed)
          "error":   str | None,
        }
"""

from __future__ import annotations

import builtins
import io
import textwrap
import threading
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module allow / block lists
# ---------------------------------------------------------------------------

BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "shutil",
        "pathlib",
        "glob",
        "tempfile",
        "fnmatch",
        "pickle",
        "shelve",
        "dbm",
        "marshal",
        "importlib",
        "imp",
        "pkgutil",
        "ctypes",
        "cffi",
        "signal",
        "threading",
        "multiprocessing",
        "concurrent",
        "asyncio",
        "aiohttp",
        "ftplib",
        "smtplib",
        "http",
        "httplib",
        "pty",
        "tty",
        "termios",
        "mmap",
        "resource",
        "pwd",
        "grp",
        "nt",
        "winreg",
    }
)

# ---------------------------------------------------------------------------
# Safe built-ins
# ---------------------------------------------------------------------------

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        # core functions
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "classmethod", "complex",
        "delattr", "dict", "dir", "divmod",
        "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "globals", "hasattr", "hash", "hex",
        "id", "int", "isinstance", "issubclass", "iter",
        "len", "list", "locals", "map", "max", "memoryview",
        "min", "next", "object", "oct", "ord", "pow", "print",
        "property", "range", "repr", "reversed", "round",
        "set", "setattr", "slice", "sorted", "staticmethod",
        "str", "sum", "super", "tuple", "type", "vars", "zip",
        # constants
        "None", "True", "False", "Ellipsis", "NotImplemented",
        # needed by exec internals
        "__build_class__", "__name__", "__doc__",
        "__package__", "__loader__", "__spec__",
        # common exceptions
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
        "BlockingIOError", "BrokenPipeError", "BufferError", "BytesWarning",
        "ChildProcessError", "ConnectionAbortedError", "ConnectionError",
        "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning",
        "EOFError", "EnvironmentError", "Exception", "FileExistsError",
        "FileNotFoundError", "FloatingPointError", "FutureWarning",
        "GeneratorExit", "IOError", "ImportError", "ImportWarning",
        "IndentationError", "IndexError", "InterruptedError",
        "IsADirectoryError", "KeyError", "KeyboardInterrupt",
        "LookupError", "MemoryError", "ModuleNotFoundError", "NameError",
        "NotADirectoryError", "NotImplementedError", "OSError",
        "OverflowError", "PermissionError", "RecursionError", "ReferenceError",
        "RuntimeError", "RuntimeWarning",
        "StopAsyncIteration", "StopIteration", "SyntaxError", "SyntaxWarning",
        "SystemError", "TabError", "TimeoutError",
        "TypeError", "UnboundLocalError", "UnicodeDecodeError",
        "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
        "UnicodeWarning", "UserWarning", "ValueError", "Warning",
        "ZeroDivisionError",
    }
)

# Output directory that sandbox code is allowed to write to
_OUTPUTS_DIR = Path("./outputs")

# ---------------------------------------------------------------------------
# Auto-prelude — prepended to every code block
# ---------------------------------------------------------------------------

# image_path, Image, np, cv2 are all injected into exec_globals before exec();
# this prelude turns them into convenient top-level variables so model code
# never has to repeat the boilerplate, even across independent steps.
_PRELUDE = """\
try:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
except Exception:
    pass
"""

# Modules that are pre-injected into the sandbox namespace.  If the model
# generates "import <name>" for any of these, we silently drop the line
# rather than letting Python fail with ModuleNotFoundError.
_PRE_INJECTED: frozenset[str] = frozenset(
    {"np", "cv2", "math", "plt", "Image", "color", "measure", "feature",
     "hough_line", "hough_circle"}
)

import re as _re
_IMPORT_LINE_RE = _re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(_PRE_INJECTED) + r")\b.*$",
    _re.MULTILINE,
)

def _strip_preinjected_imports(code: str) -> str:
    """Remove import statements for names already in the sandbox namespace."""
    return _IMPORT_LINE_RE.sub("", code)


# ---------------------------------------------------------------------------
# Restricted helpers
# ---------------------------------------------------------------------------

def _make_restricted_import(real_import: Any):
    """Return an __import__ replacement that blocks BLOCKED_MODULES."""

    def restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
        top_level = name.split(".")[0]
        if top_level in BLOCKED_MODULES:
            raise ImportError(
                f"Import of '{name}' is not permitted in the sandbox."
            )
        return real_import(name, *args, **kwargs)

    return restricted_import


def _make_restricted_open(real_open: Any):
    """Return an open() replacement that blocks writes outside _OUTPUTS_DIR."""

    def restricted_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(ch in mode for ch in ("w", "a", "x")):
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                Path(str(file)).resolve().relative_to(_OUTPUTS_DIR.resolve())
            except ValueError:
                raise PermissionError(
                    f"Sandbox: writes outside './outputs/' are not permitted "
                    f"(attempted: '{file}')."
                )
        return real_open(file, mode, *args, **kwargs)

    return restricted_open


# ---------------------------------------------------------------------------
# Core execution (runs inside worker thread)
# ---------------------------------------------------------------------------

def _run_code(
    code: str,
    context: dict[str, Any] | None,
    stdout_buf: io.StringIO,
) -> dict[str, Any]:
    """
    Build a fresh restricted namespace, merge *context*, then exec *code*.
    Captures print() output into *stdout_buf*.
    Must be called from within the worker thread created by execute().
    """
    real_import = builtins.__import__
    real_open   = builtins.open

    safe_builtins: dict[str, Any] = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if name in _SAFE_BUILTIN_NAMES and hasattr(builtins, name)
    }
    safe_builtins["__import__"] = _make_restricted_import(real_import)
    safe_builtins["open"]       = _make_restricted_open(real_open)

    exec_globals: dict[str, Any] = {"__builtins__": safe_builtins}

    try:
        import numpy as _np
        import cv2 as _cv2
        import math as _math
        import matplotlib.pyplot as _plt
        from PIL import Image as _PILImage
        from skimage import color as _color, measure as _measure, feature as _feature
        from skimage.transform import hough_line as _hough_line, hough_circle as _hough_circle
        exec_globals.update({
            "np": _np,
            "cv2": _cv2,
            "math": _math,
            "plt": _plt,
            "Image": _PILImage,
            "color": _color,
            "measure": _measure,
            "feature": _feature,
            "hough_line": _hough_line,
            "hough_circle": _hough_circle,
        })
    except ImportError:
        pass  # missing optional lib — agent code will get a normal ImportError

    if context:
        exec_globals.update(context)

    _real_print = builtins.print

    def _sandbox_print(*args: Any, **kwargs: Any) -> None:
        kwargs["file"] = stdout_buf
        _real_print(*args, **kwargs)

    exec_globals["__builtins__"]["print"] = _sandbox_print

    try:
        clean_code = _strip_preinjected_imports(textwrap.dedent(code))
        compiled = compile(clean_code, "<sandbox>", "exec")
        exec(compiled, exec_globals)  # noqa: S102
        return {
            "success": True,
            "stdout": stdout_buf.getvalue(),
            "stderr": "",
            "error": None,
        }
    except SyntaxError as exc:
        return {
            "success": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": "",
            "error": f"SyntaxError: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute(
    code: str,
    context: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """
    Execute *code* in a sandboxed environment with a timeout.

    A prelude is automatically prepended that opens the image and sets up
    ``img`` and ``arr`` — model-generated code should not repeat this.

    Parameters
    ----------
    code:
        Python source code string to execute.
    context:
        Extra variables merged into the namespace before execution.
        Must include ``image_path`` when image access is needed.
    timeout:
        Wall-clock seconds before execution is abandoned.

    Returns
    -------
    dict with keys ``success``, ``stdout``, ``stderr``, ``error``.
    """
    full_code = _PRELUDE + "\n" + code

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    stdout_buf = io.StringIO()
    result_holder: dict[str, Any] = {}

    def _worker() -> None:
        result_holder["result"] = _run_code(full_code, context, stdout_buf)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return {
            "success": False,
            "stdout": stdout_buf.getvalue(),
            "stderr": "",
            "error": f"Execution timed out after {timeout}s.",
        }

    return result_holder.get(
        "result",
        {
            "success": False,
            "stdout": "",
            "stderr": "",
            "error": "Unknown error in sandbox worker.",
        },
    )
