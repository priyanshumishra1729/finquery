"""Safe Python code validation for generated-code questions."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_python_snippet(answer: str) -> str | None:
    blocks = re.findall(r"```python\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return None


def validate_generated_code(answer: str) -> dict[str, str | bool]:
    snippet = extract_python_snippet(answer)
    if snippet is None:
        return {"status": "not_applicable", "detail": "No Python code block found."}

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = Path(temp_dir) / "generated_snippet.py"
        script_path.write_text(snippet, encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return {"status": "passed", "detail": "Python compilation succeeded."}
        except subprocess.CalledProcessError as exc:
            return {
                "status": "failed",
                "detail": exc.stderr.strip() or "Syntax or runtime compile failure.",
            }
