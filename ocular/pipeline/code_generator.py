"""Muse Code: Jinja2 Template Compiler targeting output directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader

from ocular.pipeline.base import BaseGenerator


class JinjaCodeGenerator(BaseGenerator):
    """Compiles state graphs and visual segments into runnable Python artifacts."""

    def __init__(self, templates_dir: Optional[str | Path] = None) -> None:
        if templates_dir is None:
            self.templates_dir = Path(__file__).resolve().parent.parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _compute_hash(self, dag: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
        payload = json.dumps({"dag": dag, "segments": segments}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def emit(
        self,
        dag: Dict[str, Any],
        segments: List[Dict[str, Any]],
        out_dir: str | Path = "./dist",
    ) -> Dict[str, Path]:
        """Renders playwright_driver.py and main.py into out_dir."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        compile_hash = self._compute_hash(dag, segments)
        context = {
            "dag": dag,
            "segments": segments,
            "compile_hash": compile_hash,
        }

        # 1. Render playwright_driver.py
        driver_tmpl = self.env.get_template("playwright_driver.py.j2")
        driver_code = driver_tmpl.render(context)
        driver_file = out_path / "playwright_driver.py"
        driver_file.write_text(driver_code, encoding="utf-8")

        # 2. Render main.py (FastAPI service)
        app_tmpl = self.env.get_template("fastapi_app.py.j2")
        app_code = app_tmpl.render(context)
        app_file = out_path / "main.py"
        app_file.write_text(app_code, encoding="utf-8")

        return {
            "playwright_driver": driver_file.resolve(),
            "main": app_file.resolve(),
        }
