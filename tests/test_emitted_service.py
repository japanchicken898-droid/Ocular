"""Tests for the emitted FastAPI wrapper service with real local fixture execution."""

import http.server
from pathlib import Path
import sys
import threading
import pytest
from starlette.testclient import TestClient

from ocular.cli import compile_command
from samples.create_sample_video import generate_synthetic_screen_recording


class RealFixtureHandler(http.server.BaseHTTPRequestHandler):
    workflow_json_path = None

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        elements = []
        seen_ids = {"btn_submit", "input_query", "interval_dropdown", "timeframe_15m"}
        if self.workflow_json_path and Path(self.workflow_json_path).exists():
            import json
            try:
                data = json.loads(Path(self.workflow_json_path).read_text(encoding="utf-8"))
                for aid in data.get("anchors", {}):
                    if aid not in seen_ids:
                        seen_ids.add(aid)
                        elements.append(f'<button id="{aid}" role="button" onclick="this.setAttribute(\'data-clicked\', \'1\');">{aid}</button>')
                for step in data.get("action_steps", []):
                    t = step.get("target_anchor", "")
                    if t and t not in seen_ids:
                        seen_ids.add(t)
                        elements.append(f'<button id="{t}" role="button" onclick="this.setAttribute(\'data-clicked\', \'1\');">{t}</button>')
            except Exception:
                pass
        elements_html = "\n".join(elements)
        html = f"""<!DOCTYPE html>
<html>
<head><title>Ocular Real Fixture</title></head>
<body>
    <button id="btn_submit" role="button" onclick="this.setAttribute('data-clicked', '1');">Submit</button>
    <input id="input_query" role="textbox" name="query" value="" />
    <button id="interval_dropdown" role="button" onclick="this.setAttribute('data-clicked', '1');">Interval</button>
    <button id="timeframe_15m" role="button" onclick="this.setAttribute('data-clicked', '1');">15m</button>
    {elements_html}
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def local_app_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), RealFixtureHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_emitted_fastapi_endpoints(tmp_path: Path, local_app_server: str, monkeypatch):
    monkeypatch.setenv("OCULAR_INSECURE_NO_AUTH", "1")
    app_host_port = local_app_server.replace("http://", "").rstrip("/")
    monkeypatch.setenv("OCULAR_ALLOWED_HOSTS", app_host_port)
    monkeypatch.setenv("OCULAR_ALLOWED_EGRESS_HOSTS", app_host_port)
    video_path = tmp_path / "sample.mp4"
    generate_synthetic_screen_recording(video_path, fps=60, width=1920, height=1080)

    dist_dir = tmp_path / "emitted_dist"
    compile_command(input_path=str(video_path), out_dir=str(dist_dir))
    RealFixtureHandler.workflow_json_path = dist_dir / "workflow.json"

    assert (dist_dir / "app.py").exists()
    assert (dist_dir / "playwright_driver.py").exists()

    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        import importlib
        app_mod = importlib.import_module("app")
        app = app_mod.app

        client = TestClient(app)

        # 0. Test root when docs are disabled by default (D-04)
        res_root = client.get("/", follow_redirects=False)
        assert res_root.status_code == 404

        # 1. Test /health (D-03: strictly {"status": "ok"})
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data == {"status": "ok"}

        # 2. Test /api/v1/schema
        res_schema = client.get("/api/v1/schema")
        assert res_schema.status_code == 200
        schema_data = res_schema.json()
        assert "parameters_schema" in schema_data
        assert "anchors_count" in schema_data
        assert schema_data["zero_runtime_vision"] is True

        # 3. Test POST /api/v1/execute against real local fixture server (F4 verification)
        res_exec = client.post(
            "/api/v1/execute",
            json={
                "base_url": local_app_server,
                "viewport_width": 1920,
                "viewport_height": 1080,
                "headless": True,
            },
        )
        assert res_exec.status_code == 200
        assert "X-Ocular-Compile-Hash" in res_exec.headers
        exec_data = res_exec.json()
        assert exec_data["success"] is True
        assert "execution_id" in exec_data
        assert exec_data["steps_executed"] >= 1
        assert "results" in exec_data
        assert exec_data["results"]["status"] == "success"

    finally:
        if str(dist_dir.resolve()) in sys.path:
            sys.path.remove(str(dist_dir.resolve()))
        sys.modules.pop("app", None)
        sys.modules.pop("playwright_driver", None)
