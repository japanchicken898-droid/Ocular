# Ocular: Offline Visual Compiler

Ocular is an offline visual compiler. It moves multimodal vision inference from runtime to compile time.

Instead of running slow, expensive, and fragile vision-language models at runtime during every test or automation execution, **Ocular compiles screen recordings once** into deterministic, high-performance Playwright automation scripts paired with a production-ready FastAPI operational wrapper.

## Architecture & Workflow

```text
Screen Recording (.mp4)
         │
         ▼
[ocular/pipeline/video_ingest.py]    (Keyframe detection, frame differencing, timeline indexing)
         │
         ▼
[ocular/pipeline/segmenter.py]       (Meta SAM 3.1: Canvas interactive masks & scale-invariant coordinates 0.0 -> 1.0)
         │
         ▼
[ocular/pipeline/state_graph.py]     (Muse Spark: Synthesizing state transition graph / DAG with preconditions)
         │
         ▼
[ocular/pipeline/code_generator.py]  (Muse Code: Jinja2 compilation into Playwright Driver + FastAPI Service)
         │
         ▼
`./dist/` Output Directory
   ├── playwright_driver.py          (Zero-inference deterministic Playwright runtime execution)
   └── main.py                       (FastAPI service wrapper with CORS and latency telemetry headers)
```

## Quickstart

### 1. Run the Compiler
Compile an input screen recording into runnable Python artifacts:
```bash
python -m ocular compile --input fixtures/sample_session.mp4 --out-dir ./dist
```

Sample Terminal Output:
```text
[+] Loaded screen recording: fixtures/sample_session.mp4 (1080p @ 60fps)
[+] Meta SAM 3.1: Segmenting canvas interactive masks...
[+] Muse Spark: Synthesizing state transition graph (DAG)...
[+] Muse Code: Emitting playwright_driver.py & main.py...
[✓] Compilation successful in 3.48s -> Artifacts written to ./dist
```

### 2. Start the Generated FastAPI Service
```bash
uvicorn main:app --app-dir ./dist --host 127.0.0.1 --port 8000
```

### 3. Send Automation Requests
Apply timeframe intervals or query chart canvas status with sub-second execution latency:
```bash
# Apply 15-minute timeframe interval
curl -i -X POST http://127.0.0.1:8000/api/v1/chart/interval \
  -H "Content-Type: application/json" \
  -d '{"interval": "15m", "viewport_width": 1920, "viewport_height": 1080}'

# Query chart status & registered anchors
curl -i http://127.0.0.1:8000/api/v1/chart/status
```

Response includes custom telemetry headers:
- `X-Ocular-Compile-Hash`
- `X-Execution-Latency-MS`

### 4. Run Test Suite
```bash
pytest -v
```
