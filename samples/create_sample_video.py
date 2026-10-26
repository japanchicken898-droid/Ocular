"""Synthetic Screen Recording Video Generator for Ocular Compiler Testing."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_screen_recording(
    output_path: str | Path = "sample_recording.mp4",
    fps: int = 10,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Generates a multi-state synthetic screen recording demonstrating UI flow."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Helper to draw a screen
    def make_screen(bg_color, title, subtitle, elements):
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        # Header banner
        cv2.rectangle(img, (0, 0), (width, 80), (30, 30, 40), -1)
        cv2.putText(
            img,
            "OCULAR TEST APP",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Title
        cv2.putText(
            img,
            title,
            (100, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Subtitle
        cv2.putText(
            img,
            subtitle,
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        for elem_type, box, label in elements:
            x1, y1, x2, y2 = box
            if elem_type == "input":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 120), 2)
                cv2.putText(
                    img,
                    label,
                    (x1 + 15, y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA,
                )
            elif elem_type == "button":
                cv2.rectangle(img, (x1, y1), (x2, y2), (220, 100, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (x1 + 25, y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif elem_type == "card":
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 65), -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 100), 2)
                cv2.putText(
                    img,
                    label,
                    (x1 + 20, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return img

    # Screen 1: Login
    screen1_elements = [
        ("input", (400, 260, 880, 310), "Email: admin@ocular.local"),
        ("input", (400, 340, 880, 390), "Password: ************"),
        ("button", (400, 430, 880, 485), "Sign In"),
    ]
    frame_login = make_screen(
        (20, 20, 28),
        "Authentication Portal",
        "Sign in to access offline compiled workflows",
        screen1_elements,
    )

    # Screen 2: Search / Catalog
    screen2_elements = [
        ("input", (200, 140, 800, 190), "Search operations or models..."),
        ("button", (830, 140, 1080, 190), "Filter Results"),
        ("card", (200, 230, 1080, 380), "Model Pipeline: SAM 3.1 & Muse DAG"),
        ("card", (200, 410, 1080, 560), "Compilation Target: Playwright v1.40"),
        ("button", (900, 600, 1080, 650), "Proceed to Dashboard"),
    ]
    frame_search = make_screen(
        (25, 28, 35),
        "Resource Explorer",
        "Explore compiled visual workflows and execution graphs",
        screen2_elements,
    )

    # Screen 3: Dashboard & Analytics
    screen3_elements = [
        ("card", (100, 240, 480, 450), "Latency: 0ms (Zero Vision Runtime)"),
        ("card", (520, 240, 900, 450), "Reliability: 99.98% Deterministic"),
        ("card", (940, 240, 1200, 450), "Emitted: FastAPI + Playwright"),
        ("button", (980, 140, 1200, 190), "Export Report"),
    ]
    frame_dash = make_screen(
        (18, 22, 30),
        "Mission Control Dashboard",
        "Real-time offline telemetry and automation status",
        screen3_elements,
    )

    # Write frames: 10 frames of Login, 10 frames of Search, 10 frames of Dashboard
    # Total 30 frames (3 seconds)
    for _ in range(12):
        out.write(frame_login)
    for _ in range(12):
        out.write(frame_search)
    for _ in range(12):
        out.write(frame_dash)

    out.release()
    return output_path


if __name__ == "__main__":
    path = generate_synthetic_screen_recording()
    print(f"Sample video created at: {path}")
