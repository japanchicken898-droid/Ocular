import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from ocular.pipeline.code_generator import JinjaCodeGenerator


@pytest.fixture
def driver_cls(tmp_path: Path):
    dist_dir = tmp_path / "dist"
    segments = [
        {
            "id": "indicators_btn",
            "label": "indicators_btn",
            "role": "button",
            "suggested_selector": "button#indicators_btn",
            "bbox": {"x": 0.2, "y": 0.04, "width": 0.07, "height": 0.04},
        }
    ]
    dag = {
        "workflow_name": "HybridTestWorkflow",
        "target_url": "http://localhost:3000/",
        "nodes": {"indicators_btn": {"role": "button"}},
        "edges": [
            {
                "from_node": "state_root",
                "to_node": "indicators_btn",
                "action_type": "click",
                "trigger_element": "indicators_btn",
            }
        ],
    }
    generator = JinjaCodeGenerator()
    generator.emit(dag=dag, segments=segments, out_dir=dist_dir)

    sys.modules.pop("netguard", None)
    sys.modules.pop("playwright_driver", None)
    sys.path.insert(0, str(dist_dir.resolve()))
    try:
        mod = importlib.import_module("playwright_driver")
        yield getattr(mod, "OcularPlaywrightDriver")
    finally:
        sys.path.remove(str(dist_dir.resolve()))
        sys.modules.pop("netguard", None)
        sys.modules.pop("playwright_driver", None)


@pytest.mark.asyncio
async def test_hybrid_driver_semantic_primary(driver_cls):
    driver = driver_cls(viewport_width=1920, viewport_height=1080)

    # Mock Page with a matched semantic role locator
    page = MagicMock()
    mock_locator = AsyncMock()
    mock_locator.count.return_value = 1
    mock_locator.first = mock_locator
    mock_locator.wait_for = AsyncMock()
    mock_locator.click = AsyncMock()

    page.get_by_role = MagicMock(return_value=mock_locator)
    page.locator = MagicMock(return_value=mock_locator)

    res = await driver.click_target(page, "indicators_btn")
    assert res["success"] is True
    assert res["strategy"] == "semantic_accessibility"
    assert res["role"] == "button"
    mock_locator.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_driver_sam_fallback_when_canvas_or_obfuscated(driver_cls):
    driver = driver_cls(viewport_width=1920, viewport_height=1080)

    # Mock Page where semantic locator fails or returns count 0 (e.g. canvas or obfuscated DOM)
    page = MagicMock()
    empty_locator = AsyncMock()
    empty_locator.count.return_value = 0

    page.get_by_role = MagicMock(return_value=empty_locator)
    page.locator = MagicMock(return_value=empty_locator)

    # Mouse click mock
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()

    res = await driver.click_target(page, "indicators_btn")
    assert res["success"] is True
    assert res["strategy"] == "sam_visual_coordinates"
    assert "pixel_coordinates" in res

    # Check denormalized coordinates for indicators_btn (x: 0.2 + 0.07/2 = 0.235, y: 0.04 + 0.04/2 = 0.06)
    # 0.235 * 1920 = 451.2, 0.06 * 1080 = 64.8
    assert res["pixel_coordinates"]["x"] == 451.2
    assert res["pixel_coordinates"]["y"] == 64.8
    page.mouse.click.assert_awaited_once_with(451.2, 64.8)
