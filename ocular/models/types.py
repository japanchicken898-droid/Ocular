"""Type definitions and data models for Ocular Compiler Engine."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SegmentRole(str, Enum):
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    NAVIGATION = "navigation"
    TEXT = "text"
    MODAL = "modal"
    CONTAINER = "container"
    ICON = "icon"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    HEADER = "header"
    FORM = "form"


class ActionType(str, Enum):
    CLICK = "click"
    FILL = "fill"
    NAVIGATE = "navigate"
    SELECT = "select"
    HOVER = "hover"
    SCROLL = "scroll"
    ASSERT_VISIBLE = "assert_visible"
    WAIT_FOR = "wait_for"


class BoundingBox(BaseModel):
    x: float = Field(..., description="Top-left X coordinate (0.0 to 1.0 or pixel)")
    y: float = Field(..., description="Top-left Y coordinate (0.0 to 1.0 or pixel)")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def bottom_right_x(self) -> float:
        return self.x + self.width

    @property
    def bottom_right_y(self) -> float:
        return self.y + self.height


class VisualSegment(BaseModel):
    id: str = Field(..., description="Unique ID for segment (e.g. seg_login_btn)")
    label: str = Field(..., description="Visual text or human-readable description")
    role: SegmentRole = Field(default=SegmentRole.CONTAINER, description="Inferred UI role")
    bbox: BoundingBox = Field(..., description="Bounding box on screen")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="SAM / vision model confidence")
    suggested_selector: Optional[str] = Field(
        default=None,
        description="Emitted Playwright CSS/Text/Role selector deduced from visual/text cues"
    )
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional DOM/visual metadata")


class FrameMetadata(BaseModel):
    frame_index: int
    timestamp_ms: float
    image_path: Optional[str] = None
    width: int = 1920
    height: int = 1080
    is_keyframe: bool = False
    diff_score: float = 0.0


class TransitionEvent(BaseModel):
    from_frame_index: int
    to_frame_index: int
    timestamp_ms: float
    action_type: ActionType
    target_segment_id: Optional[str] = None
    target_selector: Optional[str] = None
    input_payload: Optional[str] = None
    diff_score: float = 0.0
    confidence: float = 1.0


class DAGNode(BaseModel):
    id: str = Field(..., description="State identifier, e.g. state_login_page")
    title: str = Field(..., description="Descriptive state title")
    screen_name: str = Field(..., description="Unique screen identifier")
    keyframe_index: int = Field(default=0)
    url_pattern: Optional[str] = Field(default=None, description="Expected URL or route")
    segments: List[VisualSegment] = Field(default_factory=list, description="Visual segments on this screen")
    assertions: List[str] = Field(default_factory=list, description="State invariants / assertions")


class DAGEdge(BaseModel):
    from_node_id: str
    to_node_id: str
    action_type: ActionType
    target_segment_id: Optional[str] = None
    target_selector: str = Field(..., description="Playwright selector to trigger transition")
    target_label: Optional[str] = None
    input_payload: Optional[str] = None
    wait_condition: Optional[str] = Field(default=None, description="Wait selector or network idle")
    timeout_ms: int = Field(default=5000)
    description: Optional[str] = None


class StateGraph(BaseModel):
    name: str = Field(default="OcularAutomatedWorkflow")
    initial_node_id: str
    terminal_node_ids: List[str] = Field(default_factory=list)
    nodes: Dict[str, DAGNode] = Field(default_factory=dict)
    edges: List[DAGEdge] = Field(default_factory=list)

    def get_linear_path(self) -> List[DAGEdge]:
        """Returns the primary execution path for code generation."""
        # Follow transitions from initial node to terminal node
        path: List[DAGEdge] = []
        curr = self.initial_node_id
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            outgoing = [e for e in self.edges if e.from_node_id == curr]
            if not outgoing:
                break
            edge = outgoing[0]
            path.append(edge)
            curr = edge.to_node_id
        return path


class CompiledArtifacts(BaseModel):
    playwright_script_path: str
    fastapi_app_path: str
    state_graph_path: str
    manifest_path: str
    dist_dir: str
    steps_count: int
    states_count: int
