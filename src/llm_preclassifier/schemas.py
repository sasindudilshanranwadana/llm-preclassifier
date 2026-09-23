"""Versioned public decision models."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal[
    "chat", "classification", "coding", "extraction", "planning",
    "reasoning", "research", "summarization", "writing", "agent_action", "unknown",
]
Complexity = Literal["trivial", "simple", "moderate", "complex", "unknown"]
ToolRequirement = Literal["none", "optional", "required", "unknown"]
RecommendedModelTier = Literal["economy", "standard", "capable", "reasoning", "human_or_policy_review", "unknown"]
DecisionAction = Literal["route", "escalate", "unknown"]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = Field(min_length=1, max_length=64)
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)
    available_tools: list[str] = Field(default_factory=list, max_length=64)
    policy_flags: list[str] = Field(default_factory=list, max_length=32)


class ClassificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"] = "v1"
    task_type: TaskType
    complexity: Complexity
    tool_requirement: ToolRequirement
    recommended_model_tier: RecommendedModelTier
    confidence: float = Field(ge=0.0, le=1.0)
    action: DecisionAction
    reasons: list[str] = Field(min_length=1, max_length=16)
    policy_version: str = Field(default="v1", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class HealthResponse(BaseModel):
    status: Literal["ok"]
