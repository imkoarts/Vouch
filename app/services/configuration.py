"""Validated loading for content source and profile configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import ConfigurationError, load_yaml
from app.schemas.configuration import (
    ContentProfileConfiguration,
    ContentSourcesConfiguration,
    CostEstimatesConfiguration,
    RuntimeConfiguration,
)

ConfigurationT = TypeVar("ConfigurationT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ContentConfiguration:
    sources: ContentSourcesConfiguration
    profile: ContentProfileConfiguration
    costs: CostEstimatesConfiguration
    runtime: RuntimeConfiguration


class ConfigurationService:
    """Load fixed local YAML files and reject incomplete or unknown settings."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir

    def load_sources(self) -> ContentSourcesConfiguration:
        return self._load(
            "content_sources.yml",
            ContentSourcesConfiguration,
        )

    def load_profile(self) -> ContentProfileConfiguration:
        return self._load(
            "content_profile.yml",
            ContentProfileConfiguration,
        )

    def load_costs(self) -> CostEstimatesConfiguration:
        return self._load(
            "cost_estimates.yml",
            CostEstimatesConfiguration,
        )

    def load_runtime(self) -> RuntimeConfiguration:
        return self._load("runtime.yml", RuntimeConfiguration)

    def load(self) -> ContentConfiguration:
        return ContentConfiguration(
            sources=self.load_sources(),
            profile=self.load_profile(),
            costs=self.load_costs(),
            runtime=self.load_runtime(),
        )

    def _load(
        self,
        filename: str,
        model: type[ConfigurationT],
    ) -> ConfigurationT:
        path = self.config_dir / filename
        try:
            raw = load_yaml(path)
        except OSError as error:
            raise ConfigurationError(
                f"Unable to read required configuration file {filename}"
            ) from error
        try:
            return model.model_validate(raw)
        except ValidationError as error:
            issues = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
                for issue in error.errors(include_input=False)
            )
            raise ConfigurationError(
                f"Invalid configuration in {filename}: {issues or 'schema validation failed'}"
            ) from error
