"""Compact, conditional runtime integration for the packaged Personal Humanizer skill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.schemas.content import (
    GenerationRequest,
    GenerationResult,
    HumanizerRevisionTarget,
    QualityReport,
)
from app.services.humanizer_runtime import HumanizerRuntimeLoader

_DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "humanizer.txt"
_DEFAULT_REPLY_RULES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reply_micro.txt"
_MAX_SKILL_BYTES = 128 * 1024
_MAX_RUNTIME_WORDS = 1100


@dataclass(frozen=True, slots=True)
class HumanizerService:
    """Prepare one bounded style revision after deterministic quality rejection.

    Initial drafting receives the compact canonical writing contract through the provider, but this
    service does not choose or generate the idea. Only a rejected draft gets a bounded revision
    request with the exact source variants and named issue codes.
    """

    enabled: bool = True
    mode: str = "compact_conditional"
    rules_path: Path = _DEFAULT_RULES_PATH
    reply_rules_path: Path = _DEFAULT_REPLY_RULES_PATH
    external_skill_path: str | Path | None = None
    include_references: bool = False
    runtime_loader: HumanizerRuntimeLoader | None = None

    @staticmethod
    def _read_limited(path: Path) -> str:
        if path.stat().st_size > _MAX_SKILL_BYTES:
            raise ValueError("Humanizer rules file is larger than the configured safety limit")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _compact(text: str) -> str:
        words = text.split()
        if len(words) <= _MAX_RUNTIME_WORDS:
            return text.strip()
        return " ".join(words[:_MAX_RUNTIME_WORDS]).strip()

    def _loader(self) -> HumanizerRuntimeLoader:
        if self.runtime_loader is not None:
            return self.runtime_loader
        return HumanizerRuntimeLoader(
            configured_path=self.external_skill_path,
            include_references=self.include_references,
        )

    def rules_with_source(self, *, reply_mode: bool = False) -> tuple[str, str]:
        """Load revision rules from the same active runtime used for initial drafting."""

        runtime = self._loader().load()
        rules = (
            "Deletion-first revision. Delete unsupported rhetoric first. "
            "Do not replace deleted rhetoric with a new implication, lesson, or angle. "
            "Preserve the validated contribution ID, evidence IDs, actor, predicate, object, "
            "polarity, speech act, epistemic modality, event status, quantities, dates, "
            "attribution source, and attribution act. Do not drop material details, remove "
            "attribution, or strengthen certainty. If deletion removes the contribution, return "
            "no_post_needed rather than inventing a replacement.\n\n" + runtime.revision_contract
        )
        if reply_mode:
            rules = rules + "\n\n" + self._read_limited(self.reply_rules_path)
        source = (
            "external_runtime_contract"
            if runtime.source == "external"
            else "bundled_reply_runtime"
            if reply_mode
            else "bundled_fallback"
        )
        return self._compact(rules), source

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        """Initial drafting resolves the shared runtime inside the provider."""

        return request

    def prepare_revision_request(
        self,
        request: GenerationRequest,
        *,
        result: GenerationResult,
        reports: Sequence[QualityReport],
        feedback: str,
    ) -> GenerationRequest:
        """Attach exact drafts, per-draft issue codes, and the compact revision contract."""

        if len(result.variants) != len(reports):
            raise ValueError("revision reports must align with generated variants")

        targets: list[HumanizerRevisionTarget] = []
        all_codes: set[str] = set()
        for variant, report in zip(result.variants, reports, strict=True):
            codes = tuple(sorted({issue.code for issue in report.issues}))
            if not codes:
                raise ValueError("a humanizer revision target must contain named issue codes")
            all_codes.update(codes)
            targets.append(HumanizerRevisionTarget(variant=variant, issue_codes=codes))

        instructions = request.instructions
        metadata = {
            **request.metadata,
            "humanizer_revision": True,
            "humanizer_issue_codes": sorted(all_codes),
        }
        if self.enabled and self.mode != "disabled":
            runtime = self._loader().load()
            rules, source = self.rules_with_source(
                reply_mode=(
                    request.generation_mode.value == "reply"
                    or request.content_type.value == "reply"
                )
            )
            instructions = (*instructions, "Compact humanizer revision contract:\n" + rules)
            metadata.update(
                {
                    "humanizer": self.mode,
                    "humanizer_source": source,
                    "humanizer_version": runtime.version,
                    "humanizer_skill_hash": runtime.skill_hash,
                }
            )
        else:
            metadata.update({"humanizer": "disabled", "humanizer_source": "none"})

        return request.model_copy(
            update={
                "feedback": feedback,
                "instructions": instructions,
                "metadata": metadata,
                "revision_targets": tuple(targets),
                "revision_issue_codes": tuple(sorted(all_codes)),
            }
        )

    @staticmethod
    def _legacy_cleanup(text: str) -> str:
        import re

        cleaned = text.strip()
        cleaned = re.sub(
            r"^(?:In today'?s rapidly changing landscape|It is important to note that|"
            r"There is a report that)\s*[,.:—-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def process_result(self, result: GenerationResult) -> GenerationResult:
        """Preserve modern results; legacy cleanup remains only for explicit compatibility mode."""

        if not self.enabled or self.mode != "prompt_and_cleanup":
            return result
        variants = []
        for variant in result.variants:
            if variant.parts:
                parts = tuple(self._legacy_cleanup(part) or part.strip() for part in variant.parts)
                text = "\n\n".join(parts)
            else:
                parts = ()
                text = self._legacy_cleanup(variant.text) or variant.text.strip()
            variants.append(
                variant.model_copy(
                    update={"text": text, "parts": parts, "character_count": len(text)}
                )
            )
        return result.model_copy(update={"variants": tuple(variants)})
