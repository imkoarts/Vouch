"""Safe diagnostics for local operators without leaking credentials."""

from __future__ import annotations

import traceback
import uuid

from sqlalchemy.exc import OperationalError

from app.domain.security import FactCheckBlockedError, redact_secrets
from app.providers.responses_provider import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderInsufficientQuotaError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderStructuredOutputError,
)
from app.x_api.live import XApiError, XAuthenticationError, XPermissionError, XRateLimitError


def new_error_id() -> str:
    return uuid.uuid4().hex[:8].upper()


def safe_exception_summary(error: BaseException, *, limit: int = 1200) -> str:
    raw = str(error).replace("\x00", " ").strip()
    redacted = redact_secrets(raw)
    text = redacted if isinstance(redacted, str) else type(error).__name__
    text = " ".join(text.split())
    return (text or type(error).__name__)[:limit]


def safe_traceback(error: BaseException, *, limit: int = 20_000) -> str:
    raw = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    redacted = redact_secrets(raw)
    text = redacted if isinstance(redacted, str) else type(error).__name__
    return text[-limit:]


def _provider_label(provider: str) -> str:
    if provider == "openai":
        return "OpenAI"
    if provider == "xai":
        return "Grok 4.5 / xAI"
    return provider


def operator_message(error: BaseException, error_id: str) -> str:
    summary = safe_exception_summary(error, limit=450)
    if isinstance(error, FactCheckBlockedError):
        return (
            "Fact-check confirmation is required before approval or publication. "
            "Reopen the draft and use the verification button; nothing was published. "
            f"[{error_id}]"
        )
    if type(error).__name__ == "PostLengthMismatchError":
        return (
            f"The AI drafts missed the configured character range [{error_id}] even after one "
            "bounded revision. X was not read again. Retry /discover; if this repeats, "
            "switch post length to short or attach logs/vouch.log."
        )
    if isinstance(error, ProviderInsufficientQuotaError):
        provider = _provider_label(error.provider)
        minutes = max(1, int((error.cooldown_seconds + 59) // 60))
        return (
            f"{provider} API quota is unavailable [{error_id}]. Request delays cannot fix this. "
            "Add API credits or check the project billing and usage limit. "
            f"AI requests are paused for about {minutes} minutes, and automatic discovery will "
            "stop before additional paid X reads."
        )
    if isinstance(error, ProviderRateLimitError):
        provider = _provider_label(error.provider)
        return (
            f"Temporary {provider} rate limit [{error_id}]. The bot already made "
            f"{error.attempts} attempts with increasing delays. Try again later."
        )
    if isinstance(error, ProviderStructuredOutputError):
        provider = _provider_label(error.provider)
        return (
            f"{provider} returned invalid structured output [{error_id}]. The bot retried only "
            f"the AI request {error.attempts} time(s) without reading X again. Retry /discover; "
            "if the issue continues, attach logs/vouch.log."
        )
    if isinstance(error, ProviderAuthenticationError):
        provider = _provider_label(error.provider)
        detail = (
            "Check XAI_API_KEY, model access, team ACLs, and regional availability."
            if error.provider == "xai"
            else "Check the API key and project access to the configured model."
        )
        return f"{provider} rejected authentication [{error_id}]. {detail}"
    if isinstance(error, ProviderConnectionError):
        provider = _provider_label(error.provider)
        return (
            f"Could not connect to {provider} [{error_id}]. Check the internet connection, "
            "proxy, and OUTBOUND_PROXY setting."
        )
    if isinstance(error, ProviderRequestError):
        provider = _provider_label(error.provider)
        status = f" HTTP {error.status_code}." if error.status_code is not None else "."
        detail = (
            " Check Grok model availability for the team/region and the model in runtime.yml."
            if error.provider == "xai"
            else " Check the model name and Responses API parameters."
        )
        return f"{provider} rejected the request [{error_id}].{status}{detail}"
    if type(error).__name__ == "BadRequestError" and "invalid_json_schema" in summary:
        return (
            f"OpenAI rejected the response schema [{error_id}]. Update to a fixed release. "
            "Details are in logs/vouch.log."
        )
    if isinstance(error, XPermissionError):
        request_id = f" X request ID: {error.request_id}." if error.request_id else ""
        return (
            f"X denied manual publication [{error_id}]. No post was published. "
            f"{summary}{request_id} Run CONFIGURE_VOUCH.bat after updating the X app permissions "
            "and regenerated user token."
        )
    if isinstance(error, XAuthenticationError):
        return (
            f"X user authentication failed [{error_id}]. No post was published. {summary} "
            "Update the user access credentials in CONFIGURE_VOUCH.bat."
        )
    if isinstance(error, XRateLimitError):
        return (
            f"X write rate limit reached [{error_id}]. No automatic retry was made and nothing "
            "was published. Retry after the provider reset time."
        )
    if isinstance(error, XApiError):
        request_id = f" X request ID: {error.request_id}." if error.request_id else ""
        return (
            f"X rejected manual publication [{error_id}]. No post was published. "
            f"{summary}{request_id}"
        )
    if isinstance(error, OperationalError):
        lowered = summary.casefold()
        if "no such table" in lowered or "no such column" in lowered:
            advice = (
                "The database schema is outdated. Close Vouch and run START_VOUCH.bat again; "
                "migrations are applied automatically."
            )
        elif "database is locked" in lowered or "database table is locked" in lowered:
            advice = "SQLite is locked by another process. Close the second bot instance and retry."
        else:
            advice = "SQLite failed. Details are in logs/vouch.log."
        return f"Database error [{error_id}]. {advice}\nDiagnostic: {summary}"
    return (
        f"Processing error [{error_id}]: {type(error).__name__}. "
        f"Details are in logs/vouch.log.\nDiagnostic: {summary}"
    )
