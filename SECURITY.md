# Security policy

## Trust boundary

Vouch is a local-first review system. It can collect source material and prepare drafts,
but only the operator may approve and confirm a write action. Automatic publication is not
supported.

External X content, imported files, provider responses, and manually edited draft files are
untrusted at the boundary. Embedded commands, URLs, tool requests, and prompt instructions are not
executed.

## Reporting a vulnerability

Do not post secrets, personal data, source content, draft text, databases, or working exploits in a
public issue. Use the hosting platform's private vulnerability-reporting feature when available.

A useful report includes the affected version, synthetic reproduction steps, expected and actual
access boundaries, impact, and a proposed fix.

## Secrets and privacy

- Store real credentials only in a local `.env` or external secret manager.
- Never commit `.env`, logs containing sensitive payloads, databases, drafts, or media bundles.
- Release defaults must not contain Telegram IDs, X user IDs, account names, local usernames,
  email addresses, proxy credentials, API keys, or tokens.
- Log fields matching `TOKEN|KEY|SECRET|PASSWORD` and free-form credential patterns are redacted.
- `STORE_LLM_PAYLOADS=false` is the safe default.
- Rotate any credential exposed in chat, shell history, a commit, a log, or an artifact.

## Generation trust boundary

Topic labels, X posts, search results, trend data, and model responses are not facts by default. The
generation pipeline records source IDs, attribution, verification status, unresolved questions,
and prohibited inferences in a strict evidence packet. Provider output must cite supplied evidence
IDs and cannot silently obtain missing metadata from application defaults.

Prompt instructions embedded in external content remain quoted data. Quality repair reuses the same
evidence packet, never receives publishing tools, and cannot repeat paid X reads.

## Publication invariants

- `AUTO_PUBLISH=true` fails configuration validation.
- Every X write goes through `PublishingService`.
- Publication requires an approved current hash, valid fact-check and safety state, valid media,
  explicit manual enablement, and the exact confirmation phrase.
- Content or media changes revoke approval.
- Ambiguous write outcomes are never retried automatically.
- Remote deletion is a separate explicit workflow.

## Network and budget safety

Only official X APIs are supported. Browser scraping and authentication or rate-limit bypasses are
forbidden. Automatic discovery uses hard source caps and a preflight daily X-read budget. Provider
quota errors open a circuit breaker before future paid X reads.

The web server binds to loopback by default. A non-loopback bind requires configured Supabase
passwordless authentication. Shared dashboard APIs require a hashed, revocable application
session; mutations also require a matching CSRF token. Each authenticated account is routed to an
opaque, containment-checked local workspace with its own SQLite database, drafts, data, logs, and
configuration. Supabase service-role keys are not supported by the application.

Loopback-only local mode still requires an application session. Preview delivery is test-only and
cannot authenticate through the web UI. With `LOCAL_OTP_DELIVERY=smtp`, the raw code is sent through
configured SMTP and is never returned by the API, rendered in the browser, stored, or logged; a delivery failure rolls back the
challenge. The database stores a salted token hash and an email hash, not the raw code or email;
challenges expire after ten minutes, are single-use, rate-limited, and stop after five failed
attempts. Provider credentials exposed in App Settings are write-only, allowlisted, CSRF-protected,
and persisted only in the active account workspace `.env`; email-delivery configuration is
deliberately server-wide and environment-only. For backward compatibility the oldest local account
retains the root workspace. Every later local account is routed to an isolated opaque workspace and
does not inherit the root provider credentials.
The CLI uses the same SMTP-only OTP delivery and stores only an opaque active-account key. Content
commands bind the selected account workspace through request-local context; nested discovery
transactions cannot silently fall back to the root database.
An account-bound automatic discovery loop starts only after both an official X read capability and
the selected generation-provider credential are present. Empty tenants cannot trigger paid reads.

Voice analysis reads at most 50 posts/replies through the official X API, sends them to the
configured OpenAI model with response storage disabled, and persists only structured style features
and a digest. Raw voice samples are not stored in the database.

## Release checklist

Before publishing a release:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy app
python -m pytest
python scripts/build_release.py
```

Then scan the archive for `.env`, keys, tokens, IDs, account names, email addresses, proxy strings,
logs, databases, drafts, caches, virtual environments, Git metadata, and non-English private copy.
