"""Passwordless provider adapter and hashed local web sessions."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import smtplib
import ssl
from dataclasses import dataclass
from datetime import timedelta
from email.message import EmailMessage
from typing import Any, Protocol

import certifi
import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AuthSession, LocalOtpChallenge, UserAccount
from app.models.base import utc_now


class AuthenticationError(ValueError):
    """Return a stable boundary error without leaking provider payloads."""


@dataclass(frozen=True)
class VerifiedIdentity:
    subject: str
    email: str


@dataclass(frozen=True)
class IssuedSession:
    token: str
    csrf_token: str
    expires_in_seconds: int


class OtpDelivery(Protocol):
    async def send(self, *, email: str, token: str, lifetime_minutes: int) -> None: ...


class SmtpOtpDelivery:
    """Deliver a local passwordless code without persisting or logging the raw value."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        from_email: str,
        username: str | None = None,
        password: str | None = None,
        starttls: bool = True,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.host = host
        self.port = port
        self.from_email = from_email
        self.username = username
        self.password = password
        self.starttls = starttls
        self.timeout_seconds = timeout_seconds

    async def send(self, *, email: str, token: str, lifetime_minutes: int) -> None:
        message = EmailMessage()
        message["Subject"] = "Your Vouch sign-in code"
        message["From"] = self.from_email
        message["To"] = email
        message.set_content(
            "Use this one-time code to sign in to Vouch:\n\n"
            f"{token}\n\nThe code expires in {lifetime_minutes} minutes. "
            "If you did not request it, ignore this email."
        )

        def deliver() -> None:
            try:
                with smtplib.SMTP(self.host, self.port, timeout=self.timeout_seconds) as client:
                    if self.starttls:
                        # Python's Windows trust-store integration differs between supported
                        # runtimes. Use the locked CA bundle already shipped with Vouch so the
                        # packaged Python 3.13 runtime verifies SMTP STARTTLS consistently.
                        client.starttls(context=ssl.create_default_context(cafile=certifi.where()))
                    if self.username:
                        client.login(self.username, self.password or "")
                    client.send_message(message)
            except smtplib.SMTPAuthenticationError as exc:
                raise AuthenticationError(
                    "SMTP authentication failed. Check the provider username and API key."
                ) from exc
            except smtplib.SMTPSenderRefused as exc:
                raise AuthenticationError(
                    "The SMTP provider rejected the From address. Check the configured sender."
                ) from exc
            except smtplib.SMTPRecipientsRefused as exc:
                raise AuthenticationError(
                    "The SMTP provider rejected the recipient. With resend.dev, use the email "
                    "address that owns the Resend account."
                ) from exc
            except smtplib.SMTPDataError as exc:
                if self.host.casefold() == "smtp.resend.com":
                    raise AuthenticationError(
                        "The Resend test sender can email only the address used to create the "
                        "Resend account. Use that address, or verify a domain and update the "
                        "From email."
                    ) from exc
                raise AuthenticationError(
                    "The SMTP provider rejected the message. Check the configured From address."
                ) from exc
            except ssl.SSLCertVerificationError as exc:
                raise AuthenticationError(
                    "Vouch could not verify the SMTP provider TLS certificate."
                ) from exc
            except (OSError, smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected) as exc:
                raise AuthenticationError(
                    "Vouch could not connect to the SMTP provider. Check host, port, and TLS."
                ) from exc
            except smtplib.SMTPException as exc:
                raise AuthenticationError(
                    "The SMTP provider could not deliver the sign-in email. Check its dashboard."
                ) from exc

        await asyncio.to_thread(deliver)


class SupabasePasswordlessClient:
    """Minimal official Supabase Auth REST adapter; no service-role credential is used."""

    def __init__(
        self,
        *,
        base_url: str,
        publishable_key: str,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.publishable_key = publishable_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url}/auth/v1/{path}",
                    headers={"apikey": self.publishable_key},
                    json=payload,
                )
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AuthenticationError("Passwordless authentication failed") from exc
        if not isinstance(value, dict):
            raise AuthenticationError("Passwordless authentication returned invalid data")
        return value

    async def request_otp(self, email: str) -> None:
        await self._post("otp", {"email": email, "create_user": True})

    async def verify_otp(self, email: str, token: str) -> VerifiedIdentity:
        value = await self._post("verify", {"email": email, "token": token, "type": "email"})
        user = value.get("user")
        if not isinstance(user, dict):
            raise AuthenticationError("Passwordless authentication did not return a user")
        subject = str(user.get("id") or "").strip()
        verified_email = str(user.get("email") or "").strip().casefold()
        if not subject or verified_email != email.casefold():
            raise AuthenticationError("Passwordless identity did not match the request")
        return VerifiedIdentity(subject=subject, email=verified_email)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LocalPasswordlessService:
    """Issue and verify one-time codes for a loopback local installation.

    Only a salted digest is persisted; codes expire quickly, are single-use, and have a bounded
    number of attempts. The caller must deliver the raw value through the configured email adapter;
    the web boundary never returns it.
    """

    def __init__(
        self,
        session: Session,
        *,
        lifetime_minutes: int = 10,
        maximum_attempts: int = 5,
        minimum_interval_seconds: int = 60,
        hourly_limit: int = 5,
    ) -> None:
        self.session = session
        self.lifetime = timedelta(minutes=lifetime_minutes)
        self.maximum_attempts = maximum_attempts
        self.minimum_interval = timedelta(seconds=minimum_interval_seconds)
        self.hourly_limit = hourly_limit

    @staticmethod
    def _token_digest(*, email: str, token: str, salt: str) -> str:
        return _hash(f"{salt}:{email.casefold()}:{token}")

    def request(self, email: str) -> str:
        now = utc_now()
        email_hash = _hash(email.casefold())
        latest = self.session.scalar(
            select(LocalOtpChallenge.created_at)
            .where(LocalOtpChallenge.email_hash == email_hash)
            .order_by(LocalOtpChallenge.created_at.desc())
            .limit(1)
        )
        if latest is not None and latest > now - self.minimum_interval:
            raise AuthenticationError("Please wait before requesting another sign-in code")
        recent_count = self.session.scalar(
            select(func.count(LocalOtpChallenge.id)).where(
                LocalOtpChallenge.email_hash == email_hash,
                LocalOtpChallenge.created_at >= now - timedelta(hours=1),
            )
        )
        if int(recent_count or 0) >= self.hourly_limit:
            raise AuthenticationError("Too many sign-in codes were requested; try again later")
        token = f"{secrets.randbelow(1_000_000):06d}"
        salt = secrets.token_hex(16)
        self.session.add(
            LocalOtpChallenge(
                email_hash=email_hash,
                token_hash=self._token_digest(email=email, token=token, salt=salt),
                salt=salt,
                created_at=now,
                expires_at=now + self.lifetime,
            )
        )
        self.session.flush()
        return token

    def verify(self, email: str, token: str) -> VerifiedIdentity:
        challenge = self.session.scalar(
            select(LocalOtpChallenge)
            .where(
                LocalOtpChallenge.email_hash == _hash(email.casefold()),
                LocalOtpChallenge.consumed_at.is_(None),
            )
            .order_by(LocalOtpChallenge.created_at.desc())
            .limit(1)
        )
        now = utc_now()
        if challenge is None or challenge.expires_at <= now:
            raise AuthenticationError("The local sign-in code is missing or expired")
        if challenge.failed_attempts >= self.maximum_attempts:
            raise AuthenticationError("Too many invalid local sign-in attempts")
        challenge.failed_attempts += 1
        expected = self._token_digest(email=email, token=token.strip(), salt=challenge.salt)
        if not secrets.compare_digest(challenge.token_hash, expected):
            raise AuthenticationError("The local sign-in code is invalid")
        challenge.consumed_at = now
        return VerifiedIdentity(subject=f"local:{_hash(email.casefold())}", email=email.casefold())


class AuthSessionService:
    def __init__(self, session: Session, *, lifetime_hours: int = 168) -> None:
        self.session = session
        self.lifetime = timedelta(hours=lifetime_hours)

    def issue(
        self, identity: VerifiedIdentity, *, provider: str
    ) -> tuple[UserAccount, IssuedSession]:
        user = self.session.scalar(
            select(UserAccount).where(
                UserAccount.auth_provider == provider,
                UserAccount.auth_subject == identity.subject,
            )
        )
        if user is None:
            user = UserAccount(
                auth_provider=provider,
                auth_subject=identity.subject,
                email=identity.email,
            )
            self.session.add(user)
            self.session.flush()
        elif user.email != identity.email:
            user.email = identity.email
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(24)
        now = utc_now()
        self.session.add(
            AuthSession(
                token_hash=_hash(token),
                csrf_hash=_hash(csrf_token),
                user_id=user.id,
                created_at=now,
                expires_at=now + self.lifetime,
            )
        )
        return user, IssuedSession(
            token=token,
            csrf_token=csrf_token,
            expires_in_seconds=int(self.lifetime.total_seconds()),
        )

    def resolve(self, token: str | None) -> UserAccount | None:
        if not token:
            return None
        record = self.session.get(AuthSession, _hash(token))
        if record is None or record.revoked_at is not None or record.expires_at <= utc_now():
            return None
        return self.session.get(UserAccount, record.user_id)

    def verify_csrf(self, token: str | None, csrf_token: str | None) -> bool:
        if not token or not csrf_token:
            return False
        record = self.session.get(AuthSession, _hash(token))
        return bool(record and secrets.compare_digest(record.csrf_hash, _hash(csrf_token)))

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        record = self.session.get(AuthSession, _hash(token))
        if record is not None and record.revoked_at is None:
            record.revoked_at = utc_now()
