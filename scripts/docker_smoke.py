"""Build the image and verify liveness/readiness on a newly migrated database.

The container is isolated from every external network and receives only safe mock
configuration.  This script never supplies provider credentials or enables writes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV_MODULE_PROBE = """
from importlib.util import find_spec

for module in ("pytest", "ruff", "mypy", "pre_commit"):
    assert find_spec(module) is None, f"dev module present in runtime image: {module}"
""".strip()

HEALTH_PROBE_TEMPLATE = """
import http.client
import json

from app.health import validate_health_response

connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=2)
connection.request("GET", {path!r})
response = connection.getresponse()
payload = json.loads(response.read().decode("utf-8"))
validate_health_response({path!r}, response.status, payload)
connection.close()
""".strip()


def health_probe_source(path: str) -> str:
    """Build a probe that delegates validation to the application contract."""

    if path not in {"/health/live", "/health/ready"}:
        raise ValueError(f"Unsupported health path: {path}")
    return HEALTH_PROBE_TEMPLATE.format(path=path)


def run(
    command: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run a Docker command as an argv list, never through a shell."""

    return subprocess.run(  # noqa: S603 - argv is explicit and no shell is involved.
        command,
        cwd=ROOT,
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def container_is_running(docker: str, container: str) -> bool:
    """Return whether Docker still reports the smoke container as running."""

    result = run(
        [docker, "inspect", "--format={{.State.Running}}", container],
        check=False,
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def wait_for_health(docker: str, container: str, path: str, timeout: float) -> None:
    """Poll one loopback-only health endpoint from inside the container."""

    deadline = time.monotonic() + timeout
    last_probe = "health probe did not run"
    while time.monotonic() < deadline:
        if not container_is_running(docker, container):
            logs = run(
                [docker, "logs", container],
                check=False,
                capture_output=True,
                timeout=10,
            )
            raise RuntimeError(
                "Docker default command exited before becoming healthy.\n"
                f"stdout:\n{logs.stdout}\nstderr:\n{logs.stderr}"
            )

        probe = run(
            [docker, "exec", container, "python", "-c", health_probe_source(path)],
            check=False,
            capture_output=True,
            timeout=10,
        )
        if probe.returncode == 0:
            return
        last_probe = (probe.stderr or probe.stdout).strip()
        time.sleep(1)

    raise TimeoutError(
        f"Docker health endpoint {path} was not ready after {timeout:g}s: {last_probe}"
    )


def assert_runtime_has_no_dev_tools(docker: str, container: str) -> None:
    """Fail when a development-only package leaked into the runtime layer."""

    probe = run(
        [docker, "exec", container, "python", "-c", DEV_MODULE_PROBE],
        check=False,
        capture_output=True,
        timeout=10,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(f"Docker runtime dependency isolation failed: {detail}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for local and CI use."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=f"x-content-bot-smoke:{os.getpid()}",
        help="temporary image tag",
    )
    parser.add_argument("--skip-build", action="store_true", help="use an existing image")
    parser.add_argument("--keep-image", action="store_true", help="do not remove the image")
    parser.add_argument("--timeout", type=float, default=45.0, help="health wait in seconds")
    return parser.parse_args()


def main() -> int:
    """Build, start, probe, and clean up a network-isolated container."""

    args = parse_args()
    docker = shutil.which("docker")
    if docker is None:
        print("Docker executable was not found.", file=sys.stderr)
        return 2

    container = f"x-content-bot-smoke-{os.getpid()}"
    built_here = not args.skip_build
    try:
        if built_here:
            run([docker, "build", "--tag", args.image, "."])

        run(
            [
                docker,
                "run",
                "--detach",
                "--name",
                container,
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",  # noqa: S108 - isolated container tmpfs.
                "--env",
                "APP_ENV=test",
                "--env",
                "APP_HOST=127.0.0.1",
                "--env",
                "APP_PORT=8000",
                "--env",
                "DATABASE_URL=sqlite:////tmp/xbot-smoke.db",
                "--env",
                "DATA_DIR=/tmp/data",
                "--env",
                "DRAFTS_DIR=/tmp/drafts",
                "--env",
                "MOCK_MODE=true",
                "--env",
                "LLM_MODE=mock",
                "--env",
                "DRAFT_PROVIDER=mock",
                "--env",
                "CRITIC_PROVIDER=mock",
                "--env",
                "FINAL_PROVIDER=mock",
                "--env",
                "HEYGEN_MODE=disabled",
                "--env",
                "PUBLISH_ENABLED=false",
                "--env",
                "AUTO_PUBLISH=false",
                "--env",
                "STORE_LLM_PAYLOADS=false",
                args.image,
            ],
            capture_output=True,
            timeout=30,
        )
        wait_for_health(docker, container, "/health/live", args.timeout)
        wait_for_health(docker, container, "/health/ready", args.timeout)
        assert_runtime_has_no_dev_tools(docker, container)
        print(
            "Docker liveness/readiness passed on a migrated file database; "
            "the runtime contains no dev tools and no external network or writes were enabled."
        )
        return 0
    finally:
        run([docker, "rm", "--force", container], check=False, capture_output=True, timeout=30)
        if built_here and not args.keep_image:
            run([docker, "image", "rm", args.image], check=False, capture_output=True, timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
