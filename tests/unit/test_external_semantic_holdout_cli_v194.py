from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "run_external_semantic_holdout_v5.py"


def _document(*, schema_version: str = "5.0", mismatch: bool = False) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "holdout_id": "runner-contract-synthetic",
        "cases": [
            {
                "id": "literal-status",
                "source": "The service is available.",
                "reply": "The status is available.",
                "expect": (
                    {"literal_harm": True}
                    if mismatch
                    else {
                        "required_literal_proposition_types": ["status"],
                        "required_predicate_scope_relations": ["matrix"],
                    }
                ),
            }
        ],
    }


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository runner.
        [sys.executable, str(RUNNER), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_native_current_schema_runs_through_the_actual_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "native-current.json"
    _write(input_path, _document())

    completed = _run(str(input_path))

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["all_passed"] is True


def test_semantic_mismatch_returns_exit_code_one(tmp_path: Path) -> None:
    input_path = tmp_path / "semantic-mismatch.json"
    _write(input_path, _document(mismatch=True))

    completed = _run(str(input_path))

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["failed"] == 1


def test_invalid_and_unsupported_inputs_return_precise_exit_code_two(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    invalid = _run(str(invalid_path))
    assert invalid.returncode == 2
    assert "Invalid holdout input" in invalid.stderr

    unsupported_path = tmp_path / "unsupported.json"
    _write(unsupported_path, _document(schema_version="99.0"))
    unsupported = _run(str(unsupported_path))
    assert unsupported.returncode == 2
    assert "unsupported holdout schema version: 99.0" in unsupported.stderr


def test_schema_writer_emits_the_schema_accepted_by_the_same_cli(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    written = _run("--write-schema", str(schema_path))
    assert written.returncode == 0, written.stderr
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    current = schema["properties"]["schema_version"]["const"]

    input_path = tmp_path / "schema-native.json"
    _write(input_path, _document(schema_version=current))
    completed = _run(str(input_path))

    assert completed.returncode == 0, completed.stderr


def test_supported_legacy_document_migrates_directly_to_current_schema(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "legacy-v4.json"
    _write(input_path, _document(schema_version="4.0"))

    completed = _run(str(input_path))

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["schema_version"] == "5.0"
    assert summary["compatibility_migrations"]


def test_built_wheel_runner_executes_outside_checkout(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheel"
    subprocess.run(  # noqa: S603 - fixed interpreter and local build backend.
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
            ".",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    install_dir = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(install_dir)

    input_path = tmp_path / "installed-wheel.json"
    _write(input_path, _document())
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(install_dir)
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and installed module.
        [sys.executable, "-m", "app.semantic_holdout_cli", str(input_path)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["all_passed"] is True
