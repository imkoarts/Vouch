from __future__ import annotations

import inspect

import app.services.editorial_quality as editorial_quality
import app.services.semantic_adjudication as semantic_adjudication
import app.services.semantic_extraction as semantic_extraction
from app.services.semantic_extraction import (
    bind_event_relations,
    build_clause_frame,
    extract_entity_candidates,
    extract_lexical_atoms,
)


def test_public_extraction_api_delegates_to_composable_stages() -> None:
    source = inspect.getsource(semantic_extraction)

    assert "re.compile" not in source
    assert "extract_compositional_semantics" in source
    assert callable(extract_lexical_atoms)
    assert callable(build_clause_frame)
    assert callable(extract_entity_candidates)
    assert callable(bind_event_relations)


def test_adjudication_remains_ir_only() -> None:
    source = inspect.getsource(semantic_adjudication)

    assert "import re" not in source
    assert "source_text" not in source
    assert "reply_text" not in source


def test_structural_family_projection_uses_typed_shell_for_semantic_families() -> None:
    source = inspect.getsource(editorial_quality._reply_structure_families)

    assert "inspect_reply_shell(text)" in source
    assert '"reductive_identity",\n            re.compile' not in source
    assert '"inverse_praise",\n            re.compile' not in source
    assert '"reveal_reduction",\n            re.compile' not in source
    assert '"setup_twist",\n            re.compile' not in source
