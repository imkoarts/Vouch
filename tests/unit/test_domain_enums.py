"""All boundary layers reuse the authoritative dependency-free enums."""

from app.domain.enums import ContentType, DraftStatus, FactCheckStatus, MediaType
from app.domain.state_machine import DraftStatus as StateMachineDraftStatus
from app.models.enums import (
    ContentType as ModelContentType,
)
from app.models.enums import (
    DraftStatus as ModelDraftStatus,
)
from app.models.enums import (
    FactCheckStatus as ModelFactCheckStatus,
)
from app.schemas.content import (
    ContentFormat,
)
from app.schemas.content import (
    FactCheckStatus as SchemaFactCheckStatus,
)
from app.schemas.content import (
    MediaType as SchemaMediaType,
)


def test_domain_enums_are_reused_by_state_schema_and_persistence_layers() -> None:
    assert StateMachineDraftStatus is DraftStatus
    assert ModelDraftStatus is DraftStatus
    assert ModelFactCheckStatus is FactCheckStatus
    assert SchemaFactCheckStatus is FactCheckStatus
    assert ModelContentType is ContentType
    assert ContentFormat is ContentType
    assert SchemaMediaType is MediaType
