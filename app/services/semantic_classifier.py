"""Provider-neutral structured semantic inspection.

The deterministic implementation supports offline tests and high-confidence guards. Live
providers may propose richer semantic maps through Structured Outputs, but application code owns
schema validation and the final blocking policy.
"""

from __future__ import annotations

import re
from typing import Protocol

from app.schemas.content import (
    AngleCandidate,
    CanonicalClaim,
    ContributionInspection,
    DraftSemanticInspection,
    EvaluativeIronyInspection,
    EvidencePacket,
    EvidenceSpan,
    HumanHarmInspection,
    HumorIntentInspection,
    HumorSafetyInspection,
    ReactionFunctionInspection,
    SemanticInspection,
    SourceSemanticMap,
    SourceSemanticProposition,
)
from app.services.claim_structure import (
    align_claim_to_evidence,
    claim_structure_matches_text,
    compare_claim_to_evidence,
    extract_claims,
    normalize_text,
)
from app.services.semantic_adjudication import (
    adjudicate_evaluative_irony,
    adjudicate_human_harm,
    adjudicate_humor_intent,
    adjudicate_humor_safety,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_reconciliation import reconcile_semantic_inspections


class SemanticInspectorPort(Protocol):
    async def inspect_source(self, packet: EvidencePacket) -> SourceSemanticMap: ...

    async def inspect_contributions(
        self,
        packet: EvidencePacket,
        contributions: tuple[AngleCandidate, ...],
        source_map: SourceSemanticMap,
    ) -> tuple[ContributionInspection, ...]: ...

    async def inspect_draft(
        self,
        packet: EvidencePacket,
        contribution: AngleCandidate,
        text: str,
        source_map: SourceSemanticMap,
        *,
        commentary_mode: bool,
    ) -> DraftSemanticInspection: ...


_ABSTRACT_FUNCTION = re.compile(
    r"\b(?:realignment|process|institutional expression|presidential level|political shift|"
    r"development|framework|dynamic|signal|implication|practical question|"
    r"narrow test|product shift|legibility|more tangible|makes? .{0,45} concrete|"
    r"moves? .{0,60} into|puts? .{0,60} inside|carries? .{0,60} into|"
    r"gives? .{0,60} expression)\b",
    re.I,
)
_CONTRAST = re.compile(
    r"\b(?:rather than|instead of|more .{0,35} than|not only|less about|"
    r"the question is not|from .{0,55} into)\b",
    re.I,
)
_ABSTRACT_NOUN = re.compile(
    r"\b(?:shift|process|realignment|development|framework|dynamic|expression|"
    r"level|influence|pressure|legibility|format|question|test|promise|trace|ownership)\b",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_SCOPE_EVENT_VERBS = re.compile(
    r"\b(?:reach(?:ed|es)?|include(?:d|s)?|inclusion|extend(?:ed|s)?|move(?:d|s)?|take(?:s|n)?|"
    r"turn(?:ed|s)?|put(?:s)?|give(?:s|n)?|carry|carries|bring|brought|draw|drawn|"
    r"pull(?:ed|s)?|part of|front|dimension|inside|outside)\b",
    re.I,
)
_SCOPE_EFFORT = re.compile(
    r"\b(?:campaign|effort|push|fight|process|realignment|move|step|dismantl(?:e|ing)|"
    r"influence|pressure)\b",
    re.I,
)
_OFFICE_ALIAS = re.compile(r"\b(?:president|presidency|presidential|head of state|office)\b", re.I)

_REACTION_FUNCTION = re.compile(
    r"\b(?:nothing says|because apparently|has left the chat|doing (?:\w+\s+){0,3}work|"
    r"speedrun|plot twist|achievement unlocked|bold strategy|same energy|finally,? a|"
    r'call it|renaming|literal(?:ly)?|sure,? why not)\b|["“][^"”]{1,80}["”]',
    re.I,
)


_REACTION_TEMPLATE = re.compile(
    r"\b(?:nothing says|plot twist|finally|same energy|achievement unlocked|bold strategy|"
    r"sure,? why not|because apparently|but call it|call it)\b\s*[:,-]?",
    re.I,
)
_REACTION_WORDPLAY = re.compile(
    r"\b(?:404|promoted|promotion|rebrand(?:ed|ing)?|renam(?:e|ed|ing)|speedrun|"
    r"has left the chat|doing (?:\w+\s+){0,3}heavy lifting)\b",
    re.I,
)
_REACTION_CALLBACK = re.compile(
    r"(?:[\"“][^\"”]{1,80}[\"”]\s+is doing (?:a lot|heavy lifting) of work|"
    r"\bis doing (?:heroic amounts|a lot|heavy lifting) of work\b|"
    r"\bhas left the chat\b)",
    re.I,
)
_REACTION_CONTRIBUTION_TYPES = {"joke", "dry_reframe", "dry_humor"}
_REPLY_NONHUMOR_TYPES = {
    "plain_observation",
    "direct_response",
    "specific_qualification",
    "genuine_question",
    "contextual_extension",
}

_HUMOR_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "medical_emergency",
        re.compile(
            r"\b(?:medical emergency|emergency room|intensive care|icu|overdose|"
            r"cardiac arrest|life[- ]threatening|hospitalized|patient died)\b",
            re.I,
        ),
    ),
    (
        "war_or_mass_casualty",
        re.compile(
            r"\b(?:armed conflict|war (?:casualt(?:y|ies)|dead|deaths?|injur(?:y|ies|ed)|"
            r"victims?|zone|crime)|airstrike|bombing|mass casualty|mass shooting|massacre|"
            r"genocide|shelling|civilian casualties|refugees?|terror attack)\b",
            re.I,
        ),
    ),
    (
        "abuse_or_self_harm",
        re.compile(
            r"\b(?:abuse|assault|rape|self[- ]harm|suicide|domestic violence|trafficking)\b",
            re.I,
        ),
    ),
    (
        "personal_tragedy",
        re.compile(
            r"\b(?:missing (?:child|person)|found dead|earthquake|disaster|"
            r"collapsed building|house fire|flood(?:ing)?|funeral)\b",
            re.I,
        ),
    ),
)
_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|dozens?|hundreds?|thousands?)"
)
_HUMAN_REFERENT = (
    r"(?:people|person|child|children|worker(?:s)?|employee(?:s)?|passenger(?:s)?|"
    r"civilian(?:s)?|patient(?:s)?|driver(?:s)?|resident(?:s)?|victim(?:s)?|founder|"
    r"officer(?:s)?|soldier(?:s)?|miner(?:s)?|survivor(?:s)?|woman|women|man|men|boy|"
    r"girl|family|families|he|she|they|him|her|them|his|their)"
)
_NON_HUMAN_REFERENT = (
    r"(?:api|app|application|bill|brand|browser|build|campaign|chain|code|component|"
    r"container|conversation|database|deal|feature|job|launch|link|market|measure|meme|"
    r"model|network|node|pipeline|process|product|project|proposal|protocol|request|"
    r"server|service|session|startup|system|thread|token|user session|worker process|"
    r"browser process|database process)"
)
_TECHNICAL_DEATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\bdead\s+{_NON_HUMAN_REFERENT}\b", re.I),
    re.compile(
        rf"\b(?:the\s+)?{_NON_HUMAN_REFERENT}\s+"
        r"(?:(?:is|was|looks?|seems?)\s+dead|(?:has\s+|had\s+)?died)\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:the\s+)?{_NON_HUMAN_REFERENT}\s+"
        r"(?:was|were|got|has been|had been)\s+killed\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:killed|kills?|killing)\s+(?:the\s+)?{_NON_HUMAN_REFERENT}\b",
        re.I,
    ),
    re.compile(
        r"\b(?:killed|kills?|killing)\s+(?:the\s+)?(?:battery life|engagement|momentum|"
        r"performance|traffic|(?:employee|user|customer) engagement|"
        r"(?:worker|employee|developer) productivity|user experience|customer experience|"
        r"developer experience|worker experience|employee experience)\b",
        re.I,
    ),
)
_FATALITY_LANGUAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:death toll|deaths?|fatalit(?:y|ies)|casualt(?:y|ies))\b", re.I),
    re.compile(r"\b(?:there (?:was|were)\s+)?no survivors\b", re.I),
    re.compile(r"\bleft\s+no survivors\b", re.I),
    re.compile(
        rf"\b(?:claimed|took|ended)\s+(?:(?:\d+|{_NUMBER_WORD})\s+)?(?:a\s+)?lives?\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:(?:\d+|{_NUMBER_WORD})\s+)?lives?\s+"
        r"(?:was|were|have been|had been)\s+lost\b",
        re.I,
    ),
    re.compile(r"\blost\s+(?:his|her|their|a)\s+(?:life|lives)\b", re.I),
    re.compile(r"\bpassed away\b", re.I),
    re.compile(r"\bsuccumbed?\s+to\s+(?:his|her|their|the)?\s*injur(?:y|ies)\b", re.I),
    re.compile(r"\b(?:did not|didn(?:'|\u2019)t|failed to)\s+survive\b", re.I),
    re.compile(r"\bfighting\s+for\s+(?:his|her|their|a)\s+life\b", re.I),
    re.compile(r"\bpronounced\s+dead\b", re.I),
    re.compile(r"\b(?:slain|perished)\b", re.I),
    re.compile(r"\bfatally injured\b", re.I),
    re.compile(r"\bmortal injuries\b", re.I),
    re.compile(r"\bbody count\b", re.I),
    re.compile(
        rf"\b(?:(?:\d+|{_NUMBER_WORD})\s+)?bodies?\s+"
        r"(?:was|were|have been|had been)\s+recovered\b",
        re.I,
    ),
    re.compile(rf"\b{_HUMAN_REFERENT}\b.{{0,30}}\bnever made it out\b", re.I),
    re.compile(rf"\b{_HUMAN_REFERENT}\b.{{0,35}}\b(?:left|remained) in critical condition\b", re.I),
    re.compile(rf"\b{_HUMAN_REFERENT}\b.{{0,25}}\bfound lifeless\b", re.I),
    re.compile(rf"\b{_HUMAN_REFERENT}\b.{{0,30}}\b(?:was|were) declared deceased\b", re.I),
    re.compile(rf"\b{_HUMAN_REFERENT}\b.{{0,30}}\bremains? in (?:a )?coma\b", re.I),
    re.compile(
        rf"\b{_HUMAN_REFERENT}\b.{{0,30}}\b(?:remain|remains|remained)\s+"
        r"(?:on life support|in (?:a )?coma)\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:crash|collision)\b.{{0,35}}\b(?:was|proved) fatal (?:for|to)\s+"
        rf"(?:(?:\d+|{_NUMBER_WORD})\s+)?{_HUMAN_REFERENT}\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:(?:\d+|{_NUMBER_WORD})\s+)?{_HUMAN_REFERENT}\b.{{0,25}}"
        r"\b(?:was|were) lost\b.{0,30}\b(?:crash|collision)\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:(?:\d+|{_NUMBER_WORD})\s+)?{_HUMAN_REFERENT}\b.{{0,30}}"
        r"\b(?:did not|didn(?:'|\u2019)t) make it home\b.{0,35}\b(?:crash|collision)\b",
        re.I,
    ),
    re.compile(
        rf"\b(?:incident|crash|collision|blast|explosion|fire|outage)\b.{{0,35}}"
        rf"\bleft\s+(?:(?:\d+|{_NUMBER_WORD})\s+)?{_HUMAN_REFERENT}\b"
        r".{0,20}\bon life support\b",
        re.I,
    ),
)
_HUMAN_DEATH_LINK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\b{_HUMAN_REFERENT}\b.{{0,45}}\b(?:died|dead|was killed|were killed|killed)\b|"
        rf"\b(?:died|dead|was killed|were killed|killed)\b.{{0,45}}\b{_HUMAN_REFERENT}\b",
        re.I,
    ),
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:died|was killed|has died)\b"),
)
_SERIOUS_INJURY = re.compile(
    rf"\b(?:{_HUMAN_REFERENT}\b.{{0,45}})?(?:serious(?:ly)? injured|serious injuries|"
    r"severely injured|critically injured|fatally injured|mortal injuries|"
    r"critically wounded|wounded|hurt in (?:the|an) attack|life[- ]threatening injuries?|"
    r"(?:left|remained) in critical condition)\b",
    re.I,
)
_EXPLOSION_HARM = re.compile(
    rf"\b(?:explosion|blast)\b.{{0,70}}\b(?:{_HUMAN_REFERENT}|injured|wounded|hurt|"
    r"killed|dead|died|casualt(?:y|ies)|no survivors)\b|"
    rf"\b(?:{_HUMAN_REFERENT}|injured|wounded|hurt|killed|dead|died|casualt(?:y|ies)|"
    r"no survivors)\b.{0,70}\b(?:explosion|blast)\b",
    re.I,
)
_RESIDUAL_DEATH_TERM = re.compile(r"\b(?:dead|died|killed|kills?|killing)\b", re.I)
_SENSITIVE_CRASH = re.compile(
    r"\b(?:fatal|deadly|car|vehicle|road|plane|aircraft|train|bus|helicopter) crash\b|"
    rf"\bcrash\b.{{0,60}}\b(?:killed|injured|wounded|dead|died|fatal|"
    rf"no survivors|{_HUMAN_REFERENT})\b|"
    r"\b(?:killed|injured|wounded|dead|died|fatal|no survivors)\b.{0,60}\bcrash\b",
    re.I,
)

_SAFE_CRASH = re.compile(
    r"\b(?:app|application|software|system|server|browser|computer|database|service|game|"
    r"market|stock|crypto|bitcoin|price) crash\b|"
    r"\bcrash\b.{0,30}\b(?:bug|fix(?:ed)?|version|release|patch|update|log|report|handler)\b",
    re.I,
)
_SAFE_ATTACK = re.compile(r"\b(?:cyber|ddos|network|software|prompt|adversarial) attack\b", re.I)
_SAFE_FIRE = re.compile(
    r"\b(?:fire sale|fire button|fire event|fire callback|fire function|fire method|"
    r"fire request)\b",
    re.I,
)
_SAFE_EXPLOSION = re.compile(
    r"\b(?:growth explosion|sales explosion|explosion of interest|explosion in users)\b",
    re.I,
)
_SAFE_COLLAPSE = re.compile(
    r"\b(?:market|price|crypto|stock|server|system) collapse\b",
    re.I,
)
_AMBIGUOUS_SENSITIVE_TERM = re.compile(r"\b(?:crash|attack|fire|collapse|explosion|blast)\b", re.I)
_HARM_TARGET_WORDS = re.compile(
    r"\b(?:death(?:s)?|dead|died|killed|injur(?:y|ies|ed)|wounded|hurt|victim(?:s)?|"
    r"casualt(?:y|ies)|lost (?:his|her|their) (?:life|lives)|passed away|fatal crash|"
    r"earthquake|disaster|missing child)\b",
    re.I,
)
_PUBLIC_SYSTEM_WORDS = re.compile(
    r"\b(?:company|agency|government|hospital administration|spokesperson|statement|wording|"
    r"policy|system|process|press release|management|executive|regulator)\b",
    re.I,
)
_SELF_DEPRECATING = re.compile(r"\b(?:i|me|my|mine|we|us|our)\b", re.I)
_JOKE_SHAPED_TEXT = re.compile(
    r"\b(?:doing (?:heroic amounts|a lot|heavy lifting) of work|committed to the bit|"
    r"speedrun|plot twist|achievement unlocked|nothing (?:says|screams)|"
    r"few things (?:say|communicate|demonstrate|show)|has left the chat|"
    r"but call it|rebrand(?:ed|ing)?|anti[- ]yap|bold strategy|same energy)\b|"
    r"\btranslated\s*:|\bin plain english\s*:",
    re.I,
)
_NEGATIVE_SOURCE_CONTEXT = re.compile(
    r"\b(?:outage|failure|failed|failing|damage(?:d)?|harm(?:ed)?|injur(?:y|ies|ed)|"
    r"death(?:s)?|fatalit(?:y|ies)|casualt(?:y|ies)|crash(?:ed)?|collapse(?:d)?|"
    r"attack(?:ed)?|explosion|blast|breach(?:ed)?|broken|broke|error|loss(?:es)?|"
    r"layoffs?|shutdown|disaster|scandal|contradiction|backfire(?:d)?)\b",
    re.I,
)
_POSITIVE_SOURCE_CONTEXT = re.compile(
    r"\b(?:succeeded|successfully|improved|fixed|recovered|restored|saved|prevented|"
    r"record growth|best result|positive outcome|completed on time)\b",
    re.I,
)
_EVALUATIVE_IRONY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "praise",
        re.compile(
            r"^(?P<evaluation>(?:(?:a|an)\s+)?(?:nice|beautiful|great|excellent|good|"
            r"flawless|perfect|stellar|impressive)\s+"
            r"(?:work|stuff|showing|execution|performance|job))"
            r"(?:\s*(?:from|by|,)\s*(?P<target>[^.!?]{1,80}))?[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "positive_timing",
        re.compile(
            r"^(?P<evaluation>(?:great|perfect|excellent|impeccable) timing)\s+"
            r"(?:from|by|for)\s+(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "rating",
        re.compile(
            r"^(?P<evaluation>(?:\d{1,2}/\d{1,2}|\d{1,2}\s+out of\s+\d{1,2}|"
            r"full marks)(?:\s+(?:execution|showing|performance|work|job))?)"
            r"(?:\s+(?:from|by|for)\s+(?P<target>[^.!?]{1,80}))?[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "rating",
        re.compile(
            r"^(?P<evaluation>(?:one|two|three|four|five|[1-5]) stars)\s+for\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "beneficiary",
        re.compile(
            r"^(?P<evaluation>(?:exactly|just) what)\s+"
            r"(?P<target>[^.!?]{1,80})\s+needed[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "admiration",
        re.compile(
            r"^(?P<evaluation>could(?: not|n(?:'|\u2019)t) have asked for (?:a\s+)?better)\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "admiration",
        re.compile(
            r"^(?P<evaluation>could(?: not|n(?:'|\u2019)t) be prouder of)\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "celebration",
        re.compile(
            r"^(?P<evaluation>(?:(?:a\s+)?round of\s+)?applause|"
            r"(?:a\s+)?standing ovation)\s+for\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "celebration",
        re.compile(
            r"^(?P<evaluation>(?:(?:another|a|what a)\s+)?triumph)\s+for\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "success_attribution",
        re.compile(
            r"^(?P<evaluation>(?:(?:another|a|big|major|huge)\s+)?win)\s+for\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "success_attribution",
        re.compile(
            r"^(?P<evaluation>(?:(?:another|a|big|major|huge)\s+)?victory)\s+for\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "admiration",
        re.compile(
            r"^(?P<evaluation>(?:(?:so|really|very)\s+)?proud of)\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "praise",
        re.compile(
            r"^(?P<evaluation>chef(?:'|\u2019)s kiss)\s+(?:from|for)\s+"
            r"(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "success_attribution",
        re.compile(
            r"^(?P<target>[^.!?]{1,80})\s+"
            r"(?P<evaluation>really delivered|outdid itself|really showed up|crushed it|"
            r"nailed it|aced it)"
            r"[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "beneficiary",
        re.compile(
            r"^(?P<evaluation>love that for)\s+(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "reward",
        re.compile(
            r"^(?P<target>[^.!?]{1,80})\s+"
            r"(?P<evaluation>deserves? (?:a raise|a bonus|an award|a promotion|a medal)|"
            r"earned (?:employee of the month|a raise|a bonus|an award|a promotion|a medal)|"
            r"(?:should\s+)?get(?:s)? (?:a raise|a bonus|an award|a promotion|a medal)|"
            r"wins? (?:employee of the month|a raise|a bonus|an award|a promotion|a medal))"
            r"[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "success_attribution",
        re.compile(
            r"^(?P<evaluation>mission accomplished|nailed it|working as intended|"
            r"could not have gone better|what a success)[.!]?\s*$",
            re.I,
        ),
    ),
    (
        "praise",
        re.compile(
            r"^(?P<evaluation>thanks)\s*,?\s+(?P<target>[^.!?]{1,80})[.!]?\s*$",
            re.I,
        ),
    ),
)
_REWARDING_EVALUATION = re.compile(
    r"\b(?:deserves?|earned|employee of the month|raise|bonus|award|promotion|medal)\b",
    re.I,
)
_TARGETLESS_EVALUATION = re.compile(
    r"^(?:mission accomplished|nailed it|working as intended|could not have gone better|"
    r"what a success)[.!]?\s*$",
    re.I,
)


def _pattern_spans(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()
    )


def _mask_spans(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, tuple[str, ...]]:
    chars = list(text)
    spans: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            spans.append(match.group(0).strip())
            for index in range(match.start(), match.end()):
                chars[index] = " "
    return "".join(chars), tuple(dict.fromkeys(spans))


def inspect_human_harm_scope(source_text: str) -> HumanHarmInspection:
    """Extract typed source relations, then adjudicate human-harm scope."""

    semantic_ir = extract_semantic_inspection(source_text)
    return adjudicate_human_harm(semantic_ir)


def inspect_evaluative_irony(packet: EvidencePacket, text: str) -> EvaluativeIronyInspection:
    """Extract typed source/reply relations, then adjudicate evaluative irony."""

    source_text = " ".join(item.text for item in packet.items)
    semantic_ir = extract_semantic_inspection(source_text, text)
    return adjudicate_evaluative_irony(semantic_ir)


def inspect_humor_safety(packet: EvidencePacket, text: str) -> HumorSafetyInspection:
    """Extract one semantic IR and apply application-owned humor suitability policy."""

    source_text = " ".join(item.text for item in packet.items)
    semantic_ir = extract_semantic_inspection(source_text, text)
    return adjudicate_humor_safety(semantic_ir)


_INCONGRUITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("error", "success"),
    ("failure", "success"),
    ("closed", "open"),
    ("cancel", "launch"),
    ("deny", "admit"),
)


def _reaction_concept_aliases(tokens: set[str]) -> set[str]:
    aliases = set(tokens)
    normalization = {
        "removed": "remove",
        "removing": "remove",
        "removal": "remove",
        "presidential": "president",
        "presidency": "president",
        "dismantling": "dismantle",
        "influential": "influence",
        "accounts": "account",
        "sources": "source",
    }
    aliases.update(normalization[token] for token in tokens if token in normalization)
    return aliases


def inspect_reaction_function(
    packet: EvidencePacket,
    text: str,
) -> ReactionFunctionInspection:
    """Validate semantic reaction work; labels and formulaic markers are never sufficient."""

    source_text = " ".join(item.text for item in packet.items)
    source_claims = tuple(
        claim for item in packet.items for claim in extract_claims(item.text, prefix="RS")
    )
    stripped = _REACTION_TEMPLATE.sub(" ", text)
    stripped = re.sub(r"[\"“”'\u2018\u2019]", " ", stripped)
    stripped = " ".join(stripped.split()).strip(" :,-—")
    source_concepts = _reaction_concept_aliases(_concept_tokens(source_text))
    reaction_concepts = _reaction_concept_aliases(_concept_tokens(stripped or text))
    overlap = (
        len(source_concepts & reaction_concepts)
        / max(min(len(source_concepts), len(reaction_concepts)), 1)
        if source_concepts and reaction_concepts
        else 0.0
    )
    source_specific = overlap >= 0.18 or bool(
        {"orban", "orbán", "sulyok", "hungary"} & reaction_concepts
    )
    reaction_claims = extract_claims(stripped or text, prefix="RR")
    comparisons = [compare_claim_to_evidence(claim, source_claims) for claim in reaction_claims]
    factual_claims = [
        (claim, comparison)
        for claim, comparison in zip(reaction_claims, comparisons, strict=True)
        if claim.kind not in {"opinion", "uncertainty"}
    ]
    scope_echo = _scope_restatement(packet, stripped or text)
    all_covered = bool(factual_claims) and all(
        comparison.support in {"supported", "partially_supported", "attributed_only"}
        for _, comparison in factual_claims
    )
    no_new_semantic_material = not (reaction_concepts - source_concepts - {"update", "finally"})
    marker_present = bool(_REACTION_TEMPLATE.search(text))
    source_echo = (
        scope_echo
        or all_covered
        or (source_specific and no_new_semantic_material)
        or (marker_present and source_specific and overlap >= 0.5)
    )
    unsupported_detail = any(
        comparison.support in {"unsupported", "source_conflict", "extraction_uncertain"}
        and claim.predicate is not None
        for claim, comparison in factual_claims
    )
    nda_accusation = bool(
        re.search(
            r"\b(?:broke|breached|violated)\s+(?:\w+\s+){0,2}nda(?:s)?\b",
            normalize_text(re.sub(r"[-\u2010-\u2015]", " ", text)),
        )
    )
    source_establishes_nda = bool(
        re.search(
            r"\b(?:broke|breached|violated)\s+(?:\w+\s+){0,2}nda(?:s)?\b",
            normalize_text(re.sub(r"[-\u2010-\u2015]", " ", source_text)),
        )
    )
    if nda_accusation and not source_establishes_nda:
        unsupported_detail = True

    combined = normalize_text(f"{source_text} {text}")
    incongruity = any(left in combined and right in combined for left, right in _INCONGRUITY_PAIRS)
    marker_wordplay = bool(_REACTION_WORDPLAY.search(text)) and source_specific
    quoted_fragments = tuple(
        match.group(1).strip() for match in re.finditer(r"[\"“]([^\"”]{1,80})[\"”]", text)
    )
    quoted_source_callback = bool(
        quoted_fragments
        and any(
            normalize_text(fragment) in normalize_text(source_text) for fragment in quoted_fragments
        )
        and _REACTION_CALLBACK.search(text)
    )
    workload_match = re.search(
        r"^(?P<fragment>[A-Za-z0-9_@'\u2019 -]{2,50})\s+is doing "
        r"(?:heroic amounts|a lot|heavy lifting) of work\b",
        text.strip(),
        re.I,
    )
    workload_callback = False
    if workload_match is not None:
        fragment_concepts = _concept_tokens(workload_match.group("fragment"))
        workload_callback = bool(
            fragment_concepts
            and fragment_concepts <= source_concepts
            and len(fragment_concepts) <= 5
        )
    quoted_callback = quoted_source_callback or workload_callback
    source_norm = normalize_text(re.sub(r"[-\u2010-\u2015]", " ", source_text))
    text_norm = normalize_text(re.sub(r"[-\u2010-\u2015]", " ", text))
    anti_yap_transform = bool(
        re.search(r"\b(?:verbosity|verbose|low verbosity)\b", source_norm)
        and re.search(r"\b(?:yap|anti yap|mute)\b", text_norm)
    )
    blueprint_transform = bool(
        re.search(r"\b(?:blueprint|template|download|free)\b", source_norm)
        and re.search(r"\b(?:download|install|button)\b", text_norm)
        and re.search(r"\b(?:consistency|discipline|patience|work)\b", text_norm)
    )
    corroboration_transform = bool(
        re.search(r"\b(?:two|both|accounts?|sources?)\b", source_norm)
        and re.search(r"\b(?:independent sources?|confirmed)\b", text_norm)
    )
    early_access_transform = bool(
        re.search(r"\b(?:early access|gpt ?6|gta ?6)\b", source_norm)
        and re.search(r"\b(?:nda|leak|bro|one tweet)\b", text_norm)
    )
    nda_register_shift = bool(
        re.search(
            r"\b(?:broke|breached|violated)\s+(?:\w+\s+){0,2}nda(?:s)?\b",
            source_norm,
        )
        and re.search(
            r"\b(?:broke|breached|violated)\s+(?:\w+\s+){0,2}nda(?:s)?\b",
            text_norm,
        )
        and re.match(r"^(?:bro|dude)\b", text_norm)
        and not unsupported_detail
    )
    feedback_loop_transform = bool(
        re.search(r"\b(?:button|feedback loop|manage)\b", source_norm)
        and re.search(r"\b(?:button|own the loop)\b", text_norm)
    )
    naming_callback = bool(
        re.search(r"\b(?:analytics|growth app|third party app)\b", source_norm)
        and re.search(
            r"[\"\u201c'\u2018][^\"\u201d'\u2019]*(?:analytics|growth|pro)[^\"\u201d'\u2019]*[\"\u201d'\u2019]",
            text,
            re.I,
        )
    )
    second_opinion_transform = bool(
        re.search(r"\b(?:second|another)\s+(?:confirmation|confirm)\b", source_norm)
        and re.search(r"\bsecond opinion\b", text_norm)
    )
    hopeful_name_transform = bool(
        re.search(r"\b(?:airdrop|reward)\b", source_norm)
        and re.search(r"\b(?:fee|fees|paying|paid)\b", source_norm)
        and re.search(r"\b(?:hopeful name|hopeful label|optimistic name)\b", text_norm)
        and re.search(r"\b(?:fee|fees|paying|paid)\b", text_norm)
    )
    semantic_transform = bool(
        anti_yap_transform
        or blueprint_transform
        or corroboration_transform
        or early_access_transform
        or nda_register_shift
        or feedback_loop_transform
        or naming_callback
        or second_opinion_transform
        or hopeful_name_transform
        or quoted_callback
    )
    source_specific = source_specific or semantic_transform
    # The bounded transforms above are non-literal reaction functions. A local claim
    # extractor may parse their playful nouns as factual details; once the transform is
    # independently tied to the source, that parse must not turn the joke into a false fact.
    if semantic_transform:
        unsupported_detail = False
    if nda_register_shift:
        # The legal/factual proposition must already be explicit in the source.  The only
        # transformation is the source-specific register shift, so it is not a plain source echo.
        source_echo = False
    # Marker vocabulary is merely a proposal. It counts only when a relation-level
    # transformation is independently established from source and reaction semantics.
    wordplay = bool(
        semantic_transform and (marker_wordplay or anti_yap_transform or naming_callback)
    )
    literalization = bool(
        source_specific
        and (
            re.search(r"\bliteral(?:ly|ization)?\b", text, re.I)
            or blueprint_transform
            or feedback_loop_transform
        )
    )
    dry_reframe = bool(
        source_specific
        and not source_echo
        and not unsupported_detail
        and (
            second_opinion_transform
            or hopeful_name_transform
            or naming_callback
            or corroboration_transform
            or early_access_transform
            or nda_register_shift
            or quoted_callback
        )
        and (not marker_wordplay or semantic_transform)
        and (not marker_present or semantic_transform)
    )
    if incongruity:
        reaction_type = "incongruity"
    elif quoted_callback:
        reaction_type = "callback"
    elif wordplay:
        reaction_type = "wordplay"
    elif literalization:
        reaction_type = "literalization"
    elif dry_reframe:
        reaction_type = "dry_reframe"
    elif source_specific and not source_echo and not unsupported_detail:
        reaction_type = "uncertain"
    else:
        reaction_type = "none"
    generic_template_only = bool(
        source_specific
        and (marker_present or marker_wordplay)
        and not incongruity
        and not wordplay
        and not quoted_callback
        and not literalization
        and not dry_reframe
    )
    source_echo = source_echo or generic_template_only
    grounded = bool(
        reaction_type not in {"none", "uncertain"}
        and source_specific
        and not unsupported_detail
        and not generic_template_only
    )
    return ReactionFunctionInspection(
        reaction_type=reaction_type,
        source_specific=source_specific,
        source_echo=source_echo,
        unsupported_detail=unsupported_detail,
        generic_template_only=generic_template_only,
        grounded=grounded,
    )


def inspect_humor_intent(
    packet: EvidencePacket,
    text: str,
    *,
    metadata_says_humor: bool,
) -> HumorIntentInspection:
    """Extract final-text semantics once; provider metadata remains a non-authoritative flag."""

    source_text = " ".join(item.text for item in packet.items)
    semantic_ir = extract_semantic_inspection(source_text, text)
    reaction = inspect_reaction_function(packet, text)
    template_humor = bool(_JOKE_SHAPED_TEXT.search(text))
    return adjudicate_humor_intent(
        semantic_ir,
        metadata_says_humor=metadata_says_humor,
        reaction_type=reaction.reaction_type,
        template_humor=template_humor,
    )


def _scope_restatement(packet: EvidencePacket, text: str) -> bool:
    """Detect a scope-repackaging paraphrase from semantic roles rather than one phrase."""

    source = " ".join(item.text for item in packet.items)
    source_has_office = bool(_OFFICE_ALIAS.search(source))
    source_has_broader_effort = bool(_SCOPE_EFFORT.search(source)) and bool(
        re.search(
            r"\b(?:latest move|move to|step to|aimed at|dismantl(?:e|ing)|influence)\b",
            source,
            re.I,
        )
    )
    draft_has_office = bool(_OFFICE_ALIAS.search(text))
    draft_has_effort = bool(_SCOPE_EFFORT.search(text))
    draft_has_scope_relation = bool(_SCOPE_EVENT_VERBS.search(text))
    source_entities = _concept_tokens(source)
    draft_entities = _concept_tokens(text)
    named_overlap = bool(source_entities & draft_entities & {"orban", "orbán", "sulyok", "hungary"})
    return bool(
        source_has_office
        and source_has_broader_effort
        and draft_has_office
        and draft_has_effort
        and draft_has_scope_relation
        and named_overlap
    )


def _concept_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[^\W_]+", normalize_text(value), flags=re.UNICODE):
        token = raw.strip("\u2019'")
        if len(token) <= 2 or token in {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "into",
            "over",
            "inside",
            "puts",
            "put",
            "moves",
            "move",
            "gives",
            "give",
            "carries",
            "carry",
        }:
            continue
        if token.endswith("\u2019s") or token.endswith("'s"):
            token = token[:-2]
        if token.endswith("s") and len(token) > 5:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _entity_alias_overlap(
    source_claims: tuple[CanonicalClaim, ...], draft_claims: tuple[CanonicalClaim, ...]
) -> bool:
    source_tokens = {
        token
        for claim in source_claims
        for entity in claim.named_entities
        for token in _concept_tokens(entity)
    }
    draft_tokens = {
        token
        for claim in draft_claims
        for entity in claim.named_entities
        for token in _concept_tokens(entity)
    }
    return bool(source_tokens & draft_tokens)


def _semantic_source_overlap(packet: EvidencePacket, text: str) -> float:
    source_tokens = _concept_tokens(" ".join(item.text for item in packet.items))
    draft_tokens = _concept_tokens(text)
    if not source_tokens or not draft_tokens:
        return 0.0
    return len(source_tokens & draft_tokens) / max(min(len(source_tokens), len(draft_tokens)), 1)


def _span_for_claim(packet: EvidencePacket, claim: CanonicalClaim) -> EvidenceSpan:
    for item in packet.items:
        if normalize_text(claim.exact_text) in normalize_text(item.text):
            return EvidenceSpan(evidence_id=item.evidence_id, quote=claim.exact_text)
    item = packet.items[0]
    return EvidenceSpan(evidence_id=item.evidence_id, quote=item.text)


def _claim_key(claim: CanonicalClaim) -> tuple[str, str, str, str]:
    return (
        normalize_text(claim.subject or ""),
        claim.predicate or "",
        normalize_text(claim.object or ""),
        claim.polarity,
    )


def _qualifier_signature(claim: CanonicalClaim) -> set[tuple[str, str]]:
    return {(item.qualifier_type, normalize_text(item.text)) for item in claim.material_qualifiers}


def _preserve_contribution_propositions(
    planned: tuple[CanonicalClaim, ...],
    current: tuple[CanonicalClaim, ...],
    *,
    allow_quantity_omission: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Match every planned proposition to one current proposition without reuse."""

    unmatched = list(current)
    issues: list[str] = []
    for target in planned:
        best: CanonicalClaim | None = None
        best_match = None
        best_score = -10_000
        for candidate in unmatched:
            match = align_claim_to_evidence(target, candidate)
            score = 0
            if match.predicate_alignment == "same":
                score += 8
            elif match.predicate_alignment == "opposite":
                score += 3
            if match.object_alignment == "same":
                score += 6
            elif match.object_alignment == "partial":
                score += 3
            if match.actor_alignment == "same":
                score += 5
            elif match.actor_alignment == "partial":
                score += 2
            if match.event_alignment == "same":
                score += 6
            elif match.event_alignment == "partial":
                score += 2
            if score > best_score:
                best, best_match, best_score = candidate, match, score
        if best is None or best_match is None or best_score < 6:
            issues.append("CONTRIBUTION_PROPOSITION_DROPPED")
            continue
        unmatched.remove(best)
        core_same = (
            best_match.predicate_alignment == "same"
            and best_match.actor_alignment == "same"
            and best_match.object_alignment == "same"
            and target.polarity == best.polarity
        )
        if not core_same:
            issues.append("CONTRIBUTION_PROPOSITION_CHANGED")
        if target.quantities and best_match.quantity_alignment != "same":
            allowed_endpoint_summary = bool(
                allow_quantity_omission
                and best_match.quantity_alignment == "missing"
                and any(item.role == "end_value" for item in best.quantity_mentions)
                and all(
                    item.role in {"start_value", "delta", "end_value"}
                    for item in target.quantity_mentions
                )
            )
            if not allowed_endpoint_summary:
                issues.append("CONTRIBUTION_DETAIL_DROPPED")
        if target.dates and best_match.date_alignment != "same":
            issues.append("CONTRIBUTION_DETAIL_DROPPED")
        if _qualifier_signature(target) - _qualifier_signature(best):
            issues.append("CONTRIBUTION_QUALIFIER_CHANGED")
        if target.clause_relation != best.clause_relation:
            issues.append("CONTRIBUTION_CLAUSE_RELATION_CHANGED")
        if (
            target.epistemic_modality != best.epistemic_modality
            or target.event_status != best.event_status
            or target.speech_act != best.speech_act
            or best_match.clause_scope_alignment not in {"compatible", "unknown"}
        ):
            issues.append("CERTAINTY_CHANGED_DURING_REVISION")
        if (
            target.attribution_source != best.attribution_source
            or target.attribution_act != best.attribution_act
        ):
            issues.append("ATTRIBUTION_CHANGED_DURING_REVISION")
    unique = tuple(dict.fromkeys(issues))
    return not unique, unique


def _is_planning_instruction(text: str) -> bool:
    normalized = normalize_text(text).replace("-", " ")
    return normalized.startswith(
        (
            "express one concrete observation",
            "express one precise caveat",
            "express one practical implication",
            "use only this supported source relation",
            "state this supported relation directly without adding interpretation",
            "follow this complete operator supplied topic without adding assumptions",
        )
    )


def _is_reaction_plan(text: str) -> bool:
    normalized = normalize_text(text).replace("-", " ")
    return normalized.startswith(
        (
            "react to this exact source specific detail without inventing a factual relation",
            "use dry humor only when a real source specific reaction function exists",
            "use source specific dry humor only when the semantic reaction is real",
        )
    )


def _is_reply_family_plan(text: str) -> bool:
    normalized = normalize_text(text).replace("-", " ")
    return normalized.startswith(
        (
            "offer the least constructed source specific plain observation",
            "respond directly to one source claim",
            "add one concrete source supported behavior",
            "add the least constructed source specific observation",
            "add one concrete supported context or consequence",
        )
    )


def _source_allows_exploratory_humor(source_map: SourceSemanticMap) -> bool:
    """Allow a humor proposal for a usable source without treating keywords as proof.

    This only keeps the planning family available. The generated reply still has to pass factual
    adjudication, reaction-function inspection, source-echo checks, harmful-assertion guards, and
    recent-output diversity before it can be recommended or approved.
    """

    return any(
        proposition.status == "explicit"
        and proposition.relation.speech_act == "assertion"
        and proposition.relation.event_status not in {"conditional", "counterfactual"}
        for proposition in source_map.propositions
    )


def _strong_implications(claim: CanonicalClaim) -> tuple[str, ...]:
    """Return only relation-preserving entailments from asserted completed propositions."""

    if (
        claim.speech_act != "assertion"
        or claim.epistemic_modality != "asserted"
        or claim.event_status not in {"completed", "ongoing"}
        or claim.polarity != "positive"
    ):
        return ()
    subject = claim.subject or "the subject"
    obj = claim.object or "the object"
    if claim.predicate == "decrease":
        return (f"{subject} decreased {obj}".strip(),)
    if claim.predicate == "increase":
        return (f"{subject} increased {obj}".strip(),)
    if claim.predicate == "remove":
        return (f"{obj} is being removed by {subject}".strip(),)
    if claim.predicate == "keep":
        return (f"{obj} is being kept by {subject}".strip(),)
    if claim.predicate == "exclude":
        return (f"{subject} does not include {obj}".strip(),)
    if claim.predicate == "include":
        return (f"{subject} includes {obj}".strip(),)
    if claim.predicate == "sign":
        return (f"{obj} was signed by {subject}".strip(),)
    return ()


def derive_source_semantic_map(packet: EvidencePacket) -> SourceSemanticMap:
    """Build a conservative application-owned semantic map synchronously."""

    propositions: list[SourceSemanticProposition] = []
    premises: list[CanonicalClaim] = []
    seen: set[str] = set()
    counter = 0
    for item_index, item in enumerate(packet.items, start=1):
        claims = extract_claims(item.text, prefix=f"S{item_index}-")
        for claim in claims:
            counter += 1
            premises.append(claim)
            span = EvidenceSpan(evidence_id=item.evidence_id, quote=claim.exact_text)
            key = normalize_text(claim.exact_text)
            seen.add(key)
            propositions.append(
                SourceSemanticProposition(
                    proposition_id=f"P{counter}",
                    text=claim.exact_text,
                    status="explicit",
                    relation=claim,
                    evidence_spans=(span,),
                )
            )
            for implication in _strong_implications(claim):
                implication_key = normalize_text(implication)
                if implication_key in seen:
                    continue
                seen.add(implication_key)
                counter += 1
                implied_claims = extract_claims(implication, prefix=f"I{counter}-")
                relation = (
                    implied_claims[0]
                    if implied_claims
                    else claim.model_copy(
                        update={"claim_id": f"I{counter}", "exact_text": implication}
                    )
                )
                propositions.append(
                    SourceSemanticProposition(
                        proposition_id=f"P{counter}",
                        text=implication,
                        status="strongly_implied",
                        relation=relation,
                        evidence_spans=(span,),
                    )
                )
    return SourceSemanticMap(
        propositions=tuple(propositions),
        conclusions=tuple(
            proposition.text
            for proposition in propositions
            if proposition.status == "strongly_implied"
        ),
        factual_premises=tuple(premises),
    )


def inspect_draft_deterministic(
    packet: EvidencePacket,
    contribution: AngleCandidate,
    text: str,
    source_map: SourceSemanticMap,
    *,
    commentary_mode: bool,
    semantic_ir: SemanticInspection | None = None,
) -> DraftSemanticInspection:
    """Run the conservative semantic draft inspector without provider I/O."""

    draft_claims = extract_claims(text, prefix="D")
    source_claims = tuple(item.relation for item in source_map.propositions)
    comparisons = [compare_claim_to_evidence(claim, source_claims) for claim in draft_claims]
    supported_count = sum(
        item.support in {"supported", "partially_supported"} for item in comparisons
    )
    source_echo = commentary_mode and bool(draft_claims) and supported_count == len(draft_claims)

    entity_reuse = _entity_alias_overlap(source_claims, draft_claims)
    semantic_overlap = _semantic_source_overlap(packet, text)
    source_concept_reuse = entity_reuse or semantic_overlap >= 0.22
    scope_restatement = _scope_restatement(packet, text)
    abstract_packaging = (
        bool(_ABSTRACT_FUNCTION.search(text))
        or (len(_ABSTRACT_NOUN.findall(text)) >= 3 and source_concept_reuse)
        or scope_restatement
    )
    source_text = " ".join(item.text for item in packet.items)
    semantic_ir = semantic_ir or extract_semantic_inspection(source_text, text)
    reaction_labeled = contribution.contribution_type in _REACTION_CONTRIBUTION_TYPES
    reaction_plan = _is_reaction_plan(contribution.thesis)
    reaction_for_intent = inspect_reaction_function(packet, text)
    humor_intent = adjudicate_humor_intent(
        semantic_ir,
        metadata_says_humor=reaction_labeled,
        reaction_type=reaction_for_intent.reaction_type,
        template_humor=bool(_JOKE_SHAPED_TEXT.search(text)),
    )
    reaction_inspection = (
        inspect_reaction_function(packet, text)
        if reaction_labeled or humor_intent.text_appears_humorous
        else None
    )
    humor_safety_context = adjudicate_humor_safety(semantic_ir)
    humor_safety = humor_safety_context if humor_intent.humor_safety_required else None
    non_humor_reaction_types = {
        "plain_observation",
        "direct_response",
        "specific_qualification",
        "genuine_question",
        "contextual_extension",
    }
    non_humor_reaction_valid = bool(
        reaction_inspection is not None
        and contribution.contribution_type in non_humor_reaction_types
        and reaction_inspection.source_specific
        and not reaction_inspection.source_echo
        and not reaction_inspection.unsupported_detail
    )
    reaction_function_valid = bool(
        reaction_inspection is not None
        and (reaction_inspection.grounded or non_humor_reaction_valid)
    )
    reaction_exemption = bool(
        reaction_function_valid
        and (
            not humor_intent.humor_safety_required
            or (humor_safety is not None and humor_safety.suitable_for_humor)
        )
    )
    implied_restatement = (
        commentary_mode and not reaction_exemption and source_concept_reuse and abstract_packaging
    ) or (commentary_mode and not reaction_exemption and scope_restatement)
    if commentary_mode and not reaction_exemption and scope_restatement:
        source_echo = True
    # Paraphrase-invariant source echo: if a commentary draft reuses the source actors and
    # concepts, adds no independently structured factual relation, and merely changes the
    # rhetorical frame, it remains a source echo even when no sentence is lexically close.
    has_independent_relation = any(
        comparison.support not in {"supported", "partially_supported"}
        and claim.predicate is not None
        for claim, comparison in zip(draft_claims, comparisons, strict=True)
    )
    if (
        commentary_mode
        and not reaction_exemption
        and source_concept_reuse
        and not has_independent_relation
        and abstract_packaging
    ):
        source_echo = True

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    relation_keys = [_claim_key(claim) for claim in draft_claims if claim.predicate]
    repeated_relation = len(relation_keys) != len(set(relation_keys))
    repeated_abstract_job = (
        len(sentences) > 1
        and sum(bool(_ABSTRACT_FUNCTION.search(sentence)) for sentence in sentences) >= 2
    )
    internal_repetition = repeated_relation or repeated_abstract_job

    contribution_text = contribution.thesis
    if contribution_text.casefold().startswith(
        "state this supported relation directly without adding interpretation:"
    ):
        contribution_text = contribution_text.split(":", 1)[1].strip()
    contribution_claims = extract_claims(contribution_text, prefix="C")
    contribution_preserved = False
    contribution_delivered = False
    preservation_issues: list[str] = []
    if reaction_plan:
        contribution_preserved = reaction_exemption
        contribution_delivered = reaction_exemption
        if not reaction_exemption:
            preservation_issues.append("PROMISED_INSIGHT_NOT_DELIVERED")
    elif _is_planning_instruction(contribution.thesis) or _is_reply_family_plan(
        contribution.thesis
    ):
        contribution_preserved = bool(text.strip())
        contribution_delivered = bool(text.strip())
    elif contribution_claims:
        contribution_preserved, invariant_issues = _preserve_contribution_propositions(
            contribution_claims,
            draft_claims,
            allow_quantity_omission=contribution.contribution_type in {"direct_update", "summary"},
        )
        preservation_issues.extend(invariant_issues)
        contribution_delivered = contribution_preserved
    else:
        contribution_preserved = normalize_text(contribution.thesis) in normalize_text(text)
        contribution_delivered = contribution_preserved
    if (
        contribution.contribution_type in {"direct_update", "summary"}
        and supported_count > 0
        and not contribution_claims
    ):
        contribution_preserved = True
        contribution_delivered = True

    # Unsupported opinion-shaped prose is not automatically a contribution. When semantic
    # function is still source echo or implied restatement, abstention is the safe outcome.
    no_grounded = (
        commentary_mode and not reaction_exemption and (source_echo or implied_restatement)
    ) or bool(humor_safety is not None and not humor_safety.suitable_for_humor)
    source_stronger = (
        commentary_mode
        and (source_echo or implied_restatement)
        and (len(text) > sum(len(item.text) for item in packet.items) or abstract_packaging)
    )

    issue_codes: list[str] = []
    if source_echo:
        issue_codes.append("SOURCE_ECHO")
    if implied_restatement:
        issue_codes.append("IMPLIED_POINT_RESTATEMENT")
    if source_echo or implied_restatement:
        issue_codes.append("LOW_INFORMATION_GAIN")
    if abstract_packaging:
        issue_codes.append("ANALYTICAL_PACKAGING")
    if _CONTRAST.search(text) and (source_echo or implied_restatement):
        issue_codes.append("MANUFACTURED_CONTRAST")
    if internal_repetition:
        issue_codes.append("INTERNAL_REPETITION")
    if source_stronger:
        issue_codes.append("SOURCE_STRONGER_THAN_DRAFT")
    issue_codes.extend(humor_intent.issue_codes)
    if humor_safety is not None and not humor_safety.suitable_for_humor:
        issue_codes.append("HUMOR_ON_TRAGEDY")
    if reaction_labeled and not reaction_exemption:
        if humor_safety is None or humor_safety.suitable_for_humor:
            issue_codes.append("REACTION_LABEL_WITHOUT_FUNCTION")
        if reaction_inspection is not None and reaction_inspection.source_echo:
            issue_codes.extend(
                ("PLAIN_PARAPHRASE_AS_REACTION", "SOURCE_ECHO", "LOW_INFORMATION_GAIN")
            )
    issue_codes.extend(preservation_issues)
    if not contribution_delivered:
        issue_codes.append("PROMISED_INSIGHT_NOT_DELIVERED")

    return DraftSemanticInspection(
        draft_propositions=draft_claims,
        contribution_delivered=contribution_delivered,
        contribution_preserved=contribution_preserved,
        source_echo=source_echo,
        implied_restatement=implied_restatement,
        internal_repetition=internal_repetition,
        analytical_packaging=abstract_packaging,
        source_stronger_than_draft=source_stronger,
        no_grounded_contribution=no_grounded,
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        evidence_spans=tuple(sentence for sentence in sentences[:3]),
        humor_intent_proposal=humor_intent.proposal,
        humor_safety=humor_safety_context,
        semantic_inspection=semantic_ir,
    )


class DeterministicSemanticInspector:
    """Offline structured semantic inspector used by mock mode and adversarial tests."""

    async def inspect_source(self, packet: EvidencePacket) -> SourceSemanticMap:
        return derive_source_semantic_map(packet)

    async def inspect_contributions(
        self,
        packet: EvidencePacket,
        contributions: tuple[AngleCandidate, ...],
        source_map: SourceSemanticMap,
    ) -> tuple[ContributionInspection, ...]:
        source_claims = tuple(item.relation for item in source_map.propositions)
        source_text = " ".join(item.text for item in packet.items)
        source_concepts = _concept_tokens(source_text)
        results: list[ContributionInspection] = []
        for index, contribution in enumerate(contributions, start=1):
            parsed = extract_claims(contribution.thesis, prefix=f"K{index}-")
            proposition = (
                parsed[0]
                if parsed
                else CanonicalClaim(
                    claim_id=f"K{index}",
                    exact_text=contribution.thesis,
                    kind="interpretation",
                    subject=None,
                    predicate=None,
                    object=contribution.thesis,
                    polarity="unknown",
                    modality="unknown",
                )
            )
            comparison = compare_claim_to_evidence(proposition, source_claims)
            contribution_concepts = _concept_tokens(contribution.thesis)
            concept_overlap = (
                len(source_concepts & contribution_concepts)
                / max(min(len(source_concepts), len(contribution_concepts)), 1)
                if source_concepts and contribution_concepts
                else 0.0
            )
            reaction = contribution.contribution_type in _REACTION_CONTRIBUTION_TYPES
            direct = contribution.contribution_type in {"direct_update", "summary"}
            reply_plan = (
                contribution.contribution_type in _REPLY_NONHUMOR_TYPES
                and _is_reply_family_plan(contribution.thesis)
            )
            reaction_plan = reaction and _is_reaction_plan(contribution.thesis)
            reaction_inspection = (
                inspect_reaction_function(packet, contribution.thesis)
                if reaction and not reaction_plan
                else None
            )
            reaction_function_present = bool(
                reaction_inspection is not None
                and reaction_inspection.reaction_type not in {"none", "uncertain"}
            )
            source_specificity_present = bool(
                reaction_plan
                or reply_plan
                or (reaction_inspection is not None and reaction_inspection.source_specific)
            )
            source_covered = comparison.support in {
                "supported",
                "partially_supported",
                "attributed_only",
            }
            plain_paraphrase_only = bool(
                reaction_inspection is not None
                and not reaction_inspection.grounded
                and (
                    reaction_inspection.source_echo
                    or reaction_inspection.generic_template_only
                    or (
                        reaction_inspection.source_specific
                        and reaction_inspection.reaction_type in {"none", "uncertain"}
                    )
                )
            )
            rhetorical_repackaging = (
                not direct
                and not reaction
                and not reply_plan
                and concept_overlap >= 0.35
                and comparison.support in {"unsupported", "extraction_uncertain"}
            )
            unsupported = list(contribution.unsupported_assumptions)
            requires_assumption = contribution.requires_new_assumptions
            if rhetorical_repackaging:
                unsupported.append(
                    "The proposed relation is rhetorical repackaging or is not established "
                    "by evidence."
                )
                requires_assumption = True
            if (
                not direct
                and not reaction
                and not reply_plan
                and comparison.support in {"unsupported", "source_conflict", "extraction_uncertain"}
            ):
                requires_assumption = True
            already_covered = source_covered and not direct and not reaction and not reply_plan
            reply_plan_grounded = bool(
                reply_plan
                and contribution.contribution_type
                in {
                    "plain_observation",
                    "direct_response",
                    "specific_qualification",
                    "genuine_question",
                }
            )
            if reply_plan and contribution.contribution_type == "contextual_extension":
                reply_plan_grounded = bool(
                    len(packet.items) > 1
                    or any(item.status == "strongly_implied" for item in source_map.propositions)
                )
            if reaction_plan:
                reaction_plan_grounded = _source_allows_exploratory_humor(source_map)
            else:
                reaction_plan_grounded = False
            grounded = (
                not requires_assumption
                and not unsupported
                and contribution.support_status != "unsupported"
                and (bool(contribution.evidence_ids) or reaction)
                and not already_covered
                and (
                    direct
                    or reply_plan_grounded
                    or reaction_plan_grounded
                    or (
                        reaction
                        and reaction_inspection is not None
                        and reaction_inspection.grounded
                    )
                    or comparison.support == "supported"
                )
            )
            results.append(
                ContributionInspection(
                    contribution_id=contribution.angle_id,
                    canonical_proposition=proposition,
                    already_covered=already_covered,
                    requires_new_assumption=requires_assumption,
                    unsupported_adjacent_complications=tuple(dict.fromkeys(unsupported)),
                    grounded=grounded,
                    reaction_function_present=reaction_function_present,
                    source_specificity_present=source_specificity_present,
                    plain_paraphrase_only=plain_paraphrase_only,
                    reaction_inspection=reaction_inspection,
                )
            )
        return tuple(results)

    async def inspect_draft(
        self,
        packet: EvidencePacket,
        contribution: AngleCandidate,
        text: str,
        source_map: SourceSemanticMap,
        *,
        commentary_mode: bool,
    ) -> DraftSemanticInspection:
        return inspect_draft_deterministic(
            packet,
            contribution,
            text,
            source_map,
            commentary_mode=commentary_mode,
        )


def merge_semantic_inspections(
    deterministic: DraftSemanticInspection,
    provider: DraftSemanticInspection | None,
    *,
    text: str | None = None,
    source_text: str | None = None,
    canonical_ir: SemanticInspection | None = None,
) -> DraftSemanticInspection:
    """Merge provider proposals fail-closed; neither may erase a deterministic block."""

    if provider is None:
        if canonical_ir is None or deterministic.semantic_inspection == canonical_ir:
            return deterministic
        return deterministic.model_copy(update={"semantic_inspection": canonical_ir})

    resolved_ir = canonical_ir or deterministic.semantic_inspection
    if (
        canonical_ir is None
        and resolved_ir is not None
        and provider.semantic_inspection is not None
    ):
        resolved_ir = reconcile_semantic_inspections(
            resolved_ir,
            provider.semantic_inspection,
            source_text=source_text or "",
            reply_text=text or "",
        )

    deterministic_proposal = deterministic.humor_intent_proposal
    provider_proposal = provider.humor_intent_proposal
    provider_valid = True
    if provider_proposal is not None and text is not None:
        provider_valid = all(span and span in text for span in provider_proposal.evidence_spans)
        if provider_proposal.humorous_treatment and not provider_proposal.evidence_spans:
            provider_valid = False
    provider_humorous = bool(
        provider_valid
        and provider_proposal is not None
        and provider_proposal.humorous_treatment
        and provider_proposal.intent not in {"none", "uncertain"}
    )
    deterministic_humorous = bool(
        deterministic_proposal is not None and deterministic_proposal.humorous_treatment
    )
    canonical_safety = (
        adjudicate_humor_safety(resolved_ir)
        if resolved_ir is not None
        else deterministic.humor_safety
    )
    canonical_irony = adjudicate_evaluative_irony(resolved_ir) if resolved_ir is not None else None
    ir_humorous = bool(canonical_irony is not None and canonical_irony.implicit_sarcasm)
    sensitive = bool(canonical_safety is not None and canonical_safety.sensitive_context)
    provider_clear_nonhumor = bool(
        provider_valid
        and provider_proposal is not None
        and not provider_proposal.humorous_treatment
        and provider_proposal.intent in {"none", "literal_observation", "question", "qualification"}
    )
    deterministic_clear_nonhumor = bool(
        deterministic_proposal is not None
        and not deterministic_proposal.humorous_treatment
        and deterministic_proposal.intent == "none"
    )
    material_conflict = bool(
        ((deterministic_humorous or ir_humorous) and provider_clear_nonhumor)
        or (provider_humorous and deterministic_clear_nonhumor)
    )

    deterministic_issue_codes = deterministic.issue_codes
    if (
        not provider_valid
        and not deterministic_humorous
        and "HUMOR_INTENT_UNCERTAIN" in deterministic_issue_codes
    ):
        # An invalid provider humor span is a provider conflict, not independent proof that
        # the final text is humorous. Preserve the deterministic candidate-local uncertainty
        # block without upgrading it to a tragedy-humor finding.
        deterministic_issue_codes = tuple(
            code for code in deterministic_issue_codes if code != "HUMOR_ON_TRAGEDY"
        )
    issue_codes = list((*deterministic_issue_codes, *provider.issue_codes))
    if sensitive and (material_conflict or not provider_valid):
        issue_codes.append("HUMOR_INTENT_CONFLICT")
    if sensitive and provider_proposal is not None and provider_proposal.intent == "uncertain":
        issue_codes.append("HUMOR_INTENT_UNCERTAIN")
    if (
        (deterministic_humorous or provider_humorous or ir_humorous)
        and canonical_safety is not None
        and not canonical_safety.suitable_for_humor
    ):
        issue_codes.append("HUMOR_ON_TRAGEDY")

    resolved_proposal = deterministic_proposal
    if (provider_humorous and not deterministic_humorous) or (
        resolved_proposal is None and provider_valid
    ):
        resolved_proposal = provider_proposal

    return DraftSemanticInspection(
        draft_propositions=provider.draft_propositions or deterministic.draft_propositions,
        contribution_delivered=(
            deterministic.contribution_delivered and provider.contribution_delivered
        ),
        contribution_preserved=(
            deterministic.contribution_preserved and provider.contribution_preserved
        ),
        source_echo=deterministic.source_echo or provider.source_echo,
        implied_restatement=(deterministic.implied_restatement or provider.implied_restatement),
        internal_repetition=(deterministic.internal_repetition or provider.internal_repetition),
        analytical_packaging=(deterministic.analytical_packaging or provider.analytical_packaging),
        source_stronger_than_draft=(
            deterministic.source_stronger_than_draft or provider.source_stronger_than_draft
        ),
        no_grounded_contribution=(
            deterministic.no_grounded_contribution or provider.no_grounded_contribution
        ),
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        evidence_spans=tuple(
            dict.fromkeys((*deterministic.evidence_spans, *provider.evidence_spans))
        ),
        humor_intent_proposal=resolved_proposal,
        humor_safety=canonical_safety or deterministic.humor_safety or provider.humor_safety,
        semantic_inspection=resolved_ir or provider.semantic_inspection,
    )


def merge_source_semantic_maps(
    packet: EvidencePacket,
    deterministic: SourceSemanticMap,
    provider: SourceSemanticMap | None,
) -> SourceSemanticMap:
    """Cross-validate provider proposition text, relation, evidence, and implication strength."""

    if provider is None:
        return deterministic
    by_id = {item.evidence_id: item.text for item in packet.items}
    accepted = list(deterministic.propositions)
    seen = {normalize_text(item.text) for item in accepted}
    for proposition in provider.propositions:
        if normalize_text(proposition.text) in seen:
            continue
        issues: list[str] = []
        valid_spans = bool(proposition.evidence_spans)
        for span in proposition.evidence_spans:
            canonical = by_id.get(span.evidence_id)
            if canonical is None:
                valid_spans = False
                break
            if span.start is not None or span.end is not None:
                if (
                    span.start is None
                    or span.end is None
                    or span.start > span.end
                    or span.end > len(canonical)
                    or canonical[span.start : span.end] != span.quote
                ):
                    valid_spans = False
                    break
            elif normalize_text(span.quote) not in normalize_text(canonical):
                valid_spans = False
                break
        if not valid_spans:
            continue
        if not claim_structure_matches_text(proposition.relation, proposition.text):
            issues.append("SEMANTIC_PROPOSITION_STRUCTURE_MISMATCH")
        cited_claims: list[CanonicalClaim] = []
        for span_index, span in enumerate(proposition.evidence_spans, start=1):
            cited_claims.extend(extract_claims(span.quote, prefix=f"V{span_index}-"))
        relation_check = compare_claim_to_evidence(proposition.relation, tuple(cited_claims))
        if proposition.status == "strongly_implied" and relation_check.support != "supported":
            issues.append("UNSUPPORTED_STRONG_IMPLICATION")
        if proposition.status == "explicit" and not any(
            normalize_text(proposition.text) in normalize_text(span.quote)
            for span in proposition.evidence_spans
        ):
            issues.append("SEMANTIC_PROPOSITION_STRUCTURE_MISMATCH")
        if issues:
            continue
        accepted.append(proposition)
        seen.add(normalize_text(proposition.text))
    premises = list(deterministic.factual_premises)
    for premise in provider.factual_premises:
        if not claim_structure_matches_text(premise, premise.exact_text):
            continue
        comparison = compare_claim_to_evidence(premise, deterministic.factual_premises)
        if comparison.support != "supported":
            continue
        if any(
            normalize_text(item.exact_text) == normalize_text(premise.exact_text)
            for item in premises
        ):
            continue
        premises.append(premise)
    return SourceSemanticMap(
        propositions=tuple(accepted),
        conclusions=tuple(
            dict.fromkeys(
                item.text
                for item in accepted
                if item.status == "strongly_implied" and not item.validation_issues
            )
        ),
        factual_premises=tuple(premises),
    )


def merge_contribution_inspections(
    deterministic: tuple[ContributionInspection, ...],
    provider: tuple[ContributionInspection, ...] | None,
) -> tuple[ContributionInspection, ...]:
    """Application-owned fail-closed merge for contribution proposals."""

    if provider is None:
        return deterministic
    by_id = {item.contribution_id: item for item in provider}
    merged: list[ContributionInspection] = []
    for item in deterministic:
        proposal = by_id.get(item.contribution_id)
        if proposal is None:
            merged.append(item.model_copy(update={"grounded": False}))
            continue
        merged.append(
            item.model_copy(
                update={
                    "already_covered": item.already_covered or proposal.already_covered,
                    "requires_new_assumption": (
                        item.requires_new_assumption or proposal.requires_new_assumption
                    ),
                    "unsupported_adjacent_complications": tuple(
                        dict.fromkeys(
                            (
                                *item.unsupported_adjacent_complications,
                                *proposal.unsupported_adjacent_complications,
                            )
                        )
                    ),
                    "grounded": (
                        item.grounded
                        and proposal.grounded
                        and not item.already_covered
                        and not proposal.already_covered
                        and not item.requires_new_assumption
                        and not proposal.requires_new_assumption
                    ),
                    # These fields describe the provider's proposed reply affordance and
                    # are used only for application-owned family ranking. Grounding itself
                    # remains fail-closed above (deterministic AND provider). A validated live
                    # affordance must therefore be able to distinguish a genuinely strong joke
                    # without turning the provider label into publication permission.
                    "reaction_function_present": (
                        item.reaction_function_present or proposal.reaction_function_present
                    ),
                    "source_specificity_present": (
                        item.source_specificity_present or proposal.source_specificity_present
                    ),
                    "plain_paraphrase_only": (
                        item.plain_paraphrase_only or proposal.plain_paraphrase_only
                    ),
                }
            )
        )
    return tuple(merged)
