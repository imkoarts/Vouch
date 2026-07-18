"""Evidence, medium-fit, anti-slop, and corpus-distinctiveness quality gates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from itertools import pairwise

from app.schemas.content import (
    EditorialIntent,
    EvidencePacket,
    GenerationRequest,
    GenerationResult,
    GenerationVariant,
    QualityIssue,
    QualityReport,
    ReplyNaturalnessInspection,
    ReplyShellInspection,
    SourceCoverage,
)
from app.services.claim_structure import compare_claim_sets, extract_claims
from app.services.semantic_adjudication import adjudicate_reply_shell
from app.services.semantic_classifier import (
    inspect_humor_intent,
    inspect_humor_safety,
    inspect_reaction_function,
)
from app.services.semantic_extraction import extract_semantic_inspection
from app.services.semantic_review import review_semantics, review_structure

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NON_DRAFT_SENTINEL = re.compile(
    r"^\s*(?:(?:there (?:is|isn['\u2019]t) )?insufficient[ _-]?context"
    r"(?:\s+(?:to|for)\b[^.!?]*)?|not enough (?:context|information)(?:\s+[^.!?]*)?|"
    r"no (?:supported )?(?:contribution|post|reply|thesis)(?:\s+[^.!?]*)?|"
    r"(?:i |we |the model )?(?:cannot|can['\u2019]t|am unable to|are unable to|is unable to) "
    r"(?:determine|write|produce|generate)\b[^.!?]*)[.!?\s]*$",
    re.IGNORECASE,
)

QUALITY_ISSUE_CODES: frozenset[str] = frozenset(
    {
        "ABSTRACTION_LADDER",
        "ANNOUNCEMENT_BEFORE_CONTENT",
        "IMPORTANCE_ANNOUNCEMENT",
        "INSIGHT_PREVIEW",
        "FALSE_AGENCY",
        "HIDDEN_ACTOR",
        "ACTORLESS_CAUSALITY",
        "NARRATOR_DISTANCE",
        "ARMCHAIR_GENERALIZATION",
        "DISEMBODIED_EXPLANATION",
        "UNASKED_REASSURANCE",
        "PERMISSION_GRANTING_ENDING",
        "EMOTIONAL_HAND_HOLDING",
        "SENTENCE_WITHOUT_JOB",
        "CUTTABILITY_FAILURE",
        "REDUNDANT_MODIFIER",
        "EXPLANATION_AFTER_PUNCHLINE",
        "READER_HAND_HOLDING",
        "METRONOMIC_RHYTHM",
        "FORMULAIC_PARAGRAPH_ENDING",
        "MECHANICAL_TRIAD",
        "ABSTRACT_NOUN_CLUSTER",
        "ADJACENT_FRAME_ONLY",
        "ARTIFICIAL_STAKES",
        "AUTHORITY_INFLATION",
        "AUTOMATIC_CONTRARIANISM",
        "AUTOMATIC_OPTIMISM",
        "AUTOMATIC_SKEPTICISM",
        "BOTH_SIDES_PADDING",
        "CHECKLIST_AUTHORITY",
        "COLON_HOOK",
        "COMPRESSION_AS_CONTRIBUTION",
        "CONCRETE_TO_ABSTRACT_DRIFT",
        "CONDITIONAL_ESCAPE",
        "CORPORATE_VOICE",
        "CREATOR_COACH_VOICE",
        "EMPTY_THESIS",
        "EM_DASH_OVERUSE",
        "ENGAGEMENT_BAIT",
        "EXCESSIVE_SIGNPOSTING",
        "EXCESSIVE_TRANSITIONS",
        "EXHAUSTIVE_COMPLETENESS",
        "EXPERTISE_POSTURING",
        "EXPLANATION_LEAKAGE",
        "FAKE_CONVERSATIONALITY",
        "FAKE_DEPTH",
        "FAKE_ROUGHNESS",
        "FALSE_CASUALNESS",
        "FORCED_NOVELTY",
        "FORMAT_MISMATCH",
        "FORMAT_OVERPRODUCTION",
        "FRAMEWORK_PACKAGING",
        "GENERIC_CONCLUSION",
        "GENERIC_LINE",
        "GENERIC_VERIFICATION",
        "HISTORICAL_CONTRAST_FRAME",
        "HUMANIZER_OVERREACH",
        "IMAGINARY_AWARENESS_GAP",
        "IMPLIED_POINT_RESTATEMENT",
        "INSIGHT_PACKAGING",
        "INSUFFICIENT_CONTEXT",
        "INVENTED_ADJACENT_COMPLICATION",
        "INVENTED_EMOTION",
        "LEXICAL_MARKER_CLUSTER",
        "LIST_SHAPED_PROSE",
        "LOCAL_TO_UNIVERSAL_ESCALATION",
        "MANUFACTURED_CONTRAST",
        "MECHANICAL_SYMMETRY",
        "MECHANICAL_TRICOLON",
        "MEDIA_BRAND_VOICE",
        "METAPHOR_STACKING",
        "META_GENERATION_COMMENTARY",
        "NOMINALIZATION_OVERLOAD",
        "NOVELTY_DEFICIT",
        "NOVELTY_OVER_FIDELITY",
        "NO_REPLY_NEEDED",
        "OVERCOMPRESSED_THEORY",
        "OVEREXPLAINED_ENDING",
        "PARTICIPIAL_CLAUSE_OVERLOAD",
        "PERFECT_PARAGRAPH_ARC",
        "PERFORMATIVE_FRAGMENTS",
        "PLACEHOLDER_COMMENTARY",
        "POLISHED_CAUSAL_ARC",
        "PRECISION_THEATER",
        "PREDICTIVE_VAGUENESS",
        "QUOTABLE_ENDING",
        "RECENT_ANGLE_SIMILARITY",
        "RECENT_ENDING_SIMILARITY",
        "RECENT_OPENING_SIMILARITY",
        "RECENT_POST_SIMILARITY",
        "RECENT_RHETORIC_SIMILARITY",
        "REDUNDANT_PARAGRAPH",
        "REPEATED_EXPLANATION",
        "REPLY_OVERPRODUCTION",
        "PUNCHLINE_COMPULSION",
        "ONE_LINER_FACTORY",
        "RECENT_SYNTAX_REUSE",
        "QUOTE_CARD_REPLY",
        "FALSE_BREVITY",
        "HUMOR_ON_TRAGEDY",
        "HUMOR_INTENT_UNCERTAIN",
        "HUMOR_INTENT_CONFLICT",
        "REPLY_TYPE_MISMATCH",
        "REVERSAL_HOOK",
        "RHETORICAL_QUESTION_BAIT",
        "RHETORICAL_QUESTION_THESIS",
        "SCOPE_OVERREACH",
        "SEMANTIC_FUNCTION_EVASION",
        "SOURCE_AS_SUBJECT",
        "QUOTE_CONTEXT_REQUIRED",
        "STANDALONE_CONTEXT_GAP",
        "WEAK_ADDED_VALUE",
        "SOURCE_ECHO",
        "SOURCE_LANGUAGE_MIMICRY",
        "MISSING_SOURCE_CONTEXT",
        "MULTI_SOURCE_COLLAGE",
        "SOURCE_SUMMARY_ONLY",
        "SOURCE_THESIS_REPACKAGING",
        "TOPIC_LABEL_EXPANSION",
        "UNIFORM_SENTENCE_RHYTHM",
        "UNIVERSAL_PRESCRIPTION",
        "UNNECESSARY_METAPHOR",
        "UNSPECIFIC_UNCERTAINTY",
        "UNSUPPORTED_ANGLE",
        "UNSUPPORTED_CERTAINTY",
        "UNSUPPORTED_FACT",
        "UNSUPPORTED_FIRST_PERSON",
        "UNSUPPORTED_GROUP_GENERALIZATION",
        "UNSUPPORTED_INFERENCE",
        "UNSUPPORTED_STAKES",
        "UNSUPPORTED_SUPERLATIVE",
        "UNSUPPORTED_UNIVERSAL",
        "VAGUE_ATTRIBUTION",
        "VAGUE_LITERAL_CORE",
        "ABSTRACTION_CREEP",
        "ABSTRACT_NOUN_DENSITY",
        "ABSTRACT_PREDICATE",
        "ABSTRACT_TEST_FRAMING",
        "ANALYTICAL_PACKAGING",
        "CONCLUSION_RESTATEMENT",
        "EXAMPLE_TO_UNIVERSAL_LESSON",
        "FORCED_SYNTHESIS",
        "LONG_FORM_OVERCOMPRESSION",
        "MANUFACTURED_QUESTION",
        "MINI_ESSAY_ARC",
        "NOT_ONLY_BUT_REVEAL",
        "PARAGRAPH_RESTATEMENT",
        "PARAGRAPH_THESIS_LOOP",
        "PARAGRAPH_WITHOUT_JOB",
        "PRODUCT_MEMO_VOICE",
        "QUOTE_CARD_SENTENCE_CLUSTER",
        "RECAP_TRANSITION",
        "REPEATED_PARAGRAPH_ARC",
        "RHETORICAL_QUESTION_CHAIN",
        "STAGED_THESIS",
        "SYNTHETIC_COHESION",
        "TRANSITION_SCAFFOLDING",
        "UNIFORM_PARAGRAPH_CADENCE",
        "VAGUE_USER_PROXY",
        "VARIANT_PARAPHRASE",
        "NO_POST_NEEDED",
        "NO_GROUNDED_CONTRIBUTION",
        "ANALYTICAL_PARAPHRASE_LOOP",
        "ABSTRACT_RESTATEMENT",
        "SAME_POINT_MULTIPLE_WAYS",
        "INVENTED_COMPARISON_TARGET",
        "DEPERSONALIZING_RENAME",
        "SOURCE_FACT_OVEREXPLANATION",
        "RELATION_NOVELTY_MISSING",
        "LEXICAL_NOVELTY_ONLY",
        "UNSUPPORTED_RELATION",
        "UNSUPPORTED_CONTRIBUTION",
        "SEMANTIC_SOURCE_ECHO",
        "PARAGRAPH_JOB_DUPLICATION",
    }
)
_WORDS = re.compile(r"[a-z0-9_@#-]{2,}", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s)\]}>,]+", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|mine|we|we're|we've|our|ours)\b", re.I)
_ACTION_VERBS = re.compile(
    r"\b(?:said|says|announced|opened|closed|cut|raised|launched|filed|denied|"
    r"confirmed|removed|added|changed|blocked|paid|moved|requires|shows|means|"
    r"cites|links|uses|affects|routes|settles|prices|rewards|limits|revokes|"
    r"writes?|prevents|exposes|matches|ships?|votes?|buys?|sells?|charges?|"
    r"releases?|publishes?|deposits?|withdraws?|allocates?|moves?)\b",
    re.I,
)
_CONCRETE_MECHANISMS = re.compile(
    r"\b(?:api|evals?|corrections?|trace|snapshot|allocation|vesting|fee|route|"
    r"filing|operator|approval|gate|contract|token|protocol|wallet|order book|"
    r"liquidity|validator|settlement|shipping|repository|database|model|workflow|"
    r"latency|price|volume|vote|proposal|document|release|commit|transaction)\b",
    re.I,
)
_ABSTRACT_NOUN = re.compile(
    r"\b(?:innovation|transformation|alignment|adoption|execution|trust|"
    r"transparency|utility|impact|ecosystem|engagement|resilience|ownership|"
    r"portability|judgment|intelligence|context|strategy|infrastructure|"
    r"opportunity|potential|landscape|momentum|narrative|community|signal|noise|moat)\b",
    re.I,
)

_STANDALONE_ABSTRACT_TEST = re.compile(
    r"\b(?:a|the|this) (?:narrow |real |practical )?test\b|\btest of (?:the )?"
    r"(?:format|model|market|system|approach|strategy)\b",
    re.I,
)
_STANDALONE_MANUFACTURED_QUESTION = re.compile(
    r"\b(?:the|a) (?:real |practical |main )?question (?:is|becomes|will be)\b",
    re.I,
)
_STANDALONE_NOT_ONLY_REVEAL = re.compile(
    r"\bnot (?:just|only)\b.{0,120}\b(?:but|it is|it's)\b|"
    r"\bless about\b.{0,100}\bmore about\b|"
    r"\b(?:x|this) is one thing;? .{0,80}\bwhat matters\b",
    re.I | re.S,
)
_STANDALONE_PRODUCT_MEMO = re.compile(
    r"\b(?:practical implication|operating model|adoption curve|user experience|"
    r"market structure|product strategy|format expansion|implementation question|"
    r"the broader implication)\b",
    re.I,
)
_STANDALONE_ABSTRACT_PREDICATE = re.compile(
    r"\b(?:creates|becomes|represents|signals|reflects|turns into|functions as)\b"
    r".{0,70}\b(?:test|question|framework|dynamic|implication|experience|signal)\b",
    re.I,
)
_STANDALONE_VAGUE_USER = re.compile(r"\b(?:people|users|the market|audiences?)\b", re.I)
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "before",
    "could",
    "from",
    "have",
    "into",
    "more",
    "only",
    "other",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "until",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "will",
    "might",
    "than",
    "then",
    "they",
    "them",
    "your",
    "you",
}

# Contextual warnings. A match is combined with evidence/specificity checks rather than treated as
# proof that a sentence is invalid merely because it contains one watched token.

_HUMOR_REPLY_ANGLE_TYPES = {"concise_joke"}
_HARMFUL_JOKE_ASSERTION = re.compile(
    r"\b(?:stole|stolen|scam(?:med)?|fraud|hacked|criminal|bribed|killed|murdered)\b|"
    r"\b(?:broke|breached|violated)\s+(?:\w+\s+){0,2}nda(?:s)?\b",
    re.IGNORECASE,
)
_REPLY_OPENING_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("turns_out", re.compile(r"^turns out\b", re.IGNORECASE)),
    (
        "finally_shell",
        re.compile(
            r"^(?:finally\b|at last\b|we(?:'ve| have)? finally (?:got|have)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "nothing_says",
        re.compile(
            r"^(?:nothing|few things)\s+(?:says?|communicates?|signals?)\b",
            re.IGNORECASE,
        ),
    ),
    ("plot_twist", re.compile(r"^plot twist\b", re.IGNORECASE)),
    ("achievement_unlocked", re.compile(r"^achievement unlocked\b", re.IGNORECASE)),
)

_PATTERN_RULES: tuple[tuple[str, re.Pattern[str], str, str], ...] = (
    (
        "REVERSAL_HOOK",
        re.compile(
            r"^(?:the part .+?(?:late|miss)|the uncomfortable truth|"
            r"here(?:'s| is) what .+ missing|the real story|the biggest mistake|"
            r"the interesting part|what matters here|the signal|the takeaway)\b",
            re.I,
        ),
        "warning",
        "flatten",
    ),
    (
        "MANUFACTURED_CONTRAST",
        re.compile(r"\b(?:it|this) (?:isn'?t|is not)\b.{1,100}\b(?:it is|it's|but)\b", re.I),
        "warning",
        "flatten",
    ),
    (
        "MANUFACTURED_CONTRAST",
        re.compile(r"\bpeople think\b.{1,100}\bactually\b", re.I),
        "warning",
        "flatten",
    ),
    (
        "ARTIFICIAL_STAKES",
        re.compile(
            r"\b(?:changes everything|defining moment|major inflection point|new era|"
            r"reshape the landscape|implications are profound|game changer|wake-up call)\b",
            re.I,
        ),
        "error",
        "delete",
    ),
    (
        "FAKE_DEPTH",
        re.compile(
            r"\b(?:let that sink in|read that again|the real test starts now|"
            r"bigger than it looks|go far beyond|the final benchmark is reality|"
            r"the future is already here|who owns the future)\b",
            re.I,
        ),
        "error",
        "delete",
    ),
    (
        "PLACEHOLDER_COMMENTARY",
        re.compile(
            r"\b(?:details|execution|trust|transparency|adoption|utility|distribution) "
            r"(?:(?:will|would|could|may) (?:matter|be key)|matters?)\b",
            re.I,
        ),
        "error",
        "ground",
    ),
    (
        "PLACEHOLDER_COMMENTARY",
        re.compile(
            r"\b(?:details|fundamentals) matter more than (?:the )?(?:noise|hype)\b",
            re.I,
        ),
        "error",
        "ground",
    ),
    (
        "PLACEHOLDER_COMMENTARY",
        re.compile(
            r"\b(?:time will tell|the market will decide|the community will decide|"
            r"the real test (?:is|comes)|the community (?:is|will be) watching)\b",
            re.I,
        ),
        "error",
        "delete",
    ),
    (
        "GENERIC_CONCLUSION",
        re.compile(
            r"\b(?:companies that .+ will win|organizations that adapt|"
            r"this is where the future|the lesson is simple|only time will tell|"
            r"remains to be seen|that is how the next generation)\b",
            re.I,
        ),
        "error",
        "delete",
    ),
    (
        "GENERIC_VERIFICATION",
        re.compile(
            r"\b(?:information|this|the claim) requires (?:independent|further) verification\b",
            re.I,
        ),
        "error",
        "attribute",
    ),
    (
        "GENERIC_VERIFICATION",
        re.compile(
            r"\b(?:claims could not be independently verified|verification is needed|"
            r"situation remains fluid|more details are expected)\b",
            re.I,
        ),
        "error",
        "attribute",
    ),
    (
        "VAGUE_ATTRIBUTION",
        re.compile(
            r"\b(?:there (?:is|are) (?:a )?report|reports (?:suggest|indicate)|"
            r"people are saying|many believe|observers suggest|the community thinks|"
            r"some users claim)\b",
            re.I,
        ),
        "error",
        "attribute",
    ),
    (
        "CONDITIONAL_ESCAPE",
        re.compile(r"^(?:if|should)\b.{0,160}\b(?:could|may|might|will)\b", re.I),
        "warning",
        "ground",
    ),
    (
        "CONDITIONAL_ESCAPE",
        re.compile(r"^if (?:this|that|it) happens\b", re.I),
        "error",
        "ground",
    ),
    (
        "PREDICTIVE_VAGUENESS",
        re.compile(
            r"\b(?:could|may|might) (?:reshape|drive adoption|attract more users|"
            r"increase trust|have (?:significant|major) implications)\b",
            re.I,
        ),
        "error",
        "ground",
    ),
    (
        "CORPORATE_VOICE",
        re.compile(
            r"\b(?:drive meaningful impact|unlock value|"
            r"empower(?:s|ed|ing)? users|foster innovation|"
            r"robust ecosystem|evolving landscape|seamless integration|"
            r"accelerate transformation|deliver outcomes|leverage synergies)\b",
            re.I,
        ),
        "error",
        "flatten",
    ),
    (
        "CREATOR_COACH_VOICE",
        re.compile(
            r"\b(?:founders need to understand|most people are not ready|save this for later|"
            r"the winners will be|do not sleep on this|lesson for every builder)\b",
            re.I,
        ),
        "error",
        "delete",
    ),
    (
        "MEDIA_BRAND_VOICE",
        re.compile(
            r"^(?:in a major development|the announcement comes amid|the move signals|"
            r"this marks a significant step|industry observers are watching)\b",
            re.I,
        ),
        "warning",
        "flatten",
    ),
    (
        "FAKE_CONVERSATIONALITY",
        re.compile(r"\b(?:ngl|tbh|let'?s be real|here'?s the thing|honestly|listen)\b", re.I),
        "warning",
        "flatten",
    ),
    (
        "AUTOMATIC_CONTRARIANISM",
        re.compile(
            r"^(?:everyone is focused|most people are missing|the consensus is wrong|"
            r"unpopular opinion)\b",
            re.I,
        ),
        "warning",
        "regenerate_angle",
    ),
    (
        "BOTH_SIDES_PADDING",
        re.compile(
            r"\b(?:opportunities,? but also challenges|"
            r"promising,? (?:although|but) risks remain)\b",
            re.I,
        ),
        "warning",
        "delete",
    ),
    (
        "EXCESSIVE_SIGNPOSTING",
        re.compile(
            r"^(?:here is the thing|the key point is|the takeaway is|in other words|"
            r"put differently|more importantly|ultimately|to be clear|"
            r"it is worth noting)\b",
            re.I,
        ),
        "warning",
        "delete",
    ),
    (
        "COLON_HOOK",
        re.compile(
            r"^(?:the problem|the answer|the catch|the lesson|the result|the takeaway):", re.I
        ),
        "warning",
        "flatten",
    ),
    (
        "QUOTABLE_ENDING",
        re.compile(
            r"^(?:own the loop|portable context is the real moat|the model is replaceable|"
            r"the learning is not|.+ own the future)\.?$",
            re.I,
        ),
        "warning",
        "delete",
    ),
)

_TRANSITIONS = re.compile(
    r"^(?:however|moreover|additionally|furthermore|consequently|nevertheless|"
    r"therefore|meanwhile|in contrast|on the other hand)\b",
    re.I,
)
_PARTICIPIAL = re.compile(
    r"\b(?:allowing|creating|making|ensuring|highlighting|underscoring)\b", re.I
)
_NOMINALIZATION = re.compile(
    r"\b(?:the (?:implementation|optimization|establishment|development|"
    r"preservation|accumulation) of)\b",
    re.I,
)
_LEXICAL_WATCH = re.compile(
    r"\b(?:delve|intricate|tapestry|underscore|pivotal|crucial|robust|transformative|testament|showcase|foster|elevate|unlock|empower|leverage|navigate|multifaceted|nuanced|seamless|groundbreaking|vibrant|landscape)\b",
    re.I,
)

_UNSUPPORTED_SUPERLATIVE = re.compile(
    r"\b(?:the part that matters most|the (?:most important|biggest|hardest|real|true|only|main) "
    r"(?:part|risk|problem|issue|layer|thing|cost|source|value|story|reason|moat)|"
    r"the only thing that matters|matters most|everything else is replaceable)\b",
    re.I,
)
_DOMAIN_RANKING = re.compile(
    r"\b(?:main function|primary key|hardest benchmark|main branch|main thread|"
    r"primary source|primary market)\b",
    re.I,
)
_AWARENESS_GAP = re.compile(
    r"\b(?:(?:most|few) (?:teams|companies|people|founders|users|investors|developers) "
    r"(?:do not|don'?t|will|would|still|never)?\s*(?:realize|understand|see|notice|"
    r"underestimate|miss|ignore|think)|(?:nobody|everyone) "
    r"(?:realizes|understands|sees|is talking about|thinks)|people are missing|"
    r"will realize too late|still underappreciated|(?:teams|companies) think)\b",
    re.I,
)
_GROUP_GENERALIZATION = re.compile(
    r"\b(?:companies|founders|users|investors|developers|teams|the market|the community|"
    r"the industry|everyone|nobody|most people|most teams|most companies)\b",
    re.I,
)
_HISTORICAL_CONTRAST = re.compile(
    r"\b(?:companies|teams|we|people) (?:spent|fought|worked) "
    r"(?:years|a decade|decades|months)\b.{0,180}\b(?:now|only to|then|overnight|casually)\b",
    re.I | re.S,
)
_UNSUPPORTED_STAKES = re.compile(
    r"\b(?:dangerous|riskier|risky|serious (?:risk|problem|issue|concern|threat)|"
    r"critical|alarming|concerning|costly|"
    r"high[- ]stakes|bigger than (?:it|this) looks|far more important|"
    r"changes the risk profile|the stakes are higher)\b",
    re.I,
)
_RISK_MECHANISM = re.compile(
    r"\b(?:expos(?:e|es|ed|ure)|access|retain(?:s|ed)?|store(?:s|d)?|leak(?:s|ed)?|"
    r"reuse|migrat(?:e|es|ed|ion)|switch(?:es|ed|ing)?|depend(?:s|ed|ency)|lock[- ]?in|"
    r"cannot|can'?t|difficult|harder|lose|loss|deleted?|revoked?|provider-specific)\b",
    re.I,
)
_GRAND_IDENTITY = re.compile(
    r"\b(?:institutional memory|organizational intelligence|(?:the )?real moat|"
    r"operating system of|source code of|company learning(?: out loud)?|encoded judgment)\b",
    re.I,
)
_INSIGHT_PACKAGING = re.compile(
    r"\b(?:corrections?|evals?|feedback loop|workflow history|context|correction history|"
    r"traces?|prompts?|data|workflow|dashboard)\b.{0,55}\b(?:company learning(?: out loud)?|"
    r"institutional memory|organizational intelligence|operating system|(?:real )?moat|source code|"
    r"encoded judgment|company(?:'s)? intelligence|place where .+ gets smarter|"
    r"nervous system|engine of judgment)\b",
    re.I,
)
_METAPHOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("learning_out_loud", re.compile(r"\blearning out loud\b", re.I)),
    (
        "leave_building",
        re.compile(r"\b(?:leave|leaving|letting .{0,40} leave) the building\b", re.I),
    ),
    (
        "keys",
        re.compile(
            r"\bkeys? to (?:the )?(?:company|organization|company(?:'s)? intelligence|"
            r"intelligence)\b",
            re.I,
        ),
    ),
    ("moat", re.compile(r"\b(?:real )?moat\b", re.I)),
    ("operating_system", re.compile(r"\boperating system of\b", re.I)),
    ("institutional_memory", re.compile(r"\binstitutional memory\b", re.I)),
    ("source_code", re.compile(r"\bsource code of\b", re.I)),
    ("encoded_judgment", re.compile(r"\bencoded judgment\b", re.I)),
    ("gets_smarter", re.compile(r"\b(?:company|organization) gets smarter\b", re.I)),
    ("compounding_loop", re.compile(r"\bfeedback loop (?:compounds?|snowballs?)\b", re.I)),
    ("own_loop", re.compile(r"\bown(?:s|ed|ing)? the loop\b", re.I)),
    (
        "abstract_identity",
        re.compile(
            r"\b(?:is|are|becomes?|turns? into) (?:the )?(?:memory|intelligence|brain|"
            r"nervous system|engine|foundation|source code|operating system|moat)\b",
            re.I,
        ),
    ),
)
_SLOGAN_ENDING = re.compile(
    r"^(?:own|keep|protect|control|build|remember)\b.{0,70}$|"
    r"\b(?:the future|the moat|the lesson|the real value)\b.{0,40}$",
    re.I,
)
_GENERIC_RHETORICAL_QUESTION = re.compile(
    r"\b(?:who owns the future|what does this mean for the future|are we ready|"
    r"what are we missing|who is paying attention)\??$",
    re.I,
)
_CASUAL_MARKERS = re.compile(r"\b(?:basically|honestly|pretty|feels like|ngl|tbh|lol)\b", re.I)
_CONTRACTIONS = re.compile(
    r"\b(?:isn'?t|don'?t|can'?t|won'?t|it'?s|that'?s|they'?re|we'?re)\b", re.I
)
_CAUSAL_CONNECTORS = re.compile(
    r"\b(?:therefore|which means|that means|so |as a result|this makes|becomes?|leads? to|"
    r"turns? into|creates?)\b",
    re.I,
)
_UNIVERSAL_PRESCRIPTION = re.compile(
    r"\b(?:(?:every|all|any) (?:serious )?(?:ai )?"
    r"(?:company|companies|team|teams|founder|founders|contract|buyer|organization)|"
    r"(?:serious|responsible|mature) "
    r"(?:company|companies|team|teams|founder|founders|contract|buyer|organization)|"
    r"(?:companies|teams|founders|organizations) (?:must|need to|should))\b",
    re.I,
)
_VAGUE_REPORT_LEAD = re.compile(
    r"\b(?:(?:one|a|another|the) (?:report|account|source) "
    r"(?:says|said|claims|reports|suggests|indicates)|according to "
    r"(?:a|one|the|some)?\s*(?:report|reports|account|accounts|source|sources))\b",
    re.I,
)
_ACCURACY_DISCLAIMER = re.compile(
    r"\b(?:if|assuming) (?:that|this|the) (?:account|report|source|claim) "
    r"(?:is|was|turns out to be) accurate\b|^if (?:true|accurate)\b",
    re.I,
)
_REPORT_PROCESS_LANGUAGE = re.compile(
    r"\b(?:the|this|that) (?:report|account|source) "
    r"(?:does not|doesn'?t|did not|didn'?t) "
    r"(?:explain|show|say|provide|identify|include)\b|"
    r"\b(?:the )?supplied (?:source|sources|evidence|evidence packet|packet)\b",
    re.I,
)
_REPORT_META_LANGUAGE = re.compile(
    r"\b(?:report|reports|account|accounts|source|sources|evidence packet)\b",
    re.I,
)

_SOURCE_NARRATION = re.compile(
    r"(?:@[A-Za-z0-9_]+|the (?:source|account|post|tweet)|this (?:source|post|tweet))\s+"
    r"(?:calls?|says?|writes?|posts?|highlights?|shows?|describes?|frames?|claims?|notes?)\b",
    re.I,
)
_COLLAGE_LANGUAGE = re.compile(
    r"\b(?:one (?:post|line|account|caption).{0,80}(?:the )?other|"
    r"same (?:sport|conversation|topic).{0,60}different|"
    r"the contrast (?:between|is).{0,120}(?:post|line|caption|account))\b",
    re.I | re.S,
)

_ANNOUNCEMENT_BEFORE_CONTENT = re.compile(
    r"\b(?:the (?:important|interesting|uncomfortable|surprising) (?:part|thing|question)|"
    r"here(?:'s| is) (?:the|what)|what (?:matters|to notice|this means)|"
    r"the (?:lesson|reason this matters|deeper issue) (?:is|here))\b",
    re.I,
)
_IMPORTANCE_ANNOUNCEMENT = re.compile(
    r"\b(?:this is (?:important|huge|wild|surprising|uncomfortable)|"
    r"worth (?:noting|remembering|watching)|pay attention to this)\b",
    re.I,
)
_INSIGHT_PREVIEW = re.compile(
    r"\b(?:the (?:real|deeper|hidden) (?:lesson|point|meaning|story)|"
    r"what this (?:really )?(?:shows|reveals|tells us)|here's why)\b",
    re.I,
)
_UNASKED_REASSURANCE = re.compile(
    r"\b(?:and that(?:'s| is) (?:okay|fine)|it(?:'s| is) (?:okay|fine|normal) to|"
    r"you (?:don't|do not) have to|there(?:'s| is) nothing wrong with)\b",
    re.I,
)
_READER_HAND_HOLDING = re.compile(
    r"\b(?:remember|notice|imagine|keep in mind|the takeaway is|what you should conclude)\b",
    re.I,
)
_FALSE_AGENCY = re.compile(
    r"\b(?:the market|the industry|the company|the organization|the technology|the data) "
    r"(?:decided|wanted|chose|learned|judged|spoke|rewarded|realized)\b",
    re.I,
)
_DISEMBODIED_EXPLANATION = re.compile(
    r"\b(?:this|that|it) (?:happens?|happened|works?|changed) because\b", re.I
)
_REDUNDANT_MODIFIER = re.compile(
    r"\b(?:very|really|quite|simply|basically|clearly|obviously|literally)\b", re.I
)


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORDS.finditer(text)}


def _content_tokens(text: str) -> set[str]:
    return _tokens(text) - _STOPWORDS


def _similarity(left: str, right: str) -> float:
    left_normalized = " ".join(left.casefold().split())
    right_normalized = " ".join(right.casefold().split())
    if not left_normalized or not right_normalized:
        return 0.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = _tokens(left_normalized)
    right_tokens = _tokens(right_normalized)
    jaccard = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )
    return max(sequence, jaccard)


def _issue(
    code: str,
    message: str,
    *,
    severity: str = "error",
    sentence_index: int | None = None,
    action: str | None = None,
) -> QualityIssue:
    if code not in QUALITY_ISSUE_CODES:
        raise ValueError(f"Unknown quality issue code: {code}")
    return QualityIssue(
        code=code,
        severity=severity,
        message=message,
        sentence_index=sentence_index,
        suggested_action=action,
    )


def _evidence_tokens(packet: EvidencePacket) -> set[str]:
    tokens: set[str] = set()
    for item in packet.items:
        tokens.update(_tokens(item.text))
        if item.author_or_source:
            tokens.update(_tokens(item.author_or_source))
    return tokens


def _first_words(text: str, count: int = 5) -> str:
    return " ".join(match.group(0).casefold() for match in list(_WORDS.finditer(text))[:count])


def _last_sentence(text: str) -> str:
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return sentences[-1] if sentences else ""


def _paragraphs(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"\n\s*\n", text) if part.strip())


def _metaphor_frames(text: str) -> set[str]:
    frames = {name for name, pattern in _METAPHOR_PATTERNS if pattern.search(text)}
    identities = re.findall(
        r"\b(?:is|are|becomes?|turns? into) (?:the )?"
        r"(memory|intelligence|brain|nervous system|engine|foundation|source code|"
        r"operating system|moat)\b",
        text,
        re.I,
    )
    frames.update(f"identity:{identity.casefold()}" for identity in identities)
    return frames


def _metaphor_count(text: str) -> int:
    return len(_metaphor_frames(text))


def _rhetorical_device_count(issues: Sequence[QualityIssue]) -> int:
    device_codes = {
        "REVERSAL_HOOK",
        "MANUFACTURED_CONTRAST",
        "ARTIFICIAL_STAKES",
        "FAKE_DEPTH",
        "GENERIC_CONCLUSION",
        "QUOTABLE_ENDING",
        "MECHANICAL_SYMMETRY",
        "MECHANICAL_TRICOLON",
        "COLON_HOOK",
        "RHETORICAL_QUESTION_BAIT",
        "HISTORICAL_CONTRAST_FRAME",
        "UNSUPPORTED_SUPERLATIVE",
        "INSIGHT_PACKAGING",
        "METAPHOR_STACKING",
        "IMAGINARY_AWARENESS_GAP",
        "UNSUPPORTED_STAKES",
        "POLISHED_CAUSAL_ARC",
        "UNNECESSARY_METAPHOR",
        "FALSE_CASUALNESS",
    }
    return len({issue.code for issue in issues if issue.code in device_codes})


def _is_reply(packet: EvidencePacket | None) -> bool:
    return packet is not None and packet.mode.value == "reply"


def _source_specific_humorous_reply(
    variant: GenerationVariant,
    *,
    text: str,
    evidence_packet: EvidencePacket | None,
    evidence_tokens: set[str],
    reply_mode: bool,
) -> bool:
    """Recognize a narrow, evidence-bound humorous reply without treating humor as fact."""

    if not reply_mode:
        return False
    humor_shape = variant.angle_type in _HUMOR_REPLY_ANGLE_TYPES or bool(
        variant.angle_type == "dry_observation"
        and re.search(
            r"\b(?:turns out|finally|bro|unfortunately|confirmed|always called|"
            r"download button|anti-yap|has left the chat)\b",
            text,
            re.I,
        )
    )
    if not humor_shape:
        return False
    if evidence_packet is None or not inspect_reaction_function(evidence_packet, text).grounded:
        return False
    if not inspect_humor_safety(evidence_packet, text).suitable_for_humor:
        return False
    if len(text) > 280 or len(_paragraphs(text)) != 1:
        return False
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if len(sentences) > 2:
        return False
    content_tokens = _expanded_content_tokens(text)
    if not (content_tokens & evidence_tokens):
        return False
    evidence_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", " ".join(evidence_tokens)))
    text_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    if text_numbers - evidence_numbers:
        return False
    return not (
        _HARMFUL_JOKE_ASSERTION.search(text)
        and not _HARMFUL_JOKE_ASSERTION.search(" ".join(evidence_tokens))
    )


def _reply_opening_family(text: str) -> str | None:
    for family, pattern in _REPLY_OPENING_FAMILIES:
        if pattern.search(text.strip()):
            return family
    return None


_REDUCTIVE_SLOT_MARKER = re.compile(
    r"\b(?:just|merely|basically|paperwork|sales|branding|progress bar|extra steps|"
    r"busywork|bureaucracy|marketing|copy|wrapper|rename|rebrand|overhead|theater|"
    r"chore|bottleneck|better ux|product strategy|product clothing)\b",
    re.I,
)


def inspect_reply_shell(
    text: str,
    *,
    evidence_packet: EvidencePacket | None = None,
) -> ReplyShellInspection:
    """Extract a typed semantic shell and project it into the established contract."""

    semantic_ir = extract_semantic_inspection("", text)
    shell = adjudicate_reply_shell(semantic_ir)
    source_specific = False
    if evidence_packet is not None:
        evidence_tokens = _evidence_tokens(evidence_packet)
        source_specific = bool(_expanded_content_tokens(text) & evidence_tokens)
    return shell.model_copy(update={"source_specific": source_specific})


def _reply_shell_signature(
    shell: ReplyShellInspection,
) -> tuple[str, bool, bool, bool, bool] | None:
    if shell.operator in {"identity", "reduction", "decorated_identity"}:
        relation = "identity_reduction"
    elif shell.operator == "reveal":
        relation = "reveal"
    elif shell.operator == "inverse_praise":
        relation = "inverse_praise"
    elif shell.operator == "setup_twist":
        relation = "setup_twist"
    else:
        return None
    return (
        relation,
        shell.subject_span is not None,
        shell.complement_span is not None,
        shell.decoration_span is not None,
        shell.generic_reusability,
    )


def _reply_structure_families(text: str) -> tuple[str, ...]:
    """Return bounded communicative-shell families while ignoring topic nouns."""

    normalized = " ".join(text.casefold().split()).strip(" \"“”'")
    shell = inspect_reply_shell(text)
    families: list[str] = []
    if shell.operator in {"identity", "reduction", "decorated_identity"}:
        families.append("reductive_identity")
    if shell.operator == "decorated_identity":
        families.append("decorated_identity")
    if shell.operator == "inverse_praise":
        families.append("inverse_praise")
    if shell.operator == "reveal":
        families.append("reveal_reduction")
    if shell.operator == "setup_twist":
        families.append("setup_twist")
    patterns = (
        (
            "finally_shell",
            re.compile(r"^(?:finally\b|at last\b|we(?:'ve| have)? finally (?:got|have)\b)"),
        ),
        (
            "fake_newsroom",
            re.compile(r"^(?:breaking|update|sources? (?:confirm|say)|confirmed:)\b"),
        ),
        ("achievement", re.compile(r"\bachievement unlocked\b")),
        (
            "chat_metaphor",
            re.compile(r"\b(?:entered|left|has entered|has left) the (?:group )?chat\b"),
        ),
        ("speedrun_metaphor", re.compile(r"\bspeedrun\b")),
        (
            "heavy_lifting_metaphor",
            re.compile(r"\b(?:doing|does|did|carrying|carries) (?:the )?heavy lifting\b"),
        ),
        (
            "switch_metaphor",
            re.compile(r"\b(?:official )?(?:button|switch|toggle)\b"),
        ),
        ("early_access_metaphor", re.compile(r"\bearly access\b")),
        (
            "crypto_invented",
            re.compile(r"\b(?:crypto|web3|ai) (?:invented|discovered|reinvented)\b"),
        ),
        (
            "high_sarcasm",
            re.compile(
                r"\b(?:nothing (?:says|screams)|few things "
                r"(?:communicate|demonstrate|show)|plot twist|achievement unlocked|"
                r"bold strategy|same energy|sure,? why not|of course)\b"
            ),
        ),
    )
    for name, pattern in patterns:
        if pattern.search(normalized):
            families.append(name)
    # Keep legacy family labels in metadata/tests while using the broader semantic shells.
    if "inverse_praise" in families:
        families.append("nothing_says")
    if "reveal_reduction" in families:
        families.append("turns_out")
    return tuple(dict.fromkeys(families))


def _generic_reusable_reply_shell(text: str, families: Sequence[str]) -> bool:
    """Flag bounded quote-card shells that are reusable across unrelated sources."""

    normalized = " ".join(text.casefold().split())
    generic_families = {"reductive_identity", "inverse_praise", "reveal_reduction"}
    if not (set(families) & generic_families):
        return False
    # Concrete names, direct second-person response, or a real question reduce generic reuse risk.
    has_named_entity = bool(re.search(r"(?:^|\s)@[A-Za-z0-9_]+|\b[A-Z][a-z]+\b", text))
    direct_response = bool(
        re.match(r"^(?:yes|no|i agree|i disagree|you|your|that|this)\b", text, re.I)
    )
    return not has_named_entity and not direct_response and not normalized.endswith("?")


def _reply_syntax_skeleton(text: str) -> str:
    """Create a bounded structural signature without random synonym comparison."""

    normalized = re.sub(r"https?://\S+|@[A-Za-z0-9_]+|\b\d+(?:[.,]\d+)?\b", " X ", text.casefold())
    tokens = re.findall(r"[a-z']+|[.!?,:;]", normalized)
    keep = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "just",
        "like",
        "with",
        "extra",
        "more",
        "steps",
        "nothing",
        "says",
        "finally",
        "turns",
        "out",
        "but",
        "make",
        "call",
        "it",
        "then",
        "actually",
        "apparently",
        "will",
        "can",
        "to",
        "whoever",
        "least",
        "most",
        "not",
        "only",
        "when",
        "why",
        "what",
        "how",
        "still",
        "now",
        "because",
        "so",
        "and",
        "or",
    }
    skeleton: list[str] = []
    content_open = False
    for lexical_item in tokens:
        if lexical_item in {".", "!", "?", ",", ":", ";"} or lexical_item in keep:
            skeleton.append(lexical_item)
            content_open = False
        elif not content_open:
            skeleton.append("X")
            content_open = True
    return " ".join(skeleton)


def inspect_reply_naturalness(
    variant: GenerationVariant,
    *,
    evidence_packet: EvidencePacket | None = None,
    recent_posts: Sequence[str] = (),
) -> ReplyNaturalnessInspection:
    """Categorically inspect reply construction and structural diversity."""

    text = variant.text.strip()
    word_count = len(_WORDS.findall(text))
    shell = inspect_reply_shell(text, evidence_packet=evidence_packet)
    families = _reply_structure_families(text)
    skeleton = _reply_syntax_skeleton(text)
    recent_families = [_reply_structure_families(item) for item in recent_posts[-12:]]
    recent_skeletons = [_reply_syntax_skeleton(item) for item in recent_posts[-12:]]
    recent_shells = [inspect_reply_shell(item) for item in recent_posts[-12:]]
    shell_signature = _reply_shell_signature(shell)
    recent_shell_signatures = tuple(
        signature
        for recent_shell in recent_shells
        if (signature := _reply_shell_signature(recent_shell)) is not None
    )
    if shell_signature is not None:
        reused = any(
            recent_signature[0] == shell_signature[0]
            and recent_signature[1:3] == shell_signature[1:3]
            and recent_signature[4]
            and shell_signature[4]
            for recent_signature in recent_shell_signatures
        )
    else:
        reused = bool(
            (skeleton and any(skeleton == recent for recent in recent_skeletons))
            or any(family in prior for family in families for prior in recent_families)
        )
    generic_reusability = _generic_reusable_reply_shell(text, families)
    evidence_tokens = _evidence_tokens(evidence_packet) if evidence_packet is not None else set()
    source_specific = (
        bool(_expanded_content_tokens(text) & evidence_tokens) if evidence_tokens else True
    )
    question = text.rstrip().endswith("?")
    direct = bool(re.match(r"^(?:yes|no|agreed|i agree|i disagree|that|this)\b", text, re.I))
    qualification = bool(
        re.search(
            r"\b(?:but|except|unless|usually|sometimes|half the time|only when)\b", text, re.I
        )
    )
    metadata_says_humor = variant.angle_type == "concise_joke" or variant.contribution_type in {
        "dry_humor",
        "dry_reframe",
        "joke",
    }
    humor_intent = (
        inspect_humor_intent(
            evidence_packet,
            text,
            metadata_says_humor=metadata_says_humor,
        )
        if evidence_packet is not None
        else None
    )
    humor = bool(
        metadata_says_humor or (humor_intent is not None and humor_intent.text_appears_humorous)
    )
    reaction = (
        inspect_reaction_function(evidence_packet, text)
        if humor and evidence_packet is not None
        else None
    )
    humor_safety = (
        inspect_humor_safety(evidence_packet, text)
        if (
            evidence_packet is not None
            and humor_intent is not None
            and humor_intent.humor_safety_required
        )
        else None
    )
    if reaction is not None and reaction.grounded:
        source_specific = True
    if question:
        response_type = "question"
    elif direct:
        response_type = "direct_response"
    elif qualification:
        response_type = "qualification"
    elif humor:
        response_type = "dry_humor"
    elif re.search(r"\b(?:because|so|which means|that leaves|the constraint)\b", text, re.I):
        response_type = "contextual_extension"
    else:
        response_type = "plain_observation"
    quote_card = bool(
        len(_SENTENCE_SPLIT.split(text)) == 1
        and 5 <= word_count <= 24
        and (
            "reductive_identity" in families
            or "inverse_praise" in families
            or "reveal_reduction" in families
            or "nothing_says" in families
            or "setup_twist" in families
            or re.search(
                r"\b(?:the real|the future|what matters|always|never|bold strategy|same energy)\b",
                text,
                re.I,
            )
        )
        and not question
    )
    punchline = bool(
        humor
        and (
            families
            or re.search(
                r"\b(?:plot twist|punchline|speedrun|rebrand|achievement unlocked|"
                r"has left the chat|bold strategy|same energy|nothing says)\b",
                text,
                re.I,
            )
        )
        and (reaction is None or not reaction.grounded)
    )
    false_brevity = bool(
        word_count < 4
        or text.endswith((":", "—"))
        or (word_count <= 7 and qualification and not re.search(r"[.!?]$", text))
    )
    constructed_markers = (
        len(set(families) - {"nothing_says", "turns_out"})
        + int(quote_card)
        + int(punchline)
        + int(generic_reusability and reused)
    )
    constructedness = (
        "high" if constructed_markers >= 2 else "medium" if constructed_markers else "low"
    )
    return ReplyNaturalnessInspection(
        response_type=response_type,
        constructedness=constructedness,
        punchline_compulsion=punchline,
        quote_card_risk=quote_card,
        recent_syntax_reuse=reused,
        false_brevity=false_brevity,
        source_specific=source_specific,
        structural_families=families,
        recent_structural_match=reused,
        generic_reusability=generic_reusability,
        reply_shell=shell,
        humor_intent=humor_intent,
        humor_safety=humor_safety,
    )


def _paragraphs_are_redundant(paragraphs: Sequence[str]) -> bool:
    if len(paragraphs) < 2:
        return False
    for left, right in pairwise(paragraphs):
        left_tokens = _content_tokens(left)
        right_tokens = _content_tokens(right)
        if not left_tokens or not right_tokens:
            continue
        same_root = any(
            left_word.startswith(right_word[:7]) or right_word.startswith(left_word[:7])
            for left_word in left_tokens
            for right_word in right_tokens
            if len(left_word) >= 7 and len(right_word) >= 7
        )
        right_is_abstract = len(_ABSTRACT_NOUN.findall(right)) >= 1 or _metaphor_count(right) >= 1
        if _similarity(left, right) >= 0.52 or (same_root and right_is_abstract):
            return True
    return False


def _concrete_to_abstract_drift(sentences: Sequence[str]) -> bool:
    text = " ".join(sentences)
    concrete_matches = list(_CONCRETE_MECHANISMS.finditer(text))
    abstract_matches = list(_ABSTRACT_NOUN.finditer(text))
    if not concrete_matches:
        return False
    first_concrete = concrete_matches[0].start()
    later_abstractions = sum(match.start() > first_concrete for match in abstract_matches)
    later_metaphors = sum(
        bool(pattern.search(text[first_concrete:])) for _, pattern in _METAPHOR_PATTERNS
    )
    return later_abstractions + later_metaphors >= 2


def _evidence_texts(packet: EvidencePacket | None) -> tuple[str, ...]:
    if packet is None:
        return ()
    return tuple(item.text for item in packet.items)


def _ranking_supported(sentence: str, packet: EvidencePacket | None) -> bool:
    if _DOMAIN_RANKING.search(sentence):
        return True
    match = _UNSUPPORTED_SUPERLATIVE.search(sentence)
    if match is None:
        return True
    phrase = " ".join(match.group(0).casefold().split())
    ranking_tokens = {
        token
        for token in _content_tokens(phrase)
        if token in {"most", "biggest", "hardest", "real", "true", "only", "main"}
    }
    for evidence in _evidence_texts(packet):
        normalized = " ".join(evidence.casefold().split())
        if phrase in normalized:
            return True
        if (
            ranking_tokens
            and ranking_tokens <= _content_tokens(normalized)
            and _similarity(sentence, evidence) >= 0.72
        ):
            return True
    return False


def _awareness_gap_supported(sentence: str, packet: EvidencePacket | None) -> bool:
    group = _GROUP_GENERALIZATION.search(sentence)
    if group is None:
        return False
    group_phrase = " ".join(group.group(0).casefold().split())
    awareness_terms = {
        term
        for term in (
            "realize",
            "understand",
            "see",
            "notice",
            "underestimate",
            "miss",
            "ignore",
            "think",
        )
        if re.search(rf"\b{term}(?:s|d|ed|ing)?\b", sentence, re.I)
    }
    for evidence in _evidence_texts(packet):
        normalized = " ".join(evidence.casefold().split())
        if group_phrase in normalized and any(term in normalized for term in awareness_terms):
            return True
        if _similarity(sentence, evidence) >= 0.78:
            return True
    return False


def _historical_contrast_supported(text: str, packet: EvidencePacket | None) -> bool:
    duration = re.search(r"\b(?:years|a decade|decades|months)\b", text, re.I)
    if duration is None:
        return False
    duration_phrase = duration.group(0).casefold()
    text_tokens = _content_tokens(text)
    for evidence in _evidence_texts(packet):
        if duration_phrase not in evidence.casefold():
            continue
        if len(text_tokens & _content_tokens(evidence)) >= 6:
            return True
    return False


def _stakes_are_grounded(sentence: str, packet: EvidencePacket | None) -> bool:
    mechanisms = {match.group(0).casefold() for match in _RISK_MECHANISM.finditer(sentence)}
    if not mechanisms:
        return False
    evidence_tokens = set().union(*(_content_tokens(text) for text in _evidence_texts(packet)))
    return bool(mechanisms & evidence_tokens or _content_tokens(sentence) & evidence_tokens)


def _has_casual_surface(text: str) -> bool:
    first_alpha = next((char for char in text if char.isalpha()), "")
    return bool(
        _CASUAL_MARKERS.search(text)
        or _CONTRACTIONS.search(text)
        or (first_alpha and first_alpha.islower())
    )


def _unsupported_group_claim(sentence: str, packet: EvidencePacket | None) -> bool:
    group_match = _GROUP_GENERALIZATION.search(sentence)
    if group_match is None:
        return False
    broad_predicate = re.search(
        r"\b(?:must|should|need to|do not|don'?t|will|would|understand|realize|"
        r"underestimate|ignore|want|prefer|decide|watch|care|believe|think)\b",
        sentence,
        re.I,
    )
    if not broad_predicate:
        return False
    group_phrase = " ".join(group_match.group(0).casefold().split())
    predicate_tokens = _content_tokens(broad_predicate.group(0))
    for evidence in _evidence_texts(packet):
        normalized = " ".join(evidence.casefold().split())
        if group_phrase in normalized and predicate_tokens & _content_tokens(normalized):
            return False
        if _similarity(sentence, evidence) >= 0.78:
            return False
    return True


def _topic_substitution_generic(text: str, topic: str) -> bool:
    topic_tokens = _content_tokens(topic)
    text_tokens = _content_tokens(text)
    if not topic_tokens or not text_tokens:
        return False
    remainder = text_tokens - topic_tokens
    generic = {
        "details",
        "execution",
        "trust",
        "transparency",
        "adoption",
        "utility",
        "community",
        "market",
        "important",
        "matter",
        "watching",
        "future",
        "key",
        "potential",
        "impact",
        "distribution",
        "regulation",
        "noise",
        "hype",
    }
    return len(remainder) <= 7 and len(remainder & generic) >= max(2, len(remainder) // 2)


def _has_supported_first_person(packet: EvidencePacket | None) -> bool:
    if packet is None:
        return False
    return any(_FIRST_PERSON.search(item.text) for item in packet.items)


def _expanded_content_tokens(text: str) -> set[str]:
    tokens = _content_tokens(text)
    expanded = set(tokens)
    for token in tokens:
        expanded.update(
            part for part in re.split(r"[-_/]", token) if len(part) >= 3 and part not in _STOPWORDS
        )
    return expanded


def _sentence_has_concrete_marker(sentence: str, evidence_tokens: set[str]) -> bool:
    tokens = _expanded_content_tokens(sentence)
    return bool(
        tokens & evidence_tokens
        or _ACTION_VERBS.search(sentence)
        or _CONCRETE_MECHANISMS.search(sentence)
        or re.search(r"(?:@[A-Za-z0-9_]+|\b[A-Z]{2,}\b|\d)", sentence)
    )


def assess_variant(
    variant: GenerationVariant,
    *,
    evidence_packet: EvidencePacket | None = None,
    minimum_specificity: float = 0.55,
    minimum_evidence: float = 0.75,
    minimum_naturalness: float = 0.55,
    minimum_characters: int | None = None,
    maximum_characters: int | None = None,
    recent_posts: Sequence[str] = (),
    allow_single_source_fact_check_gate: bool = False,
    editorial_intent: EditorialIntent = "comment_on_source",
    source_coverage: SourceCoverage | None = None,
    style_mode: str | None = None,
) -> QualityReport:
    """Validate one variant without relying on a model judge."""

    if evidence_packet is not None and editorial_intent == "comment_on_source":
        if evidence_packet.mode.value == "reply":
            editorial_intent = "reply_reaction"
        elif evidence_packet.mode.value == "quote_post":
            editorial_intent = "quote_reaction"
        elif evidence_packet.mode.value == "news_claim":
            editorial_intent = "report_event"
    text = variant.text.strip()
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    issues: list[QualityIssue] = []
    evidence_tokens = _evidence_tokens(evidence_packet) if evidence_packet is not None else set()
    structural_mode = style_mode or (
        evidence_packet.requested_format if evidence_packet is not None else None
    )
    sentence_job_uniqueness = 1.0
    semantic_repetition_ratio = 0.0
    supported_relation_ratio = 1.0
    semantic_decision: str | None = None
    abstract_restatement_ratio = 0.0
    rhetorical_scaffolding_ratio = 0.0
    paragraph_advancement_ratio = 1.0
    source_proposition_overlap = 0.0
    blocking_structure = {
        "REPEATED_PARAGRAPH_ARC",
        "CONCLUSION_RESTATEMENT",
        "FORCED_SYNTHESIS",
        "ABSTRACTION_CREEP",
        "EXAMPLE_TO_UNIVERSAL_LESSON",
        "LONG_FORM_OVERCOMPRESSION",
        "ABSTRACT_TEST_FRAMING",
        "ANALYTICAL_PACKAGING",
        "ACTORLESS_CAUSALITY",
        "ARMCHAIR_GENERALIZATION",
    }
    for code in review_structure(text, mode=structural_mode):
        issues.append(
            _issue(
                code,
                "Semantic structure review found a sentence or paragraph without a distinct job.",
                severity="error" if code in blocking_structure else "warning",
                action="delete" if code not in {"ACTORLESS_CAUSALITY"} else "ground",
            )
        )

    if _ANNOUNCEMENT_BEFORE_CONTENT.search(text):
        issues.append(
            _issue(
                "ANNOUNCEMENT_BEFORE_CONTENT",
                "The draft announces the thought before stating it.",
                action="shorten",
            )
        )
    if _IMPORTANCE_ANNOUNCEMENT.search(text):
        issues.append(
            _issue(
                "IMPORTANCE_ANNOUNCEMENT",
                "The draft advertises importance instead of adding evidence.",
                action="shorten",
            )
        )
    if _INSIGHT_PREVIEW.search(text):
        issues.append(
            _issue(
                "INSIGHT_PREVIEW",
                "The draft previews a reveal instead of giving the point.",
                action="flatten",
            )
        )
    if _FALSE_AGENCY.search(text):
        issues.extend(
            (
                _issue(
                    "FALSE_AGENCY",
                    "An abstract subject is given unsupported human agency.",
                    action="ground",
                ),
                _issue(
                    "HIDDEN_ACTOR",
                    "The wording hides the supported actor or mechanism.",
                    severity="warning",
                    action="ground",
                ),
                _issue(
                    "ACTORLESS_CAUSALITY",
                    "The sentence assigns causality without naming the mechanism.",
                    severity="warning",
                    action="ground",
                ),
            )
        )
    if _UNASKED_REASSURANCE.search(text):
        issues.extend(
            (
                _issue(
                    "UNASKED_REASSURANCE",
                    "The draft reassures the reader without a support task.",
                    action="shorten",
                ),
                _issue(
                    "PERMISSION_GRANTING_ENDING",
                    "The ending grants emotional permission that the task did not request.",
                    severity="warning",
                    action="shorten",
                ),
                _issue(
                    "EMOTIONAL_HAND_HOLDING",
                    "The prose tells the reader how to feel instead of advancing the thought.",
                    severity="warning",
                    action="shorten",
                ),
            )
        )
    if _READER_HAND_HOLDING.search(text):
        issues.append(
            _issue(
                "READER_HAND_HOLDING",
                "The draft tells the reader what to notice or conclude.",
                severity="warning",
                action="shorten",
            )
        )
    if _DISEMBODIED_EXPLANATION.search(text):
        issues.append(
            _issue(
                "DISEMBODIED_EXPLANATION",
                "The explanation starts from an abstract 'this' instead of the event or mechanism.",
                severity="warning",
                action="ground",
            )
        )
    if len(_REDUNDANT_MODIFIER.findall(text)) >= 2:
        issues.append(
            _issue(
                "REDUNDANT_MODIFIER",
                "Several modifiers intensify tone without adding meaning.",
                severity="warning",
                action="shorten",
            )
        )

    if not text or len(_content_tokens(text)) < 4 or _NON_DRAFT_SENTINEL.fullmatch(text):
        issues.append(
            _issue(
                "EMPTY_THESIS",
                "The provider returned no publishable observation.",
                action="regenerate_angle",
            )
        )

    for index, sentence in enumerate(sentences):
        concrete = _sentence_has_concrete_marker(sentence, evidence_tokens)
        for code, pattern, severity, action in _PATTERN_RULES:
            if pattern.search(sentence):
                contextual_severity = severity
                if (
                    severity == "warning"
                    and not concrete
                    and code in {"REVERSAL_HOOK", "MANUFACTURED_CONTRAST", "CONDITIONAL_ESCAPE"}
                ):
                    contextual_severity = "error"
                issues.append(
                    _issue(
                        code,
                        f"High-risk rhetorical pattern: {sentence}",
                        severity=contextual_severity,
                        sentence_index=index,
                        action=action,
                    )
                )
        if _TRANSITIONS.search(sentence):
            issues.append(
                _issue(
                    "EXCESSIVE_TRANSITIONS",
                    "The sentence uses a formal transition that is usually unnecessary "
                    "in a short post.",
                    severity="warning",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if len(_PARTICIPIAL.findall(sentence)) >= 2:
            issues.append(
                _issue(
                    "PARTICIPIAL_CLAUSE_OVERLOAD",
                    "The sentence stacks participial clauses.",
                    severity="warning",
                    sentence_index=index,
                    action="split",
                )
            )
        if _NOMINALIZATION.search(sentence):
            issues.append(
                _issue(
                    "NOMINALIZATION_OVERLOAD",
                    "The sentence uses noun-heavy phrasing where a verb would be clearer.",
                    severity="warning",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if sentence.count("—") >= 2:
            issues.append(
                _issue(
                    "EM_DASH_OVERUSE",
                    "The sentence overuses em dashes.",
                    severity="warning",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if sentence.count(",") >= 4 or re.search(
            r"\b\w+,\s+\w+(?:,\s+|\s+and\s+)\w+\b", sentence, re.I
        ):
            issues.append(
                _issue(
                    "MECHANICAL_TRICOLON",
                    "The sentence uses a polished list-of-three rhythm.",
                    severity="warning",
                    sentence_index=index,
                    action="shorten",
                )
            )
        if len(_ABSTRACT_NOUN.findall(sentence)) >= 3 and not concrete:
            issues.append(
                _issue(
                    "ABSTRACT_NOUN_CLUSTER",
                    "The sentence clusters abstractions without a concrete mechanism.",
                    severity="error",
                    sentence_index=index,
                    action="ground",
                )
            )
        if len(_LEXICAL_WATCH.findall(sentence)) >= 2:
            issues.append(
                _issue(
                    "LEXICAL_MARKER_CLUSTER",
                    "The sentence clusters high-alert promotional vocabulary.",
                    severity="warning",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if sentence.endswith("?") and (
            not concrete or _GENERIC_RHETORICAL_QUESTION.search(sentence)
        ):
            issues.append(
                _issue(
                    "RHETORICAL_QUESTION_BAIT",
                    "The question is not grounded in a specific evidence item.",
                    severity="warning",
                    sentence_index=index,
                    action="delete",
                )
            )
        if re.fullmatch(r"(?:[A-Za-z]+[.!]?\s*){1,3}", sentence) and len(sentences) >= 3:
            issues.append(
                _issue(
                    "PERFORMATIVE_FRAGMENTS",
                    "Short fragments manufacture intensity without adding information.",
                    severity="warning",
                    sentence_index=index,
                    action="flatten",
                )
            )

    paragraphs = _paragraphs(text)
    reply_mode = _is_reply(evidence_packet)
    humorous_reply = _source_specific_humorous_reply(
        variant,
        text=text,
        evidence_packet=evidence_packet,
        evidence_tokens=evidence_tokens,
        reply_mode=reply_mode,
    )
    if reply_mode and _HARMFUL_JOKE_ASSERTION.search(text):
        source_text = " ".join(_evidence_texts(evidence_packet))
        if not _HARMFUL_JOKE_ASSERTION.search(source_text):
            issues.append(
                _issue(
                    "UNSUPPORTED_FACT",
                    "A joke-shaped reply introduces a believable harmful factual accusation.",
                    action="ground",
                )
            )
    reply_naturalness = (
        inspect_reply_naturalness(
            variant, evidence_packet=evidence_packet, recent_posts=recent_posts
        )
        if reply_mode
        else None
    )
    if reply_naturalness is not None:
        if (
            reply_naturalness.humor_safety is not None
            and not reply_naturalness.humor_safety.suitable_for_humor
        ):
            categories = ", ".join(reply_naturalness.humor_safety.sensitive_categories)
            issues.append(
                _issue(
                    "HUMOR_ON_TRAGEDY",
                    (
                        "The reply uses humor on a sensitive harmful event"
                        + (f" ({categories})." if categories else ".")
                    ),
                    severity="error",
                    action="delete",
                )
            )
        if reply_naturalness.humor_intent is not None:
            for humor_issue in reply_naturalness.humor_intent.issue_codes:
                issues.append(
                    _issue(
                        humor_issue,
                        (
                            "Humor intent is unresolved for a sensitive source; "
                            "final text must fail closed."
                        ),
                        severity="error",
                        action="delete",
                    )
                )
        if (
            reply_naturalness.humor_intent is not None
            and reply_naturalness.humor_intent.extraction_conflict
        ):
            issues.append(
                _issue(
                    "REPLY_TYPE_MISMATCH",
                    (
                        "Reply metadata and final-text humor intent disagree; "
                        "final text controls safety."
                    ),
                    severity="warning",
                    action="flatten",
                )
            )
        if reply_naturalness.punchline_compulsion:
            issues.append(
                _issue(
                    "PUNCHLINE_COMPULSION",
                    (
                        "The reply forces a joke shell instead of using the least "
                        "constructed response."
                    ),
                    severity="warning",
                    action="flatten",
                )
            )
        if reply_naturalness.quote_card_risk:
            issues.append(
                _issue(
                    "QUOTE_CARD_REPLY",
                    (
                        "The reply is shaped like a reusable standalone aphorism rather than "
                        "a contextual response."
                    ),
                    severity="warning",
                    action="flatten",
                )
            )
        if reply_naturalness.recent_syntax_reuse:
            issues.append(
                _issue(
                    "RECENT_SYNTAX_REUSE",
                    (
                        "The reply reuses a recent sentence skeleton or rhetorical family "
                        "with different nouns."
                    ),
                    severity="warning",
                    action="flatten",
                )
            )
        if reply_naturalness.false_brevity:
            issues.append(
                _issue(
                    "FALSE_BREVITY",
                    "Compression leaves an incomplete or unnaturally clipped response.",
                    severity="error",
                    action="ground",
                )
            )
        if reply_naturalness.constructedness == "high" and (
            reply_naturalness.quote_card_risk
            or reply_naturalness.recent_syntax_reuse
            or not reply_naturalness.source_specific
        ):
            issues.append(
                _issue(
                    "ONE_LINER_FACTORY",
                    (
                        "The line uses a constructed reusable one-liner shell instead of an "
                        "ordinary source-specific reaction."
                    ),
                    severity="warning",
                    action="flatten",
                )
            )
    source_quality_angle = variant.angle_type == "source_quality"
    standalone_mode = not reply_mode and len(text) <= 1200
    if standalone_mode:
        if _STANDALONE_ABSTRACT_TEST.search(text):
            issues.extend(
                (
                    _issue(
                        "ABSTRACT_TEST_FRAMING",
                        (
                            "The standalone post describes the event as an abstract test "
                            "instead of naming what happens."
                        ),
                        action="ground",
                    ),
                    _issue(
                        "ANALYTICAL_PACKAGING",
                        "A direct observation is packaged as an analytical construct.",
                        action="flatten",
                    ),
                )
            )
        if _STANDALONE_MANUFACTURED_QUESTION.search(text):
            issues.append(
                _issue(
                    "MANUFACTURED_QUESTION",
                    "The post stages a thesis as a practical question instead of stating it.",
                    action="flatten",
                )
            )
        if _STANDALONE_NOT_ONLY_REVEAL.search(text):
            issues.extend(
                (
                    _issue(
                        "NOT_ONLY_BUT_REVEAL",
                        (
                            "The contrast manufactures a deeper reveal instead of stating "
                            "the supported distinction."
                        ),
                        action="flatten",
                    ),
                    _issue(
                        "STAGED_THESIS",
                        "The post is arranged as setup, reframing, and reveal.",
                        action="shorten",
                    ),
                )
            )
        if _STANDALONE_PRODUCT_MEMO.search(text):
            issues.append(
                _issue(
                    "PRODUCT_MEMO_VOICE",
                    "The standalone post reads like detached product or strategy prose.",
                    severity="warning",
                    action="flatten",
                )
            )
        if _STANDALONE_ABSTRACT_PREDICATE.search(text):
            issues.append(
                _issue(
                    "ABSTRACT_PREDICATE",
                    "A vague relational predicate replaces the observable action or result.",
                    severity="warning",
                    action="ground",
                )
            )
        if len(_ABSTRACT_NOUN.findall(text)) >= 5:
            issues.append(
                _issue(
                    "ABSTRACT_NOUN_DENSITY",
                    (
                        "Several conceptual nouns carry the post instead of concrete actors "
                        "and actions."
                    ),
                    action="ground",
                )
            )
        if len(sentences) >= 4:
            issues.append(
                _issue(
                    "MINI_ESSAY_ARC",
                    "The short standalone post expands into a miniature essay arc.",
                    severity="warning",
                    action="shorten",
                )
            )
        if _STANDALONE_VAGUE_USER.search(text) and evidence_packet is not None:
            issues.append(
                _issue(
                    "VAGUE_USER_PROXY",
                    "A generic user or market proxy replaces a more specific supported actor.",
                    severity="warning",
                    action="ground",
                )
            )

    if len(paragraphs) > 1:
        for index, paragraph in enumerate(paragraphs):
            paragraph_tokens = _content_tokens(paragraph)
            if len(paragraph_tokens) < 4:
                issues.append(
                    _issue(
                        "PARAGRAPH_WITHOUT_JOB",
                        (
                            "A paragraph does not add a separate fact, mechanism, example, "
                            "caveat, consequence, decision, or action."
                        ),
                        severity="warning",
                        sentence_index=index,
                        action="delete",
                    )
                )
        for left, right in pairwise(paragraphs):
            if _similarity(left, right) >= 0.78:
                issues.append(
                    _issue(
                        "PARAGRAPH_RESTATEMENT",
                        "Adjacent paragraphs substantially repeat the same informational content.",
                        action="delete",
                    )
                )
                break
        if len(paragraphs) >= 3 and all(len(_last_sentence(p).split()) <= 12 for p in paragraphs):
            issues.append(
                _issue(
                    "PARAGRAPH_THESIS_LOOP",
                    "Several paragraphs end by converting their content into a compact thesis.",
                    severity="warning",
                    action="flatten",
                )
            )
    vague_report_lead = bool(_VAGUE_REPORT_LEAD.search(text))
    accuracy_disclaimer = bool(_ACCURACY_DISCLAIMER.search(text))
    report_process_language = bool(_REPORT_PROCESS_LANGUAGE.search(text))

    if vague_report_lead:
        issues.append(
            _issue(
                "VAGUE_ATTRIBUTION",
                "The post uses a report or account as a vague unnamed claimant.",
                severity="error",
                action="attribute",
            )
        )
        issues.append(
            _issue(
                "MEDIA_BRAND_VOICE",
                "The post opens like a news bulletin instead of a personal observation.",
                severity="error",
                action="flatten",
            )
        )

    if accuracy_disclaimer:
        issues.append(
            _issue(
                "GENERIC_VERIFICATION",
                "The post uses an accuracy disclaimer instead of a bounded claim or no-post state.",
                severity="error",
                action="ground",
            )
        )
        issues.append(
            _issue(
                "CONDITIONAL_ESCAPE",
                "The post begins its contribution with a conditional verification escape.",
                severity="error",
                action="ground",
            )
        )

    if report_process_language and not source_quality_angle:
        issues.append(
            _issue(
                "EXPLANATION_LEAKAGE",
                "The publishable copy discusses the source-review process instead of the topic.",
                severity="error",
                action="delete",
            )
        )
        issues.append(
            _issue(
                "MEDIA_BRAND_VOICE",
                "The post evaluates what a report contains instead of sounding like one person.",
                severity="error",
                action="flatten",
            )
        )

    if (
        vague_report_lead or accuracy_disclaimer or report_process_language
    ) and not source_quality_angle:
        issues.append(
            _issue(
                "SOURCE_SUMMARY_ONLY",
                "The draft turns source handling into the post instead of adding a supported lens.",
                severity="error",
                action="regenerate_angle",
            )
        )

    for index, sentence in enumerate(sentences):
        concrete = _sentence_has_concrete_marker(sentence, evidence_tokens)
        superlative_match = _UNSUPPORTED_SUPERLATIVE.search(sentence)
        if superlative_match and not _ranking_supported(sentence, evidence_packet):
            rhetorical_only = bool(
                re.search(r"\b(?:real|true|main)\b", superlative_match.group(0), re.I)
                and not re.search(
                    r"\b(?:risk|problem|cost|only|most|biggest|hardest)\b",
                    superlative_match.group(0),
                    re.I,
                )
            )
            issues.append(
                _issue(
                    "UNSUPPORTED_SUPERLATIVE",
                    "The sentence ranks an object without comparative evidence.",
                    severity="warning" if rhetorical_only else "error",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if _AWARENESS_GAP.search(sentence) and not _awareness_gap_supported(
            sentence, evidence_packet
        ):
            issues.append(
                _issue(
                    "IMAGINARY_AWARENESS_GAP",
                    "The draft gains authority by claiming a broad group does not "
                    "understand the point.",
                    severity="error",
                    sentence_index=index,
                    action="delete",
                )
            )
        if _unsupported_group_claim(sentence, evidence_packet):
            issues.extend(
                (
                    _issue(
                        "UNSUPPORTED_GROUP_GENERALIZATION",
                        "The draft generalizes beyond the supplied evidence.",
                        severity="error",
                        sentence_index=index,
                        action="ground",
                    ),
                    _issue(
                        "ARMCHAIR_GENERALIZATION",
                        "The draft narrates a broad group from outside the supported scene.",
                        severity="warning",
                        sentence_index=index,
                        action="ground",
                    ),
                    _issue(
                        "NARRATOR_DISTANCE",
                        "The writer floats above the scene instead of naming a scoped actor.",
                        severity="warning",
                        sentence_index=index,
                        action="ground",
                    ),
                )
            )
        if _UNSUPPORTED_STAKES.search(sentence) and not _stakes_are_grounded(
            sentence, evidence_packet
        ):
            issues.append(
                _issue(
                    "UNSUPPORTED_STAKES",
                    "The sentence increases severity without naming a supported mechanism.",
                    severity="error",
                    sentence_index=index,
                    action="ground",
                )
            )
        if _INSIGHT_PACKAGING.search(sentence) or _GRAND_IDENTITY.search(sentence):
            issues.append(
                _issue(
                    "INSIGHT_PACKAGING",
                    "The sentence turns a concrete artifact into a grand conceptual identity.",
                    severity="error",
                    sentence_index=index,
                    action="flatten",
                )
            )
        if _UNIVERSAL_PRESCRIPTION.search(sentence):
            issues.append(
                _issue(
                    "UNSUPPORTED_INFERENCE",
                    "The draft turns a local observation into a universal professional rule.",
                    severity="error",
                    sentence_index=index,
                    action="ground",
                )
            )

    if _HISTORICAL_CONTRAST.search(text) and not _historical_contrast_supported(
        text, evidence_packet
    ):
        issues.append(
            _issue(
                "HISTORICAL_CONTRAST_FRAME",
                "The draft manufactures a history-versus-present arc that is not "
                "established by evidence.",
                severity="error",
                action="delete",
            )
        )

    metaphor_count = _metaphor_count(text)
    slogan_after_metaphor = metaphor_count >= 1 and _SLOGAN_ENDING.search(_last_sentence(text))
    if metaphor_count >= 2 or slogan_after_metaphor:
        issues.append(
            _issue(
                "METAPHOR_STACKING",
                "The short draft moves through multiple figurative frames.",
                severity="error",
                action="flatten",
            )
        )
    elif metaphor_count == 1 and reply_mode and not humorous_reply:
        issues.append(
            _issue(
                "UNNECESSARY_METAPHOR",
                "Reply mode should prefer a literal statement when it is equally clear.",
                severity="warning",
                action="flatten",
            )
        )

    if any(issue.code == "MECHANICAL_TRICOLON" for issue in issues):
        issues.append(
            _issue(
                "MECHANICAL_TRIAD",
                "A three-part list appears to exist mainly for completeness.",
                severity="warning",
                action="shorten",
            )
        )
    if (
        reply_mode
        and len(sentences) >= 2
        and variant.angle_type in _HUMOR_REPLY_ANGLE_TYPES
        and len(_content_tokens(sentences[-1])) >= 4
    ):
        issues.append(
            _issue(
                "EXPLANATION_AFTER_PUNCHLINE",
                "The reply continues after the source-specific punchline.",
                severity="warning",
                sentence_index=len(sentences) - 1,
                action="shorten",
            )
        )
    if len(sentences) >= 3:
        sentence_lengths = [len(_content_tokens(sentence)) for sentence in sentences]
        if max(sentence_lengths) - min(sentence_lengths) <= 2:
            issues.append(
                _issue(
                    "METRONOMIC_RHYTHM",
                    "Several consecutive sentences repeat nearly identical length.",
                    severity="warning",
                    action="flatten",
                )
            )

    if _paragraphs_are_redundant(paragraphs):
        issues.extend(
            (
                _issue(
                    "REDUNDANT_PARAGRAPH",
                    "A later paragraph restates the same contribution with more abstraction.",
                    severity="error" if reply_mode else "warning",
                    action="delete",
                ),
                _issue(
                    "SENTENCE_WITHOUT_JOB",
                    "A restatement adds no separate fact, reaction, caveat, or action.",
                    severity="warning",
                    action="delete",
                ),
                _issue(
                    "CUTTABILITY_FAILURE",
                    "The draft retains prose that can be removed without losing the contribution.",
                    severity="warning",
                    action="delete",
                ),
            )
        )
    if _SLOGAN_ENDING.search(_last_sentence(text)):
        issues.append(
            _issue(
                "FORMULAIC_PARAGRAPH_ENDING",
                "The draft ends with a packaged verdict instead of the last necessary detail.",
                severity="warning",
                action="shorten",
            )
        )

    if _concrete_to_abstract_drift(sentences):
        issues.append(
            _issue(
                "CONCRETE_TO_ABSTRACT_DRIFT",
                "The draft moves from concrete artifacts to increasingly broad conceptual labels.",
                severity="warning",
                action="flatten",
            )
        )
        if sum(len(_ABSTRACT_NOUN.findall(sentence)) for sentence in sentences) >= 3:
            issues.append(
                _issue(
                    "ABSTRACTION_LADDER",
                    "The same object is repeatedly renamed at higher levels of abstraction.",
                    severity="warning",
                    action="shorten",
                )
            )

    concrete_artifact_count = len(_CONCRETE_MECHANISMS.findall(text))
    if concrete_artifact_count >= 3 and (
        len(_ABSTRACT_NOUN.findall(text)) >= 2 or _INSIGHT_PACKAGING.search(text)
    ):
        issues.append(
            _issue(
                "OVERCOMPRESSED_THEORY",
                "Several distinct artifacts are compressed into one broad theory.",
                severity="warning",
                action="shorten",
            )
        )

    setup_to_conclusion = bool(
        reply_mode
        and len(sentences) >= 2
        and (
            _HISTORICAL_CONTRAST.search(sentences[0])
            or any(
                code in {"REVERSAL_HOOK", "MANUFACTURED_CONTRAST"} and pattern.search(sentences[0])
                for code, pattern, _, _ in _PATTERN_RULES
            )
        )
        and (
            _UNIVERSAL_PRESCRIPTION.search(sentences[-1])
            or _SLOGAN_ENDING.search(sentences[-1])
            or _UNSUPPORTED_STAKES.search(sentences[-1])
        )
    )
    if (
        (len(_CAUSAL_CONNECTORS.findall(text)) >= 2 and len(sentences) >= 2)
        or (reply_mode and len(paragraphs) > 1 and metaphor_count >= 1)
        or setup_to_conclusion
    ):
        issues.append(
            _issue(
                "POLISHED_CAUSAL_ARC",
                "The draft completes a polished setup-to-warning arc beyond the contribution.",
                severity="warning",
                action="shorten",
            )
        )

    if (
        not humorous_reply
        and _has_casual_surface(text)
        and (
            metaphor_count > 0
            or _HISTORICAL_CONTRAST.search(text)
            or _INSIGHT_PACKAGING.search(text)
            or _GRAND_IDENTITY.search(text)
            or _UNSUPPORTED_SUPERLATIVE.search(text)
        )
    ):
        issues.append(
            _issue(
                "FALSE_CASUALNESS",
                "Casual surface markers sit on top of highly composed rhetoric.",
                severity="warning",
                action="flatten",
            )
        )

    if reply_mode:
        recommendation_count = len(re.findall(r"\b(?:should|must|need to|have to)\b", text, re.I))
        reply_word_count = len(_WORDS.findall(text))
        reply_overproduced = bool(
            len(sentences) > 2
            or len(paragraphs) > 1
            or len(text) > 280
            or reply_word_count > 35
            or setup_to_conclusion
            or recommendation_count >= 2
            or _UNIVERSAL_PRESCRIPTION.search(text)
            or _paragraphs_are_redundant(paragraphs)
        )
        if reply_overproduced:
            issues.append(
                _issue(
                    "REPLY_OVERPRODUCTION",
                    "The reply contains more setup, explanation, or conclusion than "
                    "its contribution needs.",
                    severity=(
                        "error"
                        if len(sentences) > 2
                        or len(paragraphs) > 1
                        or len(text) > 280
                        or reply_word_count > 50
                        else "warning"
                    ),
                    action="shorten",
                )
            )

    if re.search(
        r"\b(?:this|it) (?:is not|isn'?t)\b.{0,120}[.!?]\s*(?:this|it) is\b",
        text,
        re.I,
    ):
        issues.append(
            _issue(
                "MANUFACTURED_CONTRAST",
                "The draft builds its point from a staged not-X / is-Y contrast.",
                severity="warning",
                action="flatten",
            )
        )

    if len(sentences) >= 2:
        lengths = [len(_tokens(sentence)) for sentence in sentences]
        if max(lengths) - min(lengths) <= 2:
            issues.append(
                _issue(
                    "UNIFORM_SENTENCE_RHYTHM",
                    "Sentence lengths are mechanically uniform.",
                    severity="warning",
                    action="flatten",
                )
            )
        if all(
            re.match(r"^(?:the|this|it|you|companies|people)\b", sentence, re.I)
            for sentence in sentences[: min(3, len(sentences))]
        ):
            issues.append(
                _issue(
                    "PERFECT_PARAGRAPH_ARC",
                    "The post follows a conspicuously staged mini-essay structure.",
                    severity="warning",
                    action="shorten",
                )
            )
        for left, right in pairwise(sentences):
            if _similarity(left, right) >= 0.68:
                issues.append(
                    _issue(
                        "REPEATED_EXPLANATION",
                        "A sentence restates the previous sentence instead of adding information.",
                        severity="warning",
                        action="delete",
                    )
                )

    if len(sentences) >= 3 and all(len(sentence.split()) <= 4 for sentence in sentences[-3:]):
        issues.append(
            _issue(
                "PERFORMATIVE_FRAGMENTS",
                "The ending uses a sequence of performative fragments.",
                severity="warning",
                action="flatten",
            )
        )

    # Mechanical symmetry at sentence or adjacent-sentence level.
    if re.search(
        r"\b(?:you can|one is|the first is|x is|it is)\b.{1,90}"
        r"\b(?:you cannot|the other is|the second is|but)\b",
        text,
        re.I,
    ):
        issues.append(
            _issue(
                "MECHANICAL_SYMMETRY",
                "The draft relies on mirrored clauses for rhetorical neatness.",
                severity="warning",
                action="flatten",
            )
        )

    if _FIRST_PERSON.search(text) and not _has_supported_first_person(evidence_packet):
        issues.append(
            _issue(
                "UNSUPPORTED_FIRST_PERSON",
                "The draft invents first-person experience or ownership.",
                action="ground",
            )
        )

    available_ids: set[str] = set()
    evidence_overlap = 0
    evidence_score = 1.0
    if evidence_packet is not None:
        available_ids = {item.evidence_id for item in evidence_packet.items}
        unknown_ids = set(variant.evidence_ids) - available_ids
        if unknown_ids or not variant.evidence_ids:
            issues.append(
                _issue(
                    "UNSUPPORTED_FACT",
                    "The variant does not cite valid evidence IDs.",
                    action="ground",
                )
            )
            evidence_score = 0.0
        item_by_id = {item.evidence_id: item for item in evidence_packet.items}
        selected_items = [
            item_by_id[item_id] for item_id in variant.evidence_ids if item_id in item_by_id
        ]
        selected_text = " ".join(item.text for item in selected_items)
        selected_tokens = _content_tokens(selected_text)
        evidence_overlap = len(_expanded_content_tokens(text) & selected_tokens)
        for claim in variant.factual_claims:
            claim_tokens = _content_tokens(claim)
            if claim_tokens and not (claim_tokens & selected_tokens):
                issues.append(
                    _issue(
                        "UNSUPPORTED_FACT",
                        "A factual claim has no trace in its cited evidence.",
                        action="ground",
                    )
                )
        external_selected = [item for item in selected_items if item.source_type != "user_input"]
        nonfactual_reply_contribution = bool(
            reply_mode
            and variant.contribution_type
            in {
                "plain_observation",
                "direct_response",
                "specific_qualification",
                "genuine_question",
                "contextual_extension",
            }
        )
        if (
            external_selected
            and not variant.factual_claims
            and not variant.attributed_claims
            and not humorous_reply
            and not reply_mode
            and not nonfactual_reply_contribution
            and editorial_intent != "quote_reaction"
        ):
            issues.append(
                _issue(
                    "UNSUPPORTED_FACT",
                    "The variant did not declare its factual or attributed claim boundary.",
                    action="ground",
                )
            )
        risky = {"single_source", "unverified"}
        if (
            any(item.verification_status in risky for item in selected_items)
            and not variant.attributed_claims
            and not variant.uncertainty_markers
            and not humorous_reply
            and not reply_mode
            and not nonfactual_reply_contribution
            and editorial_intent != "quote_reaction"
            and not allow_single_source_fact_check_gate
        ):
            issues.append(
                _issue(
                    "UNSUPPORTED_CERTAINTY",
                    "Single-source or unverified evidence became an unattributed statement.",
                    action="attribute",
                )
            )
        if evidence_packet.mode.value == "news_claim" and not variant.attributed_claims:
            issues.append(
                _issue(
                    "VAGUE_ATTRIBUTION",
                    "News-claim mode requires an attributed claim.",
                    action="attribute",
                )
            )
            issues.append(
                _issue(
                    "UNSPECIFIC_UNCERTAINTY",
                    "News-claim mode requires explicit source attribution.",
                    action="attribute",
                )
            )
        if (
            reply_mode
            and len(sentences) >= 2
            and re.match(r"^(?:first|firstly|to start)\b", sentences[0], re.I)
            and any(
                code == "PLACEHOLDER_COMMENTARY" and pattern.search(" ".join(sentences[1:]))
                for code, pattern, _, _ in _PATTERN_RULES
            )
        ):
            issues.append(
                _issue(
                    "SOURCE_SUMMARY_ONLY",
                    "The reply recaps source context before adding generic commentary.",
                    action="delete",
                )
            )
        source_similarity = max(
            (_similarity(text, item.text) for item in selected_items), default=0.0
        )
        if source_similarity >= 0.88 and editorial_intent not in {
            "report_event",
            "rewrite_existing",
        }:
            issues.append(
                _issue(
                    "SOURCE_SUMMARY_ONLY",
                    "The variant is too close to a source and adds no distinct lens.",
                    action="regenerate_angle",
                )
            )
        if _topic_substitution_generic(text, evidence_packet.topic):
            issues.append(
                _issue(
                    "TOPIC_LABEL_EXPANSION",
                    "The draft expands a topic label into interchangeable commentary.",
                    action="regenerate_angle",
                )
            )

    concrete_count = sum(
        _sentence_has_concrete_marker(sentence, evidence_tokens) for sentence in sentences
    )
    thesis_score = min(1.0, 0.25 + 0.25 * min(concrete_count, 3))
    specificity_score = min(
        1.0, 0.30 + 0.20 * min(evidence_overlap, 3) + 0.25 * min(concrete_count, 3)
    )
    if sentences and concrete_count == 0 and editorial_intent != "explain_topic":
        issues.append(
            _issue(
                "GENERIC_LINE",
                "The draft contains no evidence-specific actor, action, mechanism, or consequence.",
                action="ground",
            )
        )
    if _ABSTRACT_NOUN.search(text) and concrete_count <= 1:
        specificity_score = max(0.0, specificity_score - 0.22)

    # Final-sentence deletion test: generic or low-evidence endings should be removed.
    if len(sentences) >= 2:
        final = sentences[-1]
        final_tokens = _content_tokens(final)
        final_evidence_overlap = len(final_tokens & evidence_tokens)
        prior_tokens = _content_tokens(" ".join(sentences[:-1]))
        final_is_rhetorical = bool(
            any(
                code in {"GENERIC_CONCLUSION", "QUOTABLE_ENDING", "FAKE_DEPTH"}
                and pattern.search(final)
                for code, pattern, _, _ in _PATTERN_RULES
            )
            or _SLOGAN_ENDING.search(final)
            or _UNIVERSAL_PRESCRIPTION.search(final)
            or _INSIGHT_PACKAGING.search(final)
            or _GRAND_IDENTITY.search(final)
            or (len(final_tokens) <= 9 and _ABSTRACT_NOUN.search(final))
        )
        reply_restatement = bool(
            reply_mode
            and final_evidence_overlap == 0
            and (
                _similarity(final, " ".join(sentences[:-1])) >= 0.45
                or (
                    final_tokens & prior_tokens
                    and not _sentence_has_concrete_marker(final, evidence_tokens)
                )
            )
        )
        if (final_evidence_overlap == 0 and final_is_rhetorical) or reply_restatement:
            issues.append(
                _issue(
                    "OVEREXPLAINED_ENDING",
                    "The final sentence adds rhetoric rather than a necessary fact or caveat.",
                    severity="error",
                    sentence_index=len(sentences) - 1,
                    action="delete",
                )
            )

    medium_fit_score = 1.0
    length = len(text)
    if minimum_characters is not None and length < minimum_characters:
        medium_fit_score = 0.0
        issues.append(
            _issue(
                "FORMAT_MISMATCH",
                "The draft is shorter than the selected format range.",
                action="ground",
            )
        )
    if maximum_characters is not None and length > maximum_characters:
        medium_fit_score = 0.0
        issues.append(
            _issue(
                "FORMAT_MISMATCH", "The draft exceeds the selected format range.", action="shorten"
            )
        )
    if maximum_characters is not None and maximum_characters <= 280 and len(sentences) > 3:
        issues.append(
            _issue(
                "EXHAUSTIVE_COMPLETENESS",
                "A short post tries to cover too many dimensions.",
                severity="warning",
                action="shorten",
            )
        )

    recent_similarity = max((_similarity(text, recent) for recent in recent_posts), default=0.0)
    if recent_similarity >= 0.88:
        issues.append(
            _issue(
                "RECENT_POST_SIMILARITY",
                "The draft is too similar to recent account output.",
                action="regenerate_angle",
            )
        )
    opening = _first_words(text)
    ending = _last_sentence(text)
    if opening and any(_first_words(recent) == opening for recent in recent_posts):
        issues.append(
            _issue(
                "RECENT_OPENING_SIMILARITY",
                "The opening repeats a recent account template.",
                severity="warning",
                action="flatten",
            )
        )
    opening_family = _reply_opening_family(text) if reply_mode else None
    if opening_family and any(
        _reply_opening_family(recent) == opening_family for recent in recent_posts[-5:]
    ):
        issues.append(
            _issue(
                "RECENT_OPENING_SIMILARITY",
                "The reply reuses an opening family from recent account posts.",
                severity="warning",
                action="flatten",
            )
        )
    if ending and any(
        _similarity(ending, _last_sentence(recent)) >= 0.90 for recent in recent_posts
    ):
        issues.append(
            _issue(
                "RECENT_ENDING_SIMILARITY",
                "The ending repeats a recent account template.",
                severity="warning",
                action="delete",
            )
        )

    if evidence_packet is not None:
        semantic = review_semantics(
            text,
            packet=evidence_packet,
            editorial_intent=editorial_intent,
            coverage=source_coverage,
        )
        semantic_decision = semantic.decision
        job_count = max(len(semantic.sentence_jobs), 1)
        unique_jobs = len({job.job for job in semantic.sentence_jobs})
        sentence_job_uniqueness = min(1.0, unique_jobs / job_count)
        repeated_jobs = sum(
            job.job in {"restate", "framing", "importance_announcement", "conclusion_only"}
            for job in semantic.sentence_jobs
        )
        semantic_repetition_ratio = min(1.0, repeated_jobs / job_count)
        supported_relation_ratio = min(
            1.0,
            sum(bool(job.evidence_ids) for job in semantic.sentence_jobs) / job_count,
        )
        abstract_restatement_ratio = min(
            1.0,
            sum(job.job == "restate" for job in semantic.sentence_jobs) / job_count,
        )
        rhetorical_scaffolding_ratio = min(
            1.0,
            sum(
                job.job in {"framing", "importance_announcement", "conclusion_only"}
                for job in semantic.sentence_jobs
            )
            / job_count,
        )
        paragraph_count = max(len(semantic.paragraph_jobs), 1)
        paragraph_advancement_ratio = min(
            1.0,
            sum(bool(job.new_information) for job in semantic.paragraph_jobs) / paragraph_count,
        )
        source_proposition_overlap = semantic.source_coverage_overlap
        semantic_errors = {
            "NO_GROUNDED_CONTRIBUTION",
            "ANALYTICAL_PARAPHRASE_LOOP",
            "INVENTED_COMPARISON_TARGET",
            "DEPERSONALIZING_RENAME",
            "LEXICAL_NOVELTY_ONLY",
            "UNSUPPORTED_RELATION",
            "SEMANTIC_SOURCE_ECHO",
            "PARAGRAPH_JOB_DUPLICATION",
        }
        for code in semantic.issue_codes:
            issues.append(
                _issue(
                    code,
                    "Independent semantic review rejected this proposition or paragraph job.",
                    severity="error" if code in semantic_errors else "warning",
                    action=(
                        "return_insufficient_context"
                        if code in {"NO_GROUNDED_CONTRIBUTION", "SEMANTIC_SOURCE_ECHO"}
                        else "delete"
                    ),
                )
            )

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    naturalness_score = max(0.0, 1.0 - error_count * 0.10 - warning_count * 0.03)
    voice_score = max(
        0.0,
        1.0
        - sum(
            issue.code
            in {
                "CORPORATE_VOICE",
                "CREATOR_COACH_VOICE",
                "MEDIA_BRAND_VOICE",
                "FAKE_CONVERSATIONALITY",
                "FAKE_ROUGHNESS",
                "FALSE_CASUALNESS",
                "SOURCE_AS_SUBJECT",
                "MULTI_SOURCE_COLLAGE",
                "MISSING_SOURCE_CONTEXT",
            }
            for issue in issues
        )
        * 0.16,
    )
    rhetorical_devices = _rhetorical_device_count(issues)
    rhetorical_density_score = min(1.0, rhetorical_devices / max(2, len(sentences) + 1))
    plainness_penalty_codes = {
        "UNSUPPORTED_SUPERLATIVE",
        "IMAGINARY_AWARENESS_GAP",
        "METAPHOR_STACKING",
        "INSIGHT_PACKAGING",
        "HISTORICAL_CONTRAST_FRAME",
        "UNSUPPORTED_STAKES",
        "OVERCOMPRESSED_THEORY",
        "ABSTRACTION_LADDER",
        "CONCRETE_TO_ABSTRACT_DRIFT",
        "QUOTABLE_ENDING",
        "GENERIC_CONCLUSION",
        "MANUFACTURED_CONTRAST",
        "UNNECESSARY_METAPHOR",
        "UNSUPPORTED_GROUP_GENERALIZATION",
        "FALSE_CASUALNESS",
        "REPLY_OVERPRODUCTION",
        "PUNCHLINE_COMPULSION",
        "ONE_LINER_FACTORY",
        "RECENT_SYNTAX_REUSE",
        "QUOTE_CARD_REPLY",
        "FALSE_BREVITY",
        "REDUNDANT_PARAGRAPH",
        "POLISHED_CAUSAL_ARC",
        "VAGUE_ATTRIBUTION",
        "MEDIA_BRAND_VOICE",
        "GENERIC_VERIFICATION",
        "CONDITIONAL_ESCAPE",
        "EXPLANATION_LEAKAGE",
        "SOURCE_SUMMARY_ONLY",
        "SOURCE_AS_SUBJECT",
        "MULTI_SOURCE_COLLAGE",
        "MISSING_SOURCE_CONTEXT",
    }
    plainness_penalties = len({issue.code for issue in issues} & plainness_penalty_codes)
    plainness_score = max(
        0.0,
        1.0 - plainness_penalties * 0.10 - rhetorical_density_score * 0.35,
    )
    if reply_mode:
        reply_penalties = sum(
            issue.code
            in {
                "REPLY_OVERPRODUCTION",
                "PUNCHLINE_COMPULSION",
                "ONE_LINER_FACTORY",
                "RECENT_SYNTAX_REUSE",
                "QUOTE_CARD_REPLY",
                "FALSE_BREVITY",
                "HUMOR_ON_TRAGEDY",
                "HUMOR_INTENT_UNCERTAIN",
                "HUMOR_INTENT_CONFLICT",
                "REDUNDANT_PARAGRAPH",
                "OVEREXPLAINED_ENDING",
                "GENERIC_CONCLUSION",
                "QUOTABLE_ENDING",
                "METAPHOR_STACKING",
                "UNNECESSARY_METAPHOR",
                "POLISHED_CAUSAL_ARC",
            }
            for issue in issues
        )
        reply_word_count = len(_WORDS.findall(text))
        construction_penalty = 0.0
        if reply_naturalness is not None:
            construction_penalty = {"low": 0.0, "medium": 0.06, "high": 0.16}[
                reply_naturalness.constructedness
            ]
        reply_fit_score = max(
            0.0,
            1.0
            - reply_penalties * 0.12
            - max(0, len(sentences) - 2) * 0.20
            - max(0, len(paragraphs) - 1) * 0.25
            - max(0, reply_word_count - 35) / 45
            - max(0, 4 - reply_word_count) / 8
            - construction_penalty,
        )
    else:
        reply_fit_score = 1.0

    if specificity_score < minimum_specificity and editorial_intent != "explain_topic":
        issues.append(
            _issue(
                "GENERIC_LINE",
                "The draft does not meet the configured specificity threshold.",
                action="regenerate_angle",
            )
        )
    if evidence_score < minimum_evidence or any(
        issue.code in {"UNSUPPORTED_FACT", "UNSUPPORTED_CERTAINTY"} for issue in issues
    ):
        evidence_score = min(evidence_score, 0.55)
        issues.append(
            _issue(
                "UNSUPPORTED_FACT",
                "The draft does not meet the evidence-trace threshold.",
                action="ground",
            )
        )
    if naturalness_score < minimum_naturalness:
        issues.append(
            _issue(
                "FAKE_DEPTH",
                "The draft does not meet the configured naturalness threshold.",
                action="flatten",
            )
        )

    deduped = tuple(
        {(issue.code, issue.message, issue.sentence_index): issue for issue in issues}.values()
    )
    passed = not any(issue.severity == "error" for issue in deduped)
    return QualityReport(
        passed=passed,
        thesis_score=max(0.0, min(thesis_score, 1.0)),
        specificity_score=max(0.0, min(specificity_score, 1.0)),
        evidence_score=max(0.0, min(evidence_score, 1.0)),
        naturalness_score=max(0.0, min(naturalness_score, 1.0)),
        medium_fit_score=medium_fit_score,
        voice_score=voice_score,
        plainness_score=max(0.0, min(plainness_score, 1.0)),
        rhetorical_density_score=max(0.0, min(rhetorical_density_score, 1.0)),
        reply_fit_score=max(0.0, min(reply_fit_score, 1.0)),
        distinctiveness_score=max(0.0, 1.0 - recent_similarity),
        recent_similarity_score=recent_similarity,
        sentence_job_uniqueness=sentence_job_uniqueness,
        semantic_repetition_ratio=semantic_repetition_ratio,
        supported_relation_ratio=supported_relation_ratio,
        abstract_restatement_ratio=abstract_restatement_ratio,
        rhetorical_scaffolding_ratio=rhetorical_scaffolding_ratio,
        paragraph_advancement_ratio=paragraph_advancement_ratio,
        source_proposition_overlap=source_proposition_overlap,
        semantic_decision=semantic_decision,
        issues=deduped,
    )


def assess_result(
    result: GenerationResult,
    request: GenerationRequest,
    *,
    minimum_specificity: float = 0.55,
    minimum_evidence: float = 0.75,
    minimum_naturalness: float = 0.55,
    maximum_recent_similarity: float = 0.88,
    require_distinct_angles: bool = True,
) -> tuple[QualityReport, ...]:
    """Assess all variants and enforce angle, semantic, and corpus distinctiveness."""

    reports = [
        assess_variant(
            variant,
            evidence_packet=request.evidence_packet,
            minimum_specificity=minimum_specificity,
            minimum_evidence=minimum_evidence,
            minimum_naturalness=minimum_naturalness,
            minimum_characters=request.minimum_characters,
            maximum_characters=request.maximum_characters,
            recent_posts=request.recent_posts,
            allow_single_source_fact_check_gate=(
                request.metadata.get("single_source_anchor") is True
                and request.generation_mode.value != "news_claim"
            ),
            editorial_intent=request.editorial_intent,
            source_coverage=request.source_coverage,
            style_mode=request.content_type.value,
        )
        for variant in result.variants
    ]
    pair_similarity: dict[int, float] = {index: 0.0 for index in range(len(result.variants))}
    duplicate_pair = False
    for left_index, left in enumerate(result.variants):
        for right_index in range(left_index + 1, len(result.variants)):
            right = result.variants[right_index]
            similarity = _similarity(left.text, right.text)
            pair_similarity[left_index] = max(pair_similarity[left_index], similarity)
            pair_similarity[right_index] = max(pair_similarity[right_index], similarity)
            if similarity >= maximum_recent_similarity:
                duplicate_pair = True

    angle_ids = [variant.angle_id for variant in result.variants]
    angle_types = [variant.angle_type for variant in result.variants]
    distinct_angle_ids = len(angle_ids) == len(set(angle_ids))
    distinct_angle_types = len(result.variants) < 2 or len(set(angle_types)) >= 2
    candidate_by_id = {angle.angle_id: angle for angle in request.angle_candidates}
    revision_target_by_id = {target.variant.angle_id: target for target in request.revision_targets}
    type_counts = Counter(angle_types)
    final: list[QualityReport] = []
    for index, report in enumerate(reports):
        issues = list(report.issues)
        variant = result.variants[index]
        candidate = candidate_by_id.get(variant.angle_id)
        if require_distinct_angles and (
            duplicate_pair or not distinct_angle_ids or not distinct_angle_types
        ):
            issues.append(
                _issue(
                    "VARIANT_PARAPHRASE",
                    "Multiple variants must express distinct supported contributions.",
                    action="regenerate_angle",
                )
            )
        if type_counts[variant.angle_type] == 3:
            issues.append(
                _issue(
                    "RECENT_ANGLE_SIMILARITY",
                    "All variants use the same intellectual move.",
                    severity="warning",
                    action="regenerate_angle",
                )
            )
        if candidate is None:
            issues.append(
                _issue(
                    "UNSUPPORTED_ANGLE",
                    "The variant uses an angle not supplied by the pipeline.",
                    action="regenerate_angle",
                )
            )
        else:
            if variant.angle_type != candidate.angle_type:
                issues.append(
                    _issue(
                        "UNSUPPORTED_ANGLE",
                        "The variant changed the selected angle type.",
                        action="regenerate_angle",
                    )
                )
            if not set(variant.evidence_ids).issubset(set(candidate.evidence_ids)):
                issues.append(
                    _issue(
                        "UNSUPPORTED_FACT",
                        "The variant cites evidence outside its selected angle.",
                        action="ground",
                    )
                )

        if request.metadata.get("single_source_anchor") is True:
            anchor_id = str(request.metadata.get("anchor_evidence_id") or "")
            cited_ids = tuple(dict.fromkeys((*variant.evidence_ids, *variant.source_post_ids)))
            if (
                len(set(variant.evidence_ids)) != 1
                or (anchor_id and set(variant.evidence_ids) != {anchor_id})
                or any(source_id != anchor_id for source_id in variant.source_post_ids if anchor_id)
                or len(set(source_id for source_id in cited_ids if source_id)) > 1
                or _COLLAGE_LANGUAGE.search(variant.text)
            ):
                issues.append(
                    _issue(
                        "MULTI_SOURCE_COLLAGE",
                        "A single-source draft combined or compared independent feed items.",
                        action="regenerate_angle",
                    )
                )
            username = str(request.metadata.get("anchor_author_username") or "").lstrip("@")
            handle_pattern = (
                re.compile(rf"(?<![A-Za-z0-9_])@{re.escape(username)}\b", re.I)
                if username
                else None
            )
            if request.editorial_intent not in {"report_event", "rewrite_existing"} and (
                _SOURCE_NARRATION.search(variant.text)
                or (handle_pattern is not None and handle_pattern.search(variant.text))
            ):
                issues.append(
                    _issue(
                        "SOURCE_AS_SUBJECT",
                        "The public copy narrates the source account instead of speaking directly.",
                        action="flatten",
                    )
                )
            strategy = str(request.metadata.get("source_context_strategy") or "standalone")
            if strategy in {"attach_source_media", "quote_post"} and re.search(
                r"\b(?:in (?:the|this) (?:clip|video|post|tweet)|the (?:clip|video|post|tweet) "
                r"(?:shows?|says?|captures?))\b",
                variant.text,
                re.I,
            ):
                issues.append(
                    _issue(
                        "MISSING_SOURCE_CONTEXT",
                        "The copy explains media context that should be carried by the "
                        "attachment or quote.",
                        action="shorten",
                    )
                )
        revision_target = revision_target_by_id.get(variant.angle_id)
        if request.revision_targets:
            if revision_target is None:
                issues.append(
                    _issue(
                        "UNSUPPORTED_ANGLE",
                        "The revision omitted or replaced a supplied source variant.",
                        action="regenerate_angle",
                    )
                )
            else:
                source_variant = revision_target.variant
                if variant.revision_status == "not_applicable":
                    issues.append(
                        _issue(
                            "FORMAT_MISMATCH",
                            "A humanizer revision must declare its revision status.",
                            action="ground",
                        )
                    )
                if variant.revision_status == "insufficient_context":
                    issues.append(
                        _issue(
                            "INSUFFICIENT_CONTEXT",
                            "The bounded revision cannot be completed without unsupported facts.",
                            action="return_insufficient_context",
                        )
                    )
                if (
                    variant.revision_status == "no_change_needed"
                    and variant.text != source_variant.text
                ):
                    issues.append(
                        _issue(
                            "FORMAT_MISMATCH",
                            "A no-change revision modified the source text.",
                            action="ground",
                        )
                    )
                if variant.factual_claims != source_variant.factual_claims:
                    issues.append(
                        _issue(
                            "UNSUPPORTED_FACT",
                            "The humanizer revision changed the factual-claim boundary.",
                            action="ground",
                        )
                    )
                if variant.attributed_claims != source_variant.attributed_claims:
                    issues.append(
                        _issue(
                            "VAGUE_ATTRIBUTION",
                            "The humanizer revision changed attribution.",
                            action="attribute",
                        )
                    )
                if variant.uncertainty_markers != source_variant.uncertainty_markers:
                    issues.append(
                        _issue(
                            "UNSUPPORTED_CERTAINTY",
                            "The humanizer revision changed uncertainty markers.",
                            action="attribute",
                        )
                    )
                source_claims = extract_claims(source_variant.text, prefix="HR-S")
                revised_claims = extract_claims(variant.text, prefix="HR-R")
                claim_changes = compare_claim_sets(source_claims, revised_claims)
                semantic_fields = {
                    "subject",
                    "predicate",
                    "object",
                    "polarity",
                    "quantities",
                    "dates",
                    "attribution_source",
                    "modality",
                }
                if source_claims and not revised_claims:
                    issues.append(
                        _issue(
                            "NO_GROUNDED_CONTRIBUTION",
                            "The humanizer removed the canonical contribution entirely.",
                            action="return_insufficient_context",
                        )
                    )
                elif any(
                    change.change_type == "changed"
                    and bool(set(change.fields_changed) & semantic_fields)
                    for change in claim_changes
                ):
                    issues.append(
                        _issue(
                            "UNSUPPORTED_CONTRIBUTION",
                            "The humanizer changed the canonical contribution while "
                            "retaining its IDs.",
                            action="ground",
                        )
                    )
                if set(_URL.findall(variant.text)) != set(_URL.findall(source_variant.text)):
                    issues.append(
                        _issue(
                            "UNSUPPORTED_FACT",
                            "The humanizer revision changed or removed a preserved link.",
                            action="ground",
                        )
                    )
        elif variant.revision_status != "not_applicable":
            issues.append(
                _issue(
                    "FORMAT_MISMATCH",
                    "Initial generation must use revision_status=not_applicable.",
                    action="ground",
                )
            )

        distinctiveness = max(
            0.0, 1.0 - max(pair_similarity[index], report.recent_similarity_score)
        )
        passed = report.passed and not any(issue.severity == "error" for issue in issues)
        final.append(
            report.model_copy(
                update={
                    "passed": passed,
                    "distinctiveness_score": distinctiveness,
                    "issues": tuple(issues),
                }
            )
        )
    return tuple(final)


def repair_feedback(reports: Iterable[QualityReport]) -> str:
    """Build a compact deletion-first contract from canonical issue codes only."""

    codes = sorted({issue.code for report in reports for issue in report.issues})
    rendered = ", ".join(codes) or "GENERIC_LINE"
    return (
        "Fix only the listed issues: "
        f"{rendered}. Preserve the supported factual claim, selected angle, attribution, "
        "uncertainty, evidence IDs, requested language, links, and technical meaning. First "
        "identify the supported core claim. For reply mode, first try deleting unnecessary "
        "setup, extra paragraphs, metaphors, repeated meaning, broad recommendations, and the "
        "final conclusion. Delete newsroom leads, vague report attribution, accuracy disclaimers, "
        "source-review commentary, unsupported hierarchy, awareness-gap claims, historical irony, "
        "inflated stakes, and slogan endings before changing wording. Keep usable "
        "concrete sentences unchanged. Rewrite only text made grammatically broken by deletion. "
        "Do not choose a new angle, add facts, strengthen certainty, invent group behavior, or "
        "replace deleted rhetoric with a hook, slogan, lesson, superlative, question, "
        "generalization, or new conclusion. For STANDALONE_CONTEXT_GAP add only the minimum "
        "source-supported actor, action, object, and significance needed to understand the post "
        "without opening the source; do not mention the post, report, caption, wording, or source "
        "account. For WEAK_ADDED_VALUE preserve the event context but replace generic paraphrase "
        "with one concrete bounded implication, mechanism, decision, or personal observation "
        "already licensed by the evidence. For SOURCE_ECHO keep the supported event but remove "
        "the source's sentence structure and wording; write one independent observation rather "
        "than a close paraphrase. For X_REPLY_MICRO, preserve the validated response function. "
        "Prefer the least constructed complete reply in 4-35 words, one sentence by default and "
        "two when a necessary qualification, genuine question, or concrete consequence requires "
        "it. Lowercase and humor are optional. Delete punchline compulsion, quote-card packaging, "
        "repeated syntax, and false brevity without changing the contribution. Set "
        "revision_status=no_change_needed when the source already "
        "passes unchanged. If a concrete revision requires unsupported facts, set "
        "revision_status=insufficient_context and do not improvise a replacement claim. Return "
        "one revision per supplied candidate, preserving the same angle IDs and evidence "
        "boundaries."
    )
