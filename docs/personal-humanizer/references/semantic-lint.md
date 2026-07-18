# Conditional Semantic Lint

Use this pass after the draft has a supported idea.

Its job is to remove predictable prose behavior without flattening voice, humor, technical accuracy, empathy, or natural variation.

## Semantic coverage boundary

A consumed linking verb does not prove that a human result state is understood. Bind the affected
referent, state or dependency complement, assertion state, causal context, and exact evidence. An
asserted untyped physiological or medical result remains `possible_harm`; ordinary location,
schedule, possession, or non-human dependency controls do not become medical harm by structure
alone.

## Principle

Do not ban a word or grammatical form merely because it often appears in generated prose.

Identify what the sentence, paragraph, or document is doing.

Revise a pattern only when its function makes the text less direct, less accurate, less specific, less trustworthy, or less natural.

## Issue registry

| Code | Trigger | Default repair |
|---|---|---|
| `ANNOUNCEMENT_BEFORE_CONTENT` | A sentence previews or advertises the next thought | delete the announcement and state the content |
| `IMPORTANCE_ANNOUNCEMENT` | The text claims a point is important, surprising, uncomfortable, or overlooked without evidence | remove the importance claim |
| `INSIGHT_PREVIEW` | A setup promises a reveal, lesson, explanation, or hidden meaning | begin with the actual point |
| `FALSE_AGENCY` | An abstract or inanimate subject receives inaccurate human intention, judgment, desire, or choice | name the actor or literal mechanism |
| `HIDDEN_ACTOR` | Wording conceals a known actor who matters to meaning or accountability | name the actor |
| `ACTORLESS_CAUSALITY` | The sentence claims change without a supported actor or mechanism | state the mechanism or narrow the claim |
| `NARRATOR_DISTANCE` | The writer lectures about people or society from a detached position | use a scoped actor, behavior, or situation |
| `ARMCHAIR_GENERALIZATION` | A broad claim about groups exceeds the evidence | narrow or remove |
| `DISEMBODIED_EXPLANATION` | Explanation hides who did what in the actual case | name the event, actor, or mechanism |
| `UNASKED_REASSURANCE` | The text comforts the reader although the task does not call for support | delete the reassurance |
| `PERMISSION_GRANTING_ENDING` | The ending grants permission or acceptance without need | delete it |
| `EMOTIONAL_HAND_HOLDING` | The prose tells readers how to feel instead of advancing the task | remove unless care is part of the genre |
| `SENTENCE_WITHOUT_JOB` | A sentence performs no unique informational, practical, or humorous job | delete it |
| `CUTTABILITY_FAILURE` | Removable modifiers, transitions, restatements, or completion lines remain | cut without replacement |
| `REDUNDANT_MODIFIER` | A modifier intensifies tone without changing meaning or certainty | remove it |
| `EXPLANATION_AFTER_PUNCHLINE` | Prose explains or moralizes after the joke lands | stop at the punchline |
| `PUNCHLINE_COMPULSION` | Reply prose forces a twist or punchline when a plain response would be more natural | remove the joke shell and keep the grounded response |
| `ONE_LINER_FACTORY` | A reusable polished sentence shell is substituted for source-specific reaction | rewrite as ordinary speech with the concrete source detail |
| `RECENT_SYNTAX_REUSE` | Recent account replies reuse the same sentence skeleton with different nouns | choose another communicative function or plainer syntax |
| `QUOTE_CARD_REPLY` | A contextual reply is optimized as a standalone aphorism or quote-card line | restore direct response to the source |
| `FALSE_BREVITY` | Compression removes a qualification or makes the response incomplete or unnatural | restore only the missing material thought |
| `HUMOR_ON_TRAGEDY` | Final text uses humorous treatment of deaths, fatality euphemisms, wounded or injured people, victims, medical emergencies, war casualties, abuse, self-harm, missing persons, or personal tragedy; metadata cannot suppress this check | remove the humorous candidate; retain a respectful non-humor response when grounded |
| `HUMOR_INTENT_UNCERTAIN` | Final text has a medium-confidence evaluative-irony signal on a sensitive source | hold or remove the humor candidate until live semantic inspection or human review; keep non-humor families eligible |
| `HUMOR_INTENT_CONFLICT` | Deterministic and provider-backed humor-intent inspections materially disagree on a sensitive source, or provider evidence spans do not match final text | fail closed for the candidate and keep final-text deterministic safety authoritative |
| `REPLY_TYPE_MISMATCH` | Provider metadata labels a reply as non-humor while final text has a callback, wordplay, incongruity, dry reframe, literalization, or joke-shaped template | classify from final text and run the applicable safety checks |
| `READER_HAND_HOLDING` | The writer tells readers what to notice or conclude when the text is clear | trust the reader |
| `METRONOMIC_RHYTHM` | Consecutive sentences repeat nearly identical length and syntax | vary or merge only where natural |
| `FORMULAIC_PARAGRAPH_ENDING` | Paragraphs repeatedly end with slogans, verdicts, warnings, reassurance, or fragments | end on the last necessary detail |
| `MECHANICAL_TRIAD` | A three-item list exists mainly to sound complete | keep only necessary items |
| `ANALYTICAL_PACKAGING` | A direct observation is inflated into an analytical construct, framework, test, or thesis | recover and state the literal observation |
| `ABSTRACT_TEST_FRAMING` | An event is described as a “test” instead of saying what it will reveal | name the observable result |
| `STAGED_THESIS` | Sentences are arranged as setup, reframing, and reveal | begin with the actual point |
| `MANUFACTURED_QUESTION` | “The question is…” packages a statement rather than requests information | state the claim directly |
| `NOT_ONLY_BUT_REVEAL` | A contrast creates artificial depth by replacing X with a supposedly deeper Y | keep only the supported distinction |
| `ABSTRACT_NOUN_DENSITY` | Several conceptual nouns interact through weak predicates | use concrete actors, actions, and objects |
| `ABSTRACT_PREDICATE` | A vague relational predicate replaces an observable action or result | use the literal verb |
| `PRODUCT_MEMO_VOICE` | A short post reads like detached product strategy or research prose | rewrite as a direct observation |
| `MINI_ESSAY_ARC` | A short post follows setup → analysis → reveal → conclusion without need | retain only sentences with unique jobs |
| `VAGUE_USER_PROXY` | A generic actor replaces a supported specific group | name the supported actor |
| `PARAGRAPH_WITHOUT_JOB` | A paragraph adds no new fact, mechanism, example, caveat, consequence, decision, or action | delete or merge it |
| `PARAGRAPH_RESTATEMENT` | A paragraph substantially repeats an adjacent paragraph | retain the clearer version |
| `RECAP_TRANSITION` | A paragraph begins by paraphrasing the prior paragraph before advancing | start with the new content |
| `TRANSITION_SCAFFOLDING` | A transition simulates cohesion but carries no information | remove it |
| `REPEATED_PARAGRAPH_ARC` | Multiple paragraphs repeat the same rhetorical structure | vary structure by actual paragraph job |
| `PARAGRAPH_THESIS_LOOP` | Each paragraph ends by converting its content into a thesis or lesson | stop at the last fact or mechanism |
| `SYNTHETIC_COHESION` | Excessive signposting makes the document feel assembled rather than connected | let content create the connection |
| `CONCLUSION_RESTATEMENT` | The conclusion repeats the introduction or body without adding synthesis, decision, or action | delete or replace with required outcome |
| `FORCED_SYNTHESIS` | Separate observations are compressed into a grand unifying theory without support | keep them separate |
| `ABSTRACTION_CREEP` | The document moves toward larger conceptual nouns without new evidence | return to concrete actors and facts |
| `EXAMPLE_TO_UNIVERSAL_LESSON` | A single example becomes a broad rule or moral | state only the supported mechanism |
| `RHETORICAL_QUESTION_CHAIN` | Several questions replace explanation or evidence | answer directly or keep one necessary question |
| `UNIFORM_PARAGRAPH_CADENCE` | Paragraphs repeat the same visual and syntactic length pattern mechanically | vary only when the repetition is noticeable |
| `QUOTE_CARD_SENTENCE_CLUSTER` | Several sentences are independently polished as shareable aphorisms | use plain connective and explanatory prose |
| `LONG_FORM_OVERCOMPRESSION` | Necessary reasoning or evidence is compressed into slogans or fragments | restore the missing explanation |

### Reply-diversity issue contracts

These reply issues are emitted by runtime logic and exercised by
`evals/micro-reply-fixtures.json`.

| Code | Applies to | Severity guidance | Meaning | Repair |
|---|---|---|---|---|
| `PUNCHLINE_COMPULSION` | X_REPLY_MICRO, especially humor-labeled candidates | warning for a repairable style shell; block through semantic reaction inspection when no grounded reaction function exists | A twist, punchline, or joke marker is forced around a plainer supported response | Remove the joke shell and keep the least constructed grounded response |
| `ONE_LINER_FACTORY` | X_REPLY_MICRO | warning that lowers recommendation; block when paired with source echo or unsupported reaction semantics | A reusable polished sentence shell replaces source-specific reaction | Rewrite as ordinary speech using the concrete source detail or abstain |
| `RECENT_SYNTAX_REUSE` | X_REPLY_MICRO with recent account history | warning that lowers recommendation | Recent replies repeat a sentence skeleton, rhetorical family, metaphor family, fake-newsroom voice, setup-to-twist cadence, or sarcasm shape with different nouns | Choose another communicative function or use plainer syntax; do not diversify with random synonyms |
| `QUOTE_CARD_REPLY` | contextual replies | warning that lowers recommendation | A reply is optimized as a standalone aphorism rather than a response to the source | Restore a direct observation, response, qualification, or real question |
| `FALSE_BREVITY` | X_REPLY_MICRO | block when the compressed text is incomplete or loses a material qualification | Compression makes ordinary speech unnatural or changes the supported meaning | Restore only the missing material thought; do not expand into analysis |
| `HUMOR_ON_TRAGEDY` | X_REPLY_MICRO text that is humorous by metadata or final-text function | block and do not retry the humor family | The reaction may be factually grounded and source-specific but is unsuitable because it trivializes victims, deaths, fatality euphemisms, wounded or injured people, medical emergency, war casualties, abuse, self-harm, a missing person, or personal tragedy. Software, server, browser, market, crypto, and price crashes are disambiguated from human crashes. | Delete the humor candidate and consider a plain observation, direct response, qualification, or respectful question |
| `HUMOR_INTENT_UNCERTAIN` | X_REPLY_MICRO on a sensitive source | block pending live inspection or human review | Positive evaluative language may be ironic, but target or intent is not high-confidence | Keep respectful non-humor candidates available and do not infer safety from a plain provider label |
| `HUMOR_INTENT_CONFLICT` | X_REPLY_MICRO on a sensitive source | block the candidate | Provider and deterministic inspections materially disagree, or the provider did not cite exact final-text spans | Trust deterministic final-text safety, record the conflict, and retain non-humor candidates |
| `REPLY_TYPE_MISMATCH` | X_REPLY_MICRO | warning and trust-boundary audit signal | Provider metadata and final-text humor intent disagree | Keep the final-text classification authoritative and run humor safety when the text is humorous |

### Humor grounding versus humor suitability

A reaction function and factual support are necessary but not sufficient for humor. The runtime
checks humor suitability separately. Humor intent is inferred from the rendered text and its
relation to the source as well as metadata, so a provider label cannot disable the check. The
bounded evaluative inspection records evaluation polarity, evaluation relation, source-event
polarity, the evaluation target, target-source alignment, literal-reading plausibility, confidence,
and exact final-text spans. Praise, celebration, ratings, rewards, admiration, success attribution,
positive timing, and beneficiary language aimed at a grounded harmful, failed, contradictory, or
negative event can be implicit sarcasm without a registered joke template. Provider proposals must cite exact final-text spans; provider `none`
never erases deterministic humor, and material conflict or uncertainty on a sensitive source fails
closed for that candidate.

Human-harm scope is referent-aware and records harm state plus event type. Active, passive,
nominal, and result-state wording includes no-survivor and declared-deceased reports, fatal or
mortal injuries, people lost in a physical crash, people who did not make it home, coma and
life-support states, body-count and recovered-body reports, people who never made it out, lives
ended, critical-condition reports, and people found lifeless. Bounded technical, product, business,
political, and metaphorical relations such as dead code, a killed process, a build or pipeline that
died, a dead database or node, a model or launch that died, a bill killed in committee, a killed
proposal, a dead brand, or a network, chain, or token that died do not become human tragedy when the
non-human referent is clear. An unresolved death referent blocks only humorous treatment,
preserving respectful plain, direct, qualified, or question candidates. Safe technical event
language cannot suppress literal human harm elsewhere in the same source.

Recent-history comparison uses the relation operator and subject, complement, decoration,
repackaging marker, and opener slots. Connector or decoration changes such as `merely`, `glorified`,
`fancy`, `pretending to be`, `cosplaying as`, `rebranded`, `in disguise`, `a fresh coat of paint`,
`better branding`, `nothing but`, `equals ... plus ...`, or `strip away the branding` do not create a
fresh shell when the reply still performs the same reduction or repackaging job. Literal equations, definitions, and physical
clarifications remain outside the bounded reductive controls.

## Sentence-level checks

### Announcement before content

Flag a sentence when its main purpose is to tell the reader that the next sentence contains the important, uncomfortable, interesting, deeper, or overlooked point.

Examples are not a phrase blacklist.

Keep an introductory sentence when it supplies actual orientation.

### False agency and hidden actors

Do not require a human subject in every sentence.

Flag only when an abstract subject receives intention, judgment, choice, speech, learning, or responsibility that belongs to an actor or mechanism.

Do not flag literal technical, legal, or causal behavior:

- the API returned a 429
- the contract bars resale
- the model produced invalid JSON
- the price fell after the filing
- the report lists the affected accounts

Do not invent an actor when none is known. Passive voice may be more accurate.

### Narrator distance

Flag broad unsupported narration about people, teams, founders, users, markets, or society.

Do not automatically replace `people` with `you`.

Prefer a named actor, scoped group, observed behavior, or concrete situation.

### Unasked reassurance

Delete reassurance in informational, analytical, humorous, or transactional text.

Keep empathy where care is part of the communication job.

### Sentence jobs

Each sentence should perform at least one unique job:

- reaction
- joke
- fact
- attribution
- condition
- distinction
- mechanism
- caveat
- correction
- useful question
- necessary context
- requested action

Delete sentences whose only job is emphasis, transition, preview, restatement, closure, lesson, or emotional validation.

A punchline is a real job.

## Standalone-post checks

Use `x-post-contract.md`.

Flag packaging when a straightforward observation becomes:

- an abstract test
- a staged practical question
- a `not only X; it is Y` reveal
- a conceptual definition
- an abstract noun chain
- detached product-memo prose
- a complete miniature essay

Recover the literal core:

1. who acts?
2. what happens?
3. under what condition?
4. what object is involved?

If the draft is mostly a polished expansion of that core, use the core.

## Long-form and paragraph checks

Use `long-form-contract.md`.

### Paragraph job

Every paragraph must add a unique fact, mechanism, example, caveat, comparison, consequence, decision, or action.

### Transition test

Remove the transition phrase.

If the paragraph remains clear, the phrase may be scaffolding.

### Repetition test

Compare adjacent paragraphs semantically, not lexically.

Repeated terminology is acceptable. Repeated informational content is not.

### Paragraph-arc test

Look for repeated:

- setup → contrast → takeaway
- example → lesson
- question → verdict
- history → reversal → warning

One instance may be natural. Repetition across the document is the issue.

### Ending test

Do not make every paragraph end with a quotable sentence.

Plain explanatory endings are valid.

### Abstraction test

Track whether named actors and events are replaced over time by larger concepts.

Every increase in abstraction must contribute necessary synthesis.

### Conclusion test

The conclusion must add a decision, action, answer, or genuine synthesis.

If it only restates, delete it.

## Rhythm

Do not vary rhythm randomly.

Flag only noticeable mechanical repetition:

- three or more sentences with identical structure
- many equal one-sentence paragraphs
- every paragraph ending with a punchline
- automatic triads
- clusters of aphorisms

## No absolute grammar bans

Never apply these as universal rules:

- remove every adverb
- require active voice in every sentence
- require a human subject in every sentence
- ban every question
- ban every Wh-word opening
- ban every em dash
- reduce every list to two items
- ban every fragment
- reject every inanimate subject
- remove all analytical language
- force every paragraph to a different length
- require a conclusion

Forms are acceptable when they serve accuracy, voice, timing, structure, or care.

## Compact scoring

For short prose, score 0–2 on:

- directness
- specificity
- actor/mechanism clarity
- density
- medium fit

For long prose, also score:

- paragraph advancement
- transition quality
- structural variety
- abstraction control
- conclusion necessity

Revise only dimensions with a concrete failure.

Do not chase a perfect numerical score.

## Compositional extraction and coverage

The semantic route separates normalization, clause segmentation, lexical atoms, entity and
predicate candidates, argument binding, relation construction, coverage accounting,
reconciliation, and adjudication. Business policy never reads raw source or reply strings.

For every material clause, predicate-like and evaluative spans must be either consumed into a typed
relation, classified as a literal or harmless control, or retained as unresolved material with an
exact span and category. Missing extraction is not evidence of safety:

- unresolved harm or medical outcome material keeps the human-harm referent unknown and fails
  closed only for a humorous candidate;
- unresolved positive evaluation aligned with a harmful source becomes uncertain humor intent;
- unresolved copular, reduction, or packaging material becomes an uncertain reply shell rather
  than a high-confidence `none`;
- negated, possible, planned, feared, reported, conditional, and historical-comparison outcomes
  retain their assertion state instead of becoming completed facts.

Lexical resources normalize individual lemmas and short compounds into categories. Clause grammar
and relation binders own subject, predicate, affected entity, target, complement, decoration, and
assertion roles. Adding a synonym does not add a policy rule, and adding a syntax frame does not
duplicate harm, humor, or diversity adjudication.

## Semantic IR adjudication invariants

The application separates extraction from adjudication. Extraction may normalize bounded lexical categories and must retain exact evidence spans, entity type, predicate, affected referent, event outcome, evaluation target, shell operator, confidence, and unresolved material. Adjudication consumes only those typed relations.

- Human harm requires a human referent plus an explicit death, serious-injury, critical-condition, or non-survival relation.
- Technical, product, business, legal, political, market, crypto, and process referents resolve to non-human failure when the relation is clear.
- Sarcasm requires positive evaluation, negative or harmful source polarity, target-event alignment, and an implausible literal reading.
- Provider extraction is additive and cannot erase deterministic relations. Invalid spans or material conflicts remain explicit and fail closed only for the affected candidate.
- Structural reuse compares canonical operator and subject/complement/decoration roles. Literal definitions, formulas, quotations, and necessary factual clarifications remain outside reusable shell families.

## Clause identity, request-local coverage, and artifact integrity

Do not bind a predicate in one coordinated clause to an entity in another clause merely because it
is the nearest recognized head. Unknown asserted human result-state complements remain material
when structural roles indicate loss, limitation, reduced ability, support dependency, or another
potential physiological impairment. Ordinary interface, schedule, location, preference, social
demeanor, and non-human functional states remain controls.

On an explicit fatal source, incomplete reply coverage is not proof of a plain observation.
Candidate-level `HUMOR_INTENT_UNCERTAIN` remains until a validated typed relation or high-confidence
literal communicative function resolves the request-local item. Provider `none`, stale item IDs,
changed categories, altered spans, or missing roles cannot erase uncertainty.

## Materiality and proposition controls in Vouch 0.19.3 RC

Do not infer harm from a human noun plus a gerund, location, or ordinary dependency. Require a
bound impairment or abnormal-outcome role. Do not call a bare fragment a plain observation; a
literal communicative function requires proposition completeness and exact evidence spans. On a
source with typed human harm or `possible_harm`, unknown evaluative material remains candidate-local
uncertainty. Provider spans must align to Unicode token boundaries and one clause scope.
