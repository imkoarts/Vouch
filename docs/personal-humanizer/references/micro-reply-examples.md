# Personal Humanizer 3.6.0 Micro-Reply Examples

These are functional examples, not reusable templates.

## Plain observation preferred

Source: A brand criticizes its social manager, who also handles ten other jobs with no budget.

> half the time the social manager is good, they just gave them 10 other jobs and no budget

Why it works:
- ordinary speech
- source-specific constraint
- complete thought
- no punchline requirement

More constructed and less preferred:

> brands will spend millions building a product then hand the account to whoever is least busy

This may be accurate, but it is shaped like a quote-card line. Prefer the plain response when both are grounded.

## Genuine question

Source: A product says exported scores can be downloaded but does not say whether test cases are included.

> do the test cases come with the export, or just the scores?

Why it works:
- the question is genuinely unresolved
- exact source context
- no manufactured premise

## Specific qualification

Source: A post says automation can replace social media work.

> maybe the posting part, but not the judgment about what is worth saying

Why it works:
- direct qualification
- two supported parts
- no fake thesis

## Two sentences when necessary

Source: A service says migration is supported, but historical corrections do not transfer.

> the model can move. the correction history still cannot, which is the part teams actually trained

Why it works:
- the second sentence carries a necessary consequence
- no forced compression

## Genuine short humor

Source: A model adds a low-verbosity setting.

> finally an official anti-yap switch

Why it works:
- source-specific semantic transformation
- concise because the joke is complete
- no explanation

The spaced form `low verbosity` must receive the same routing and grounding result. Hyphenation is
not a semantic affordance.

## Source-explicit legal joke

Source: A founder broke two NDAs in one tweet.

> bro broke two NDAs in one tweet

Why it works:
- the source explicitly establishes the two NDA breaches
- the reply changes register rather than inventing the allegation
- the exact event stays source-specific

The same reply is invalid when the source only says that confidential partnership details were
disclosed. A humor label cannot turn that implication into an established NDA breach.

## Source-specific dry reframe

Source: Two accounts independently post the same harmless meme.

> two independent sources have now confirmed solana summer

Why it works:
- fake-newsroom language is the actual incongruity
- the source contains two accounts
- harmless joke, not a factual accusation

## Invalid marker-wrapped paraphrase

Source: Hungary removes President Sulyok as part of dismantling Orbán's influence.

> anti-Orbán campaign speedrun

Why it fails:
- `speedrun` is only a marker
- the remainder is source echo
- no semantic transformation

## Invalid quote-card reply

> brands will build the future then hand the account to whoever is least busy

Why it fails when repeatedly selected:
- reusable under unrelated sources
- polished setup-to-twist cadence
- ordinary response was available

## False brevity

Source: The plan works only when a human reviews the result.

> works. mostly

Why it fails:
- the qualification has been clipped away
- brevity changes the meaning

## Grounded reaction that is still unsuitable humor

Source: A hospital reports 10 deaths after an outage.

> the outage is doing heroic amounts of work

Why it fails:
- the callback is source-specific and can be factually grounded
- it trivializes a fatal event
- reaction grounding does not override humor suitability
- keep a respectful plain observation, direct response, qualification, or question instead

## Metamorphic one-liner reuse

Recent reply:

> fundraising is just sales with extra steps

Candidate:

> onboarding boils down to paperwork in a nicer font

Why it is lowered:
- the nouns and connector changed
- both lines use the same reductive-identity quote-card shell
- source specificity and structural diversity are separate dimensions

Recent reply:

> nothing says simplicity like six screens

Candidate:

> few things communicate simplicity better than six more screens

Why it is lowered:
- both use the same inverse-praise contradiction
- bounded paraphrase normalization detects the communicative shell
- one isolated use is not automatically blocked

## Humor intent is text-first

Source: A hospital reports 10 deaths after an outage.

Final text, even when metadata says `plain_observation`:

> the outage is doing heroic amounts of work

Why it fails:
- the final text performs a joke-shaped callback
- provider metadata cannot disable humor safety
- `HUMOR_ON_TRAGEDY` blocks the candidate and `REPLY_TYPE_MISMATCH` records the classification disagreement

## Common harm wording

Treat `lost their lives`, `passed away`, `wounded`, and an explosion or blast that injured people as
explicit human harm. These formulations are semantically equivalent to the already covered fatality
and serious-injury language for humor suitability.

## Crash disambiguation

Human tragedy:

> five people died in a car crash

Non-tragedy product context:

> the app crash was fixed in version 2.1

Do not infer personal tragedy from the bare word `crash`. Use bounded context. Software, server,
browser, market, crypto, and price crashes are not human tragedies; vehicle, fatal, killed, injured,
or passenger-linked crashes are sensitive.

## Additional bounded shell reuse

Recent reply:

> nothing says simplicity like six screens

Candidate:

> nothing screams simplicity like eight screens

Both use the same inverse-praise shell.

Recent reply:

> fundraising is just sales with extra steps

Candidates:

> onboarding is paperwork with better branding

> onboarding, translated: paperwork with a progress bar

> onboarding is the same thing as paperwork, just with nicer copy

All perform the same decorated or reductive identity job. A provider label such as
`plain_observation` does not suppress structural-family inspection.


## Implicit evaluative sarcasm on tragedy

Source: A hospital reports 10 deaths after an outage.

> the outage really delivered

Why it fails:
- the positive evaluation points to the harmful event
- the source makes a literal positive reading implausible
- a `plain_observation` label cannot suppress humor safety

The same applies to `love that for the hospital`, `excellent work from the outage`, `nice work
from the outage`, `10/10 execution from the outage`, `exactly what the hospital needed`, and reward
relations such as `the outage earned employee of the month` when the target is grounded by the
source. Literal praise remains allowed when the source establishes a genuinely positive result.

## Technical death language is not human tragedy

Source: The API killed the worker process and dead code was removed.

> dead code has left the chat

Why it may remain eligible:
- both death expressions have bounded non-human referents
- no victim, fatality noun, injury predicate, or human disaster consequence is present
- humor suitability is evaluated independently from the literal token `dead` or `killed`

If the same source also says that two patients died, the source remains sensitive. Explicit
human-harm relations also include `no survivors`, `fatally injured`, `mortal injuries`, `body
count`, `bodies were recovered`, `never made it out`, `ended ten lives`, `critical condition`, and
`found lifeless`.

## Decorated identity and reveal reuse

Recent reply:

> fundraising is just sales with extra steps

Candidates such as these share a bounded identity/reduction relation and should be lowered:

> onboarding is paperwork dressed up as product strategy

> onboarding is paperwork under a new name

Additional repackaging variants such as `paperwork pretending to be product strategy`,
`paperwork cosplaying as product strategy`, `paperwork, rebranded`, `paperwork plus a progress bar`,
and `paperwork in product clothing` share the same relation even when their connector words differ.

Reveal variants are likewise compared by operator and slots:

> turns out onboarding was just paperwork

> as it turns out, onboarding is paperwork

A literal definition such as `OAuth, in other words, is an authorization protocol` is not a
reductive one-liner merely because it uses `in other words`.


## Relation-scope additions in Vouch 0.17.6

Sarcastic evaluation is relation-based. `great timing from the outage`, `a round of applause for
the outage`, `five stars for the outage`, `could not be prouder of the grid`, `the outage crushed
it`, and `another triumph for the grid` all combine a positive evaluation relation with a harmful
source and a grounded target. The same language remains literal praise when the source describes a
genuinely positive result.

Human-harm result states include a human being declared deceased, remaining in a coma, being left
on life support, a physical crash proving fatal for passengers, passengers being lost in that
crash, and people not making it home after the crash. The detector requires a human relation or
physical-event context rather than treating every use of `lost`, `fatal`, or `coma` as tragedy.

Clear non-human failures include builds, pipelines, databases, nodes, models, bills, and launches
that died, were killed, or were dead on arrival. A mixed source such as `The server died. Two
patients died during the outage.` remains human-sensitive.

Decorated reductions such as `onboarding is glorified paperwork`, `onboarding is paperwork in
disguise`, `onboarding is paperwork with a fresh coat of paint`, and `onboarding: paperwork with
better branding` share the same relational family as other reusable reduction shells. Literal
definitions, formulas, and source-specific factual clarifications remain outside that family.

## Hybrid semantic extraction examples in Vouch 0.19.0 RC

```text
Source: A hospital reported ten deaths after the outage.
Reply: stellar execution from the outage
Extraction: positive praise → outage event; death outcome → people
Adjudication: evaluative irony + human-sensitive source → block humor candidate
```

```text
Source: The deployment died after the rollback.
Extraction: death predicate → software deployment
Adjudication: non-human technical failure → no tragedy block
```

```text
Recent: fundraising is just sales with extra steps
Candidate: onboarding is paperwork with a makeover
Extraction: reduction/decorated-identity operator with subject, complement, and decoration slots
Adjudication: structurally reusable shell despite changed surface wording
```


### Coverage-aware unresolved material

```text
Source: Two clinicians were incapacitated after the blast.
Extraction: human/medical context + unbound outcome predicate → unresolved possible_harm
Adjudication: do not assert death; fail closed only for a humorous candidate
```

```text
Source: A clinic reported deaths after a power failure.
Reply: remarkable from that system
Extraction: positive evaluation head, unresolved target alignment
Adjudication: uncertain humor intent; do not silently treat it as a plain observation
```

```text
Recent: fundraising is just sales with extra steps
Candidate: onboarding, in polished terms, is paperwork
Extraction: copular/reduction material with bound subject and complement, packaging role retained
Adjudication: compare the canonical relation and slots rather than punctuation or adjective order
```


### Coverage-driven provider escalation

Unknown evaluation or packaging language is not silently classified as safe. Deterministic
coverage retains the exact unresolved span; live inspection may request a typed provider proposal,
while offline inspection fails closed only for that candidate.

Literal recovery remains a positive event. `recovered` becomes a death outcome only when relation
roles establish a fatality object such as recovered bodies. Provider output cannot override this
through an approval or issue-code field.

## Human result-state uncertainty

Source: `The collision left two passengers glassy-eyed and barely rousable.`

Unsafe humor candidate: `another flawless commute`

The human result-state complement is materially physiological but not deterministically diagnosed.
Keep it as `possible_harm`, request bounded provider extraction in live mode, and block only the
humor candidate offline. A respectful factual reply or genuine question remains available.

Control: `The delay left two users waiting outside.` is an ordinary location/activity state and is
not promoted to human medical harm without additional safety context.

## Result-state and sensitive-source examples in Vouch 0.19.2 RC

A technical failure and a later human result state remain separate clause-local relations. The
technical predicate cannot borrow the later human referent. An unfamiliar asserted human
impairment remains `possible_harm` and blocks only the affected humor candidate offline.

On a source with explicit human deaths, an unresolved ceremonial, evaluative, or otherwise
material reply is not silently treated as plain. It remains `HUMOR_INTENT_UNCERTAIN` unless a
validated literal communicative function or typed provider relation resolves the exact request
item.


A participial, gerund, `with`, dependency, or coordinated human result complement remains one bound
result frame. Unknown physiological or functional states stay `possible_harm`; ordinary UI,
business-authority, transport, location, schedule, and non-human states remain controls.

On a typed or unresolved human-harm source, a materially incomplete reply is candidate-local
uncertainty unless exact spans validate a plain factual assertion, direct response, qualification,
or genuine question. Negated factual actions and passive evidential uncertainty are literal
controls, not reply-shell operators.

## Materiality and literal controls in Vouch 0.19.3 RC

- `The delay left two riders waiting under the shelter.` is ordinary activity/location, not harm.
- `The impact left two riders unable to keep their balance.` remains `possible_harm` until typed.
- `a ceremonial marker for the failed system` is an incomplete fragment, not a plain observation.
- `the response should focus on the affected people` is a direct action recommendation.
- `the failure sequence remains under review` is an evidential qualification, not a reply shell.
