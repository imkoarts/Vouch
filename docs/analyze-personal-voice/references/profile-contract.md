# Voice profile contract

Analyze functions, not a costume.

## Observable features

- distribution of words per message and sentences per message;
- lowercase/uppercase tendency without turning it into a quota;
- terminal punctuation, questions, exclamations, ellipses, and emoji frequency;
- fragments, contractions, conjunction openings, and paragraph rhythm;
- directness, skepticism, confidence, and informal uncertainty markers;
- agreement, disagreement, qualification, questioning, and practical next-action patterns;
- humor through literal answers, reversal, understatement, teasing, absurd extension, or
  self-deprecation;
- vocabulary that recurs across topics versus nouns tied to one subject.

## Interpretation rules

- Prefer local context over corpus averages.
- Treat approved outputs as stronger evidence than merely observed samples.
- Preserve the social move before surface quirks.
- Do not add fake typos, forced lowercase, slang, profanity, or emojis as a style costume.
- Attack claims or situations, never identity or private life.
- Do not make a reply meaner than its source.
- A plain observation can be more authentic than a polished one-liner.
- Keep factual and evidential constraints above voice matching.

## Confidence

Mark a trait high confidence only when it appears across multiple topics and contexts. Mark it
medium when repeated in one context. Mark it tentative when based on a few samples. If fewer than
five usable samples remain, return an onboarding preference profile and explicitly defer corpus
claims.
