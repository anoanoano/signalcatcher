# SignalCatcher

A benchmark for what a piece of writing actually contributed to the discourse.

For a given text — a blog post now, a publication's corpus at scale — it
extracts the substantive claims the text carries and asks, for each claim,
where it sits in time relative to everything else on its subject:

```
surprisal    1 − strength of the closest PRIOR statement found by a hard search
             (higher = nobody had said it)
adoption     rise in the claim's expression rate in its topical neighbourhood
             after first use, at peak   (higher = discourse moved toward it)
vindication  signed balance of later evidence: borne out (+1) vs refuted (−1)

predictive value = surprisal × adoption × (1 + vindication) / 2
```

Read it as: *you said something nobody was saying, the conversation came around
to it, and the record proved you right.* A claim everyone was already making
scores ~0 via surprisal. A claim nobody picked up scores 0 via adoption. A claim
events refuted is crushed by the vindication term — and is labeled **wrong**,
which the method deliberately keeps distinct from **ignored**.

Two refusals define the design. It does **not** ask whether later writers *read*
the text — causal influence is unknowable from text alone, and a writer who
anticipated where discourse was going without causing it still saw it (the
question is whether content *predicts* later content). And it does **not**
reward being right about what everyone already expected: a correct prediction
that was common knowledge scores zero, on purpose.

## The live report

The pilot results — seven texts, 35 claims, every number expandable to its
dated, quoted receipts — render as a self-contained explorer:

```bash
python scripts/build_report.py out.html
```

## How a claim is scored

1. **Canvass** (`score/canvass.py`) — extract the text's claims
   (`extract/claims.py`), then select the 4–10 boldest with forced diversity
   across kinds: facts, theses, frames, predictions, syntheses. Claims must be
   *portable* — stated so they can be searched for in another writer's work
   without the original's vocabulary.
2. **Anchor at true first use** (`score/firstuse.py`) — writers repeat their
   theses for years; the author's own back catalog is searched for the earliest
   statement and every clock starts there. Measured: one claim's clock moved
   back 761 days from the article it was extracted from.
3. **Assemble the evidence shell** (`shell.py`) — see architecture below.
4. **Surprisal** (`score/predictive.py`) — a high-recall search (query
   expansion into other vocabularies + lexical + dense + verbatim-phrase
   retrieval) over everything dated before first use; the single strongest
   prior defeats novelty in proportion to how fully it anticipates the claim,
   and is stored as a dated, quoted receipt. This is an **existence** test: an
   earlier rate-based design compressed every claim into 0.85–1.0, because any
   specific proposition is rare even among documents on its own topic.
5. **Adoption** — in time windows after first use (months for event
   interpretation, years for slow-burn theses), a *seeded random sample* of the
   claim's topical neighbourhood is judged with a six-relation taxonomy:
   `states (1.0) · entails (0.9) · anticipates_directionally (0.5) ·
   partially_anticipates (0.3) · orthogonal (0) · contradicts (−1)`.
   Random sampling keeps windows of very different sizes comparable;
   top-similarity candidates are judged too but feed the evidence trail and
   vindication, not the rate. `entails` is where **implicit** content earns
   credit; `contradicts` is what gives the measure a sign.
6. **Arithmetic** — no further model judgment.

### Judging disciplines, enforced in code

| discipline | prevents |
|---|---|
| **Quote or downgrade** — verdicts stronger than orthogonal must include a verbatim span, checked mechanically; unquotable verdicts are demoted | matching from training memory instead of the evidence shown |
| **Hindsight flows one way** — the judge sees only dated excerpts and may not use what it remembers about outcomes | scoring fame instead of foresight |
| **First-use anchoring** | crediting a restatement as an origination |
| **Syndication collapse** (`ingest/dedup.py`, MinHash + LSH) before judging and at display | one wire story in twenty outlets counting as twenty adoptions (measured: 26–31% of fetched news) |

## Evidence architecture: spine, shell, manifest

No corpus can host "the whole discourse," and none needs to — the right
evidence base is **claim-shaped**. So:

- **Spine** (persistent, small): a fixed panel of ~34 independent writers'
  full archives, shared by every evaluation — what keeps texts comparable.
- **Shell** (per text, ephemeral): evidence fetched by a fixed, versioned
  recipe (`shell.py`, currently `shell-v1`) from fixed channels, dense exactly
  where the text's claims live.
- **Manifest** (`data/manifests/`): every query, hit count, and failure —
  "searched and found nothing" and "could not search" are different facts.

Channels, all free and keyless:

| channel | source | role |
|---|---|---|
| news | GDELT, keyword per claim, prior year included | pickup beyond the blogosphere; priors the trade press already broke |
| international | fixed panel of leading dailies via GDELT `domainis:` (Le Monde, FAZ, Spiegel, Corriere, El País, El Observador, Asahi, SCMP…) | cross-border, cross-language diffusion |
| academic | arXiv abstracts, date range inside the query | scholarly prior art — stops aggregators inheriting researchers' foresight |
| reference | Wikipedia article-creation dates per language; Wayback first-capture fallback when Wikimedia 403s | a datable "reached common knowledge" milestone, per language |
| forum | Hacker News (Algolia) | dated discovery and discussion |

Storage: SQLite + FTS5 (external-content index; time-sliced search is the core
primitive), local embeddings (bge-small — pinned weights beat a hosted endpoint
that can be reweighted under you), every LLM judgement disk-cached. The corpus
compresses to ~40% for cold storage; `snapshot`/`restore` archive it (Google
Drive auto-detected). The HTTP cache is capped with oldest-first eviction.

## Pilot results (Aug 2026)

| text | S̄ | Ā | V̄ | best PV |
|---|---|---|---|---|
| *Sizing up the New Axis* (Noah Smith) | 0.89 | 0.11 | +0.65 | **0.209** |
| *DeepSeek v3* (Zvi Mowshowitz) | 0.48 | 0.09 | +0.85 | 0.194 |
| *Ignore Allan Lichtman* (Good Reason — unknown writer) | 0.78 | 0.05 | +0.50 | 0.166 |
| *SVB run, day one* (Noah Smith) | 0.63 | 0.21 | +0.86 | 0.112 |
| *Expecting GPT-5* (Gary Marcus) | 0.80 | 0.09 | +0.46 | 0.089 |
| *AI Predictions 2023* (Tech Times — content mill) | **0.39** | 0.04 | +0.93 | 0.046 |
| *The Media Very Rarely Lies* (ACX — famous essay) | 0.80 | 0.04 | +0.58 | 0.029 |

The metric does not follow reputation, for auditable reasons: the famous essay
was argued *about* rather than adopted (dense engagement, near-zero adoption,
one net-negative vindication); the content mill's aggregated claims lost their
novelty once the arXiv channel put the actual academic priors in reach; the
unknown writer's Lichtman debunking — bold, right before the falsifying
election, vindicated — outscored both.

Separately, an ablation harness (`score/ablation.py`) measures a source's value
to a model: closed-book vs decoy-context vs source-in-context, with a
decoy-that-never-ran reported as unavailable rather than zero. Measured
contrast: niche construction reporting +0.44 inference-time value with 33%
already in the model's weights; ACX −0.05 with 100% — value captured at
training time. The publisher-facing and lab-facing measurements share one
claim layer.

## Validation

Controls are predictions that must hold if the benchmark works
(`validate/controls.py`): **date_shift** PASS (re-dated claims find their own
era as prior art, 1.00→0.80); **no_retrieval** PASS (withhold evidence, scores
collapse — the judge reads the corpus, not its memory); **judge_agreement**
PASS (verdict shift 0.11 across reasoning settings); shuffled-dates and
decoy-writer now decidable, next scheduled run.

The strongest trust argument is that the harness has repeatedly falsified
itself, with published numbers changing each time: surprisal compression
(SVB collapse fact 0.87→0.14 once the search actually looked, receipt
attached), aggregator credit (Tech Times thesis 0.96→0.15 once arXiv was in
reach), dense-window selection bias (fixed by seeded random samples), a
confounded control (date-shift compared a coverage-blended score that cancelled
the effect it tested), vacuous controls reporting FAIL instead of
not-decidable, and a serialization bug that shipped scores with empty evidence
trails. Details in each commit message.

## Running it

```bash
uv venv && uv pip install -e .
export ANTHROPIC_API_KEY=...        # or neither: see below

# corpus management
signalcatcher corpus                 # contents + disk breakdown
signalcatcher ingest astralcodexten.com          # Substack archive
signalcatcher ingest slatestarcodex.com --kind wordpress
signalcatcher snapshot my-corpus     # compress + archive (Drive auto-detected)
signalcatcher restore my-corpus
signalcatcher purge-cache

# the predictive pipeline (current path) runs via scripts:
python scripts/run_unit_deepseek.py  # template: one text under shell-v1
python scripts/build_report.py out.html
```

Judgements run on Claude (Opus). With no API credits, the pipeline falls back
to headless `claude -p` billed against a Claude subscription
(`LLM(backend="cli")`) — note a set `ANTHROPIC_API_KEY` silently takes
precedence over the claude.ai login, so the CLI backend scrubs it from the
subprocess environment. Long runs pace themselves against the subscription's
usage windows.

`--data-dir` or `SIGNALCATCHER_DATA` points the corpus at another volume.
The `score`/`validate` CLI subcommands run the earlier interval-based
originality/influence scorers, kept for the controls harness; the
surprisal/adoption/vindication path above supersedes them for evaluation.

## Layout

```
signalcatcher/
  models.py        claim / document / evidence schema
  db.py            SQLite + FTS5; time-sliced search primitive
  llm.py           Claude wrapper: structured output, caching, grounding,
                   API or subscription-CLI backend
  shell.py         per-text evidence recipe (shell-v1) + manifests
  ingest/          substack, wordpress, hn, gdelt, arxiv, wikipedia,
                   article extraction, syndication dedup
  index/           local embeddings + RRF fusion of three retrievers
  extract/         claims.py — texts to portable, checkable propositions
  score/           canvass, firstuse, predictive (S/A/V), anticipate,
                   adjudicate, ablation; legacy originality/influence
  validate/        controls.py — the negative controls
scripts/           corpus build, unit runners, rescores, report generator
data/manifests/    per-unit shell manifests (what was searched, what failed)
```

## Honest limits

- **Pilot scale**: 35 claims across 7 texts; per-text averages over ~5 claims
  are indicative, not stable. The interesting comparisons are claim-level and
  receipt-backed.
- **Everything is corpus-relative**: a prior living in a paywalled archive, a
  podcast, or an untapped language is invisible. The shell narrows this
  per-claim; it cannot close it.
- **The judge is a model**, constrained by grounding, quote-enforcement, and
  measured agreement — individual verdicts can still err; every one is stored
  with its supporting quote for audit.
- **The recipe is part of the instrument**: a biased "where to look" biases
  scores invisibly. Mitigations are a fixed versioned recipe, manifests, and
  coverage discounts — not neutrality.
- **Unbuilt**: podcast transcripts; a broader ring of small publications;
  decade-scale durability; a gold set of claims with independently known
  priority; the two pending controls at current scale.
