# SignalCatcher

A benchmark for what an information source actually contributed.

For a given writer or publication, it extracts the substantive claims their work
carries and scores each one twice:

- **Originality** — was this claim already in the record when it was published?
  Measured by searching everything in the corpus dated *strictly before* it.
- **Influence** — did it travel afterwards, and did it travel *from here*?
  Measured by searching forward in windows, and comparing uptake against what
  the topic was doing anyway.

The two are combined multiplicatively, not additively. An idea that was new but
went nowhere, and an idea that spread but was already common, both fail to show
that this source contributed something. Only the conjunction does.

## The problem this design is mostly about

The naive version of this benchmark does not work, in a way that is easy to miss
because it still produces confident numbers.

**Absence of prior art is an argument from silence.** Finding an earlier source
proves a claim was not new. *Not* finding one proves only that you did not find
one. A pipeline that scores "no prior art → 1.0" is reporting the size of its own
corpus, and will rate obscure claims most highly.

So originality is reported as an interval:

```
hi = 1 - prior_strength     what the gathered evidence supports
lo = hi × coverage          what survives if the unsearched space is as full
                            of prior art as the searched space
```

`coverage` is driven mainly by whether the pre-window corpus contained
*topically relevant* material at all. Searching a thousand on-topic documents
and finding no statement of the claim is real evidence of novelty. Searching a
corpus that had nothing on the subject is evidence of nothing, and the interval
widens to say so. **A thin corpus cannot mint high scores.**

**The judge has already read the internet.** Ask a frontier model "was this
original in 2019?" and it answers from hindsight — the score tracks fame, not
priority. Three defences, in `score/adjudicate.py`:

1. The judge sees only dated excerpts and is instructed to reason from them alone.
2. It must return a **verbatim quote** from the candidate supporting its verdict.
   Unsupported matches are downgraded in code, not by request — so confident
   hand-waving cannot score.
3. The `no_retrieval` control re-runs every judgement with the evidence removed.
   If the scores survive, the evidence was never doing the work.

**Correlation is not transmission.** A columnist writing about a live story is
always followed by more coverage of that story. Counting those follow-ons as
influence scores the busiest desk highest. So uptake is measured against the
topical background — documents on the same subject in the same window that do
*not* assert the claim — and influence is the excess.

## Influence, in terms a newsroom already uses

| Dimension | What it is |
|---|---|
| **Lead time** | Days before the next *independent* source said the same thing — the scoop, in days |
| **Pickup breadth** | How many independent outlets carried the claim, per window |
| **Attribution rate** | How many of them said where it came from |
| **Phrase spread** | Verbatim reuse of the source's own distinctive wording |
| **Lift** | Pickup relative to the topical background rate |

Windows: 7 / 30 / 90 / 180 / 365 / 1095 days.

Phrase spread is the strongest evidence available. Semantic similarity cannot
distinguish influence from two writers independently noticing the same thing; a
rare coinage reappearing verbatim in someone else's work is much harder to
explain by coincidence. Fingerprints are validated as exact substrings of the
source text before use — models paraphrase when asked to quote, and an invented
phrase would either match nothing or match generic wording and manufacture
influence.

## The controls

Scores ship with the controls that test them. Each is a prediction that must come
true if the benchmark works.

| Control | Prediction |
|---|---|
| `date_shift` | Score claims as if published 4 years later. Prior-art strength must **rise**. |
| `no_retrieval` | Re-judge with evidence withheld. Scores must move or flatten. |
| `decoy_source` | A contemporaneous other writer must score lower on influence. |
| `shuffled_claim` | Give claims the wrong publication dates. Influence must fall. |
| `judge_agreement` | Re-judge at another effort setting. Verdicts must not swing wildly. |

Two details matter more than they look:

- `date_shift` compares `hi` (the evidence-only term), **not** the reported score.
  Moving the date forward enlarges the prior window, which raises coverage, which
  raises the blended score — so comparing scores lets a coverage gain cancel the
  very prior-art effect the control exists to detect, and the control clears
  itself. This was a real bug, caught by the control failing.
- A control with no signal to test reports **n/a**, never PASS or FAIL. Comparing
  0 against 0 is not evidence that the metric discriminates, and printing FAIL
  for it would read as a verdict on the source rather than on the corpus.

## Status

Working end to end. On a 1.4k-document, 5-year corpus:

```
[PASS] date_shift       baseline=+1.000 observed=+0.797 delta=-0.203
[PASS] no_retrieval     baseline=+0.425 observed=+0.444  -> passed via collapse to uniform
[PASS] judge_agreement  mean abs. diff 0.106 (max 0.264)
[n/a ] shuffled_claim   -> baseline influence ~0: nothing for the shuffle to destroy
[n/a ] decoy_source     -> target influence ~0: no signal to compare
```

The two originality controls pass — the metric responds to dates, and to the
presence of evidence rather than to the judge's memory. **The two influence
controls are undecidable on this corpus**, and that is the honest current state:
diffusion cannot be observed across eight sources. Corpus breadth is the binding
constraint, not the scoring logic.

## Usage

```bash
uv venv && uv pip install -e .

signalcatcher corpus                          # what is in the pinned corpus
signalcatcher ingest astralcodexten.com       # add a Substack
signalcatcher ingest slatestarcodex.com --kind wordpress
signalcatcher score "Scott Alexander" --docs 3
signalcatcher validate "Scott Alexander" --decoy "Noah Smith" --json out.json
```

Corpus builds are restartable and idempotent: documents dedupe on URL and every
HTTP response is cached to disk, so a re-run fills only what is missing.

## Disk usage

The corpus grows with every publication ingested, so nothing here is unbounded
by default.

| Item | What it is | Disposable? |
|---|---|---|
| `corpus.db` | The dataset: document text, FTS index, embeddings, judgements | **No** |
| `cache/` | Raw HTTP responses, already parsed into the DB | Yes — capped at 512 MB, evicted oldest-first |

The HTTP cache is pure speed. Every response in it has already been parsed into
the corpus, so evicting an entry costs a re-fetch and never data. Left uncapped
it reaches several gigabytes over a large build, which is not a trade anyone
agreed to.

```bash
signalcatcher corpus                  # includes a disk breakdown
signalcatcher purge-cache             # trim to the cap
signalcatcher purge-cache --all       # empty it entirely
```

**Putting the corpus on another volume** — the repository then stays a few
megabytes of code:

```bash
export SIGNALCATCHER_DATA=/Volumes/YourDrive/signalcatcher
# or per-command:
signalcatcher --data-dir /Volumes/YourDrive/signalcatcher corpus
```

Resolution order is `--data-dir` → `$SIGNALCATCHER_DATA` → `./data`.

Note that `.venv` is a further ~870 MB, about 500 MB of which is PyTorch, pulled
in by the local embedding model. Dropping `sentence-transformers` reclaims that
but disables dense retrieval, which lowers coverage and therefore *widens* the
originality intervals — the scores get more conservative, not wrong.

## Layout

```
signalcatcher/
  models.py       claim / document / evidence schema
  db.py           SQLite + FTS5 store; time-sliced search is the core primitive
  config.py       pinned, hashable run configuration
  llm.py          Claude wrapper: structured output, caching, grounded mode
  ingest/         substack, wordpress, hn, gdelt, generic article extraction
  index/          embeddings (local by default) + RRF fusion of three retrievers
  extract/        claims.py — documents to portable, checkable propositions
  score/          adjudicate, originality, influence, aggregate
  validate/       controls.py — the five negative controls
  pipeline.py     end to end: a source in, an audited scorecard out
  report.py       renders a run with its receipts
```

## Data sources

Verified working: **Substack** (archive API + per-post bodies), **WordPress**
REST (the pre-2021 blogosphere, which is the prior-art window for everything
since), **Hacker News** via Algolia (dated discovery back to 2007, plus hard
transmission evidence), **GDELT** (news pickup breadth, 2017+, keyless),
**OpenAlex** (citation ground truth for calibration). Two ingestion traps worth
knowing: Substack serves short archive pages at arbitrary offsets, so treating a
short page as end-of-archive silently truncates a publication to its most recent
weeks; and comment threads must be stripped, or a popular post is scored on its
readers' words rather than the author's.

## What is not built yet

- **The AI-value link.** The claim layer is designed to support it — measuring
  model performance with and without a source in the retrieval context — but the
  ablation harness is not written.
- **Durability.** Long-horizon vindication (was the claim corroborated,
  contested, retracted?) has a place in the schema and no scorer.
- **A gold set.** Known-original and known-derivative pieces, with datable first
  use, to calibrate the metric against cases whose answers are already known.

## Value to a model (`ablate`)

Two questions, measured separately:

- **Inference-time value** — does putting this source in the context window make
  the model answer better than it otherwise would?
- **Training-set value (proxy)** — does the model already know this unaided? For
  a claim that *originated* with this source, a correct closed-book answer is
  evidence the contribution is already priced into the weights.

The measurement only means anything because of the third condition. Comparing
"with the source" against "with nothing" mostly measures whether context helps,
which it always does. So every question is also asked with a **decoy** context:
the same amount of topically matched material from the same period, by other
writers.

```
inference_value = score(with source) - max(score(closed book), score(decoy))
```

When the corpus has no contemporaneous material to build a decoy from, the decoy
condition **did not run** and is reported as `n/a`, never as 0.0 — averaging an
unavailable control in as zero silently inflates the source's apparent value.

Measured on the current corpus:

| Source | closed book | + decoy | + source | inference value | already in weights |
|---|---|---|---|---|---|
| Brian Potter (Construction Physics) | 0.542 | n/a | 0.983 | **+0.442** | 33% |
| Scott Alexander (ACX) | 0.895 | 0.795 | 0.875 | **−0.050** | **100%** |

The contrast is the point. The model already knows essentially all of ACX, so
adding it to context buys nothing — its value was realised at training time.
Niche construction reporting still carries substantial inference-time value.
Those are different assets, and a benchmark that reported one number would hide
the distinction that matters most to both a publisher and a data buyer.

Caveat worth stating plainly: two ACX claims scored *negative* value, meaning the
supplied excerpt made the answer worse than no context. That is an
excerpt-selection failure, not a property of the source.
