# Capstone Report — Ranking Signal Analysis

- **Author:** si-ux
- **Lane:** Ranking Signal Analysis
- **Repo:** https://github.com/si-ux/FlyrankAI-Internship
- **Date:** 2026-08-09

## 0. Abstract

Which pages in a content portfolio are most likely to lose search position next month, and can
that be turned into a review queue a person can actually work down? Using the FlyRank
pseudonymized warehouse release — 78.8 million daily rows across 104 clients, queried in place
with DuckDB over remote Parquet — I framed the question as ranking, labelled a page as declining
when its impression-weighted average search position worsened by at least one place between a
90-day feature window and the following month, and compared a learned model against a frozen
four-condition hand-written rule on a client-grouped split. On a sealed June 2026 test month the
random forest put 90% declining pages in its top 50 against a 56.2% base rate, while the
transparent rule reached 78% — and deeper in the queue the rule overtakes it, so the two win at
different depths. The same model scored 0.96 under a careless random split that let it memorise
clients. The output is a reason-coded
review queue with a per-client cap, an abstention gate for portfolios where skill was never
measured, and drift monitors. It is decision-support for ordering manual review, not a prediction
of any page's fate and not a claim about Google's algorithm.

## 1. Problem framing

**Decision supported:** the *order* of a manual content review queue — nothing else.

A strategist who owns a portfolio can open perhaps 20–50 pages in a review cycle. Today that
order comes from intuition, from whoever escalated loudest, or from a dashboard sorted by traffic
— which surfaces the *biggest* pages rather than the ones *slipping*.

- **Unit of analysis:** one pseudonymized content item (a page).
- **Output:** a ranked queue carrying an action, a confidence band, and reason codes.
- **Action a human takes:** open the page, establish what changed, decide whether to refresh.
- **Cost of a wrong call:** roughly an hour spent on a page that turned out fine.

That asymmetry — an hour wasted versus a valuable page left to decay — is why this ships as a
ranking rather than an automation, and why precision at the *top* of the list matters more than
overall accuracy.

**Why ML helps at all.** The folk rule ("refresh the old stuff") is testable, and it largely
fails: content age carries little weight once ranking behaviour is in the model. What does work
is recent position movement interacting with volatility and visibility — awkward as an
if-statement, easy to learn. The honest qualifier, established in section 3, is that a
transparent four-condition rule captures most of that signal on its own, and beats the model
outright from K=500 outward.

## 2. Data safety

**Used:** `fact_content_daily_performance` (daily grain, aggregated to windows) joined to
`dim_content` (static page metadata), from build `v20260703`. `dim_clients` informed coverage
checks only.

**Deliberately excluded, with reasons:**

| Excluded | Why |
|---|---|
| `fact_content_query_90d`, entire table | its fixed window opens **2026-04-02**, inside my outcome month — future information for this label. Established by a date check, not judgement |
| `trend_direction`, `trend_pct` | label-derived by construction |
| `last_optimized_date`, `optimization_eligible_date` | decision-derived — they encode an action FlyRank's own system already chose; learning them means learning the old rule, not the world |
| `client_hash_id`, `content_hash_id`, all hashes | pseudonyms — grouping, joining and splitting only, never features |
| GA4 family (`ga4_*`, `sessions_*`, `ai_*`, `scroll_events`) | three-valued availability flags; ~30.7% of rows in a single month carry a **NULL** `ga4_data_available`, neither zero-filled nor flagged false |
| `is_published`, `is_deleted` | platform state that can change after the prediction cut |

**Leakage risks considered and tested.** Label-derived features (none present, asserted in code);
overlapping windows (feature and label windows verified disjoint); decision-derived product flags
(excluded); identifiers as features (excluded); grouped-split integrity (asserted per fold). A
**positive control** plants `o_pos` into the feature set and confirms the harness detects it —
AUC jumps 0.659 → 0.950 — so the clean results are clean because the test *can* fail.

**Public safety confirmed.** The release contains no client names, domains, URLs, page titles or
raw queries. All identifiers appearing in notebooks and figures are truncated pseudonyms
(`p1df31e`, `c8636`). Datasets and prediction caches are gitignored; metrics JSONs are committed
as receipts; CI fails any commit containing a dataset. The Hugging Face token is never written
into a notebook cell.

## 3. Baseline

Before any model, two signals were checked — both premises behind real FlyRank flags — and only
the survivors were allowed into the rule.

| Signal | Flag it underpins | Verdict | Evidence |
|---|---|---|---|
| Staleness (`days_since_update`) | refresh flags | **FALSE** | the field is future-dated for 81.8% of pages; on the 19,403 testable pages the gradient is 0.586 vs a 0.567 base, and every other bucket is under the 50-row floor |
| CTR vs position, within tier | CTR-fix | **MIXED** | supports on `page_1` (+10.7pt, n=50,913) and `top_3` (+7.8pt); flat on `page_3_5`; untestable on `deep` |

Staleness was removed from the rule by its own verdict. CTR-for-tier entered only for pages at
position ≤ 20, where the claim's own logic applies.

**The rule** — four conditions, no fitted parameters, all inside the feature window:

| Condition | Points |
|---|---:|
| `sliding` — March position ≥ 1.0 place worse than January | **3** |
| `shallow` — 0 < position ≤ 20 | 1 |
| `low_ctr_for_position` — below tier-median CTR **and** shallow | 1 |
| `unstable` — position volatility ≥ median | 1 |
| `visible` — ≥ 1,000 impressions | 1 |

Each page carries exactly **one** reason code (first match wins, most specific first) and one
action label. `sliding_on_page_1_2` is the highest-yield code at a **0.777** observed decline rate.

**Why it is a fair comparison:** same rows, same label, same precision@K metric as the model,
**frozen before modelling began** and carried unchanged onto the sealed frame. One asymmetry is
stated rather than hidden: the rule's within-tier CTR median is a population statistic over all
rows, so the rule sees the full feature distribution while every model number is strictly
out-of-fold. No label touches it, but the comparison mildly favours the rule.

**A finding from building it.** Four integer conditions produce only **8 distinct scores** across
106,461 pages, so thousands of pages tie. On integers alone the rule still beats the base rate
(P@50 0.680) but is far weaker at the top than the tie-broken version (0.820). The frame arrives
clustered by client, so an unbroken tie block is largely one client's pages. Breaking ties by
exposure is a **priority judgement** — "among equal evidence, review what more people see" — and
roughly a third of the rule's top-of-list performance rests on it.

**Its numbers:** P@10 0.900, P@50 0.820, P@500 0.916 against a 0.567 base rate.

## 4. Model / analysis

**Target.** `is_position_decline = 1` when a page's impression-weighted average search position
worsens by ≥ 1.0 places between the feature window (Jan 1 – Mar 31 2026) and the outcome month
(April 2026). Base rate **0.5673**.

Position is impression-weighted throughout — `SUM(gsc_sum_position) / SUM(gsc_impressions)` —
because averaging daily position figures would let a 2-impression day count as much as a
20,000-impression one.

**Methods, in increasing opacity:** logistic regression (readable floor), a depth-4 decision tree
(printable), and a random forest (300 trees, `min_samples_leaf=5`). Simplicity is a feature; the
forest earns its opacity only if it beats the other two and the rule.

**Features (18).** Ranking behaviour from the feature window — position, volatility, within-window
trend, impressions, clicks, CTR, days with impressions — plus static metadata from `dim_content`
(word count, character count, keyword token count, search volume, competition, CPC, backlinks,
content type, intent, competition level, age at the cut date).

**Left out on purpose:** everything in section 2, plus per-day `gsc_avg_position` (unweighted and
misleading). Because missingness follows `content_type`, four explicit `has_*` flags are added
rather than median-filling blind, which would smuggle content type into the features.

## 5. Evaluation

**Split:** `GroupKFold(5)` on `client_hash_id`, out-of-fold. Pages come in portfolios sharing a
CMS, a template and one team's habits; a random split lets a model recognise the client and
recite its average outcome. The honest question — and the deployment case — is whether it works
on a portfolio it has never seen.

**Development window (label April 2026), grouped OOF:**

| Model | ROC AUC | P@50 | P@500 |
|---|---:|---:|---:|
| base rate | 0.500 | 0.567 | 0.567 |
| logistic_regression | 0.617 | 0.740 | 0.682 |
| decision_tree_d4 | 0.612 | 0.320 | 0.362 |
| **baseline_rule (frozen)** | 0.637 | 0.820 | **0.916** |
| **random_forest** | **0.651** | **0.880** | 0.844 |
| rf + client-relative | 0.637 | 0.860 | 0.870 |

**Sealed test (label June 2026, model fitted on the dev window only):**

| Evaluation | base rate | ROC AUC | P@50 |
|---|---:|---:|---:|
| SEALED, all clients | 0.562 | **0.692** | **0.900** |
| SEALED, unseen clients only | 0.295 | 0.608 | 0.600 |
| SEALED, frozen rule | 0.562 | 0.656 | 0.780 |

**The split diagnostic — the headline methodological result.** The same random forest scores
**AUC 0.777 and P@50 = 0.960** under a random split. Nothing about the model improved;
it was merely allowed to see other pages from the same client. Published carelessly, that would
have claimed roughly **double** the true skill above base rate.

**Error analysis.** Permutation importance is dominated by `f_pos_trend` (0.103, ~3× the next
feature), followed by position, volatility and days-with-impressions — all *ranking-behaviour*
signals; content properties barely register. Since a dominant feature is the classic leakage
symptom, I ran the prescribed test: removing it costs 0.071 AUC and degrades gracefully rather
than collapsing, so it is a strong signal, not a leak.

Per-client AUC ranges **0.48 to 0.77** — for at least one portfolio the model is no better than a
coin flip, which a pooled AUC hides entirely. Calibration is monotone but over-confident at the
top (predicting ~0.84 where ~0.80 occurred). Confident false positives are pages that slid then
stabilised — regression to the mean, which no feature here separates from genuine decay.
Confident false negatives looked stable for ninety days and dropped anyway, driven by competitor
moves, SERP layout changes or algorithm updates that the panel cannot observe.

**On the unseen-client row.** Absolute precision drops (0.900 → 0.600) but its base rate is also
far lower (0.562 → 0.295), so in lift terms it does not degrade: 2.03× versus 1.60×. Both
readings are true and they answer different questions; quoting either alone would mislead.

## 6. Interpretation

For a lane called Ranking Signal Analysis, the observed answer is that **how a page has recently
been ranking predicts where it goes next far better than what the page is made of.** The top four
features are all ranking behaviour; word count, character count, CPC and backlinks contribute
almost nothing.

**Negative results worth stating.** Content age — the basis of the most common refresh heuristic
in SEO — carries little weight once ranking momentum is present. The "expand thin content"
intuition finds no support here either: content-property features are near the bottom of the
importance ranking.

**The most useful surprise was a negative one about staleness.** "How long since we last touched
it" — the premise behind refresh flags — returned a **FALSE** verdict. `content_updated_date` in
`dim_content` is a live snapshot, so 81.8% of pages carry an update date *after* the decision
moment, making the field future information; and on the 19,403 pages that are testable the decline
gradient is 0.586 against a 0.567 base. Ordering a refresh queue by age is measurably worse than
chance at the top (precision 0.460 at K=50 against a 0.562 base). Ordering by observed movement
holds 0.82–0.90 across the same range.

The client-relative variant also failed to earn its place: P@50 of 0.860 (below the plain forest's
0.880) while making concentration worse — 3 distinct clients in the top 50 against 5.

**And the honest headline:** a transparent four-condition rule captures most of what the forest
finds, and beats it outright deeper in the queue. On the sealed month the model leads at P@50 (0.900 vs 0.780); on the development window the rule leads from K=500 outward (0.916 vs 0.844). That is a real win and a small
one, and the rule remains a legitimate fallback.

## 7. Recommendation

Work the capped queue top-down. Each row's action is chosen by *which evidence fired*, not by
score height:

| Action | Evidence | What to do | Pages |
|---|---|---|---:|
| `review_now` | already sliding **and** on page 1–2 | open it, find what changed, refresh if warranted | 53,978 |
| `investigate_volatility` | unstable, no clear direction | look for technical or SERP-layout causes *before* touching content | 20,622 |
| `defend_position` | visible, shallow, currently stable | leave the content alone; watch weekly | 14,222 |
| `monitor` | everything else | nothing this cycle | 27,834 |

**Three rules ship with the queue.**

1. **Cap each client at 5 pages per 50.** Uncapped, one client takes 40% of the top 50 and only 6
   portfolios appear. The cap costs **16 points of precision (0.900 → 0.740)** and buys a queue
   **13 portfolio owners** can each act on. That is a judgement call and is stated as one.
2. **Abstain where skill was never measured.** Six of 28 measured clients fall below AUC 0.60 and
   are served no queue. A queue that is wrong for a portfolio is worse than no queue.
3. **Check before acting.** Did the page change, or did the SERP? Is the decline in queries you
   care about? Is there a technical cause? Reason codes are the model's argument, so a reviewer
   can disagree with a specific claim rather than with "the model".

**Never automate off this score:** no auto-rewrite, no auto-deindex or delete, no client-facing
performance reporting or billing, no probability language, no causal claim.

**Monitoring.** Four triggers with thresholds: base rate outside 0.45–0.65; rolling P@50 below
0.65 for two cycles; any served client's AUC below 0.60; median position or volatility drifting
more than 25% from the training window. **Two of these fired on the sealed window** — median
position +29.4%, volatility +30.4% — and are reported rather than tuned away. Most of that is
panel composition (262k content items in January to 409k in June, newer pages ranking deeper) and
precision held, but the disclosure stands and the model needs refitting on a matching composition.

**Confidence:** directional and modest. Quarterly retrain regardless; the fallback to the frozen
rule is real and costs little.

## 8. Reproducibility

From a fresh clone:

```bash
pip install -r requirements.txt
pip install duckdb pyarrow          # warehouse access
export HF_TOKEN=hf_...              # READ token, gate accepted first
python work/scripts/build_modeling_frame.py --sealed
```

Then run, in order: `work/notebooks/w03_data_contract.ipynb`, `w04_baseline_score.ipynb`,
`w05_model.ipynb`, `w06_validation_audit.ipynb`, `w07_action_playbook.ipynb`, `capstone.ipynb`.
Heavy warehouse scans cache to `work/outputs/` on first run, so reruns are cheap; delete that
folder to force fresh scans.

**Seed:** `SEED = 20260808`, fixed in the frame builder and every notebook.
**Environment:** pandas 3.0.3, numpy 2.5.1, scikit-learn 1.9.0, duckdb 1.5.5. Tree-ensemble
figures move a point or two across library versions; the ~1.5× lift over base rate is the stable
claim, not the third decimal.

**Sealed-evaluation receipts, both committed:**
- the frame builder — `work/scripts/build_modeling_frame.py`, committed **before** the sealed
  frame was read, and producing dev and sealed frames from the same code so they cannot drift;
- the metrics it produced — `work/outputs/w06_sealed_test_metrics.json`.

Every number in this report traces to a committed JSON in `work/outputs/`:
`w03_data_contract.json`, `w04_baseline_metrics.json`, `w05_model_metrics.json`,
`w06_sealed_test_metrics.json`, `w07_playbook_metrics.json`.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — <https://flyrank.ai>. The warehouse release is a
pseudonymized export prepared for this internship; all analysis, errors and opinions here are
mine. Track leads: Mirza Ašćerić (ML) and Hole (data engineering).
