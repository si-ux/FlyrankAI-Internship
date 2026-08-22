"""Build the modeling frame for the Ranking Signal Analysis lane.

Implements the ML-04 data contract (work/outputs/w03_data_contract.json) against the
warehouse release. One row per content item; features from a feature window, label from a
strictly later outcome window.

Two frames, built by the same code so they cannot drift apart:

  dev     features 2026-01-01..2026-03-31  ->  label 2026-04   (develop here)
  sealed  features 2026-03-01..2026-05-31  ->  label 2026-06   (touch ONCE, at the end)

The sealed frame exists so the final number has a receipt: this file is the frame-builder the
leakage skill asks to see committed.

Usage:
    python work/scripts/build_modeling_frame.py            # dev only
    python work/scripts/build_modeling_frame.py --sealed    # both

Requires a Hugging Face READ token with access to FlyRank/internship-warehouse, supplied via
the HF_TOKEN env var or the standard ~/.cache/huggingface/token file. Never hardcode it.
"""
from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import time

print = functools.partial(print, flush=True)

import duckdb

B = "hf://datasets/FlyRank/internship-warehouse"
SEED = 20260808  # fixed everywhere for reproducibility

# The two splits, exactly as the contract states them.
SPLITS = {
    "dev": {
        "feature_months": ["2026-01", "2026-02", "2026-03"],
        "first_month": "2026-01",
        "last_month": "2026-03",
        "outcome_month": "2026-04",
    },
    "sealed": {
        "feature_months": ["2026-03", "2026-04", "2026-05"],
        "first_month": "2026-03",
        "last_month": "2026-05",
        "outcome_month": "2026-06",
    },
}

DECLINE_THRESHOLD = 1.0   # positions; label = 1 when position worsens by >= this
MIN_FEATURE_IMPS = 100    # volume floor so feature-window position is meaningful
MIN_OUTCOME_IMPS = 30     # volume floor in the outcome month (uses outcome-window info: disclosed)


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "data/raw/content_refresh_anonymized.csv").exists():
            return p
    return pathlib.Path.cwd()


def hf_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    cached = pathlib.Path.home() / ".cache/huggingface/token"
    if cached.exists() and cached.read_text().strip():
        return cached.read_text().strip()
    sys.exit("No HF token found. Set HF_TOKEN or run `huggingface-cli login`.")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN ?)", [hf_token()])
    return con


def fact(*months: str) -> str:
    """hf:// has no brace-glob support — pass an explicit list of partition paths."""
    paths = ",".join(f"'{B}/fact_content_daily_performance/month={m}/*.parquet'" for m in months)
    return f"read_parquet([{paths}])"


def build_sql(split: dict) -> str:
    fm, first, last, out = (
        split["feature_months"], split["first_month"], split["last_month"], split["outcome_month"],
    )
    return f"""
WITH feat AS (
    SELECT
        content_hash_id,
        ANY_VALUE(client_hash_id)                                        AS client_hash_id,
        SUM(gsc_impressions)                                             AS f_impressions,
        SUM(gsc_clicks)                                                  AS f_clicks,
        SUM(gsc_sum_position) / NULLIF(SUM(gsc_impressions), 0)          AS f_pos,
        COUNT(DISTINCT report_date) FILTER (WHERE gsc_impressions > 0)   AS f_days_with_impressions,
        STDDEV_SAMP(gsc_avg_position) FILTER (WHERE gsc_impressions > 0) AS f_pos_volatility,
        -- direction BEFORE the cut: last feature month vs first feature month, each
        -- impression-weighted. Never spans the feature/outcome boundary.
        (SUM(gsc_sum_position) FILTER (WHERE month = '{last}')
             / NULLIF(SUM(gsc_impressions) FILTER (WHERE month = '{last}'), 0))
      - (SUM(gsc_sum_position) FILTER (WHERE month = '{first}')
             / NULLIF(SUM(gsc_impressions) FILTER (WHERE month = '{first}'), 0))
                                                                         AS f_pos_trend
    FROM {fact(*fm)}
    WHERE gsc_data_available IS TRUE          -- IS TRUE, not = TRUE: the flag is three-valued
    GROUP BY content_hash_id
),
outc AS (
    SELECT
        content_hash_id,
        SUM(gsc_impressions)                                    AS o_impressions,
        SUM(gsc_sum_position) / NULLIF(SUM(gsc_impressions), 0) AS o_pos
    FROM {fact(out)}
    WHERE gsc_data_available IS TRUE
    GROUP BY content_hash_id
)
SELECT
    f.content_hash_id,
    f.client_hash_id,
    f.f_impressions,
    f.f_clicks,
    f.f_pos,
    100.0 * f.f_clicks / NULLIF(f.f_impressions, 0)  AS f_ctr,
    f.f_days_with_impressions,
    f.f_pos_volatility,
    f.f_pos_trend,
    -- content metadata: static properties, knowable at the cut
    d.content_type,
    d.main_intent,
    d.competition_level,
    d.search_volume,
    d.competition,
    d.cpc,
    d.backlinks,
    d.word_count,
    d.char_count,
    d.keyword_token_count,
    DATE_DIFF('day', d.content_created_date, DATE '{last}-01' + INTERVAL 1 MONTH - INTERVAL 1 DAY)
                                                     AS content_age_days,
    -- Days since the page itself was last updated, measured AT the decision moment.
    -- This is a property of the content (knowable at the cut), not the platform's
    -- optimisation decision -- `last_optimized_date` stays excluded.
    DATE_DIFF('day', d.content_updated_date, DATE '{last}-01' + INTERVAL 1 MONTH - INTERVAL 1 DAY)
                                                     AS days_since_update,
    -- label block
    o.o_impressions,
    o.o_pos,
    o.o_pos - f.f_pos                                AS pos_delta,
    CASE WHEN o.o_pos - f.f_pos >= {DECLINE_THRESHOLD} THEN 1 ELSE 0 END AS is_position_decline
FROM feat f
JOIN outc o USING (content_hash_id)
LEFT JOIN read_parquet('{B}/dim_content.parquet') d USING (content_hash_id)
WHERE f.f_impressions >= {MIN_FEATURE_IMPS}
  AND o.o_impressions >= {MIN_OUTCOME_IMPS}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", action="store_true",
                    help="also build the sealed frame (June 2026 label). Use once.")
    ap.add_argument("--force", action="store_true", help="rebuild even if the parquet exists")
    args = ap.parse_args()

    root = repo_root()
    outdir = root / "work/outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    names = ["dev"] + (["sealed"] if args.sealed else [])
    con = connect()

    for name in names:
        target = outdir / f"modeling_frame_{name}.parquet"
        if target.exists() and not args.force:
            print(f"[{name}] exists, skipping -> {target.relative_to(root).as_posix()}")
            continue

        split = SPLITS[name]
        print(f"[{name}] features {split['feature_months']} -> label {split['outcome_month']}")
        t = time.time()
        df = con.sql(build_sql(split)).df()
        df.to_parquet(target, index=False)

        # Guard: the label must never be reconstructable from a feature column.
        leaky = [c for c in df.columns if c.startswith("o_") or c in ("pos_delta",)]
        feature_cols = [c for c in df.columns
                        if c not in leaky + ["is_position_decline", "content_hash_id",
                                             "client_hash_id"]]
        print(f"[{name}] {len(df):,} rows · {df.client_hash_id.nunique()} clients · "
              f"base rate {df.is_position_decline.mean():.4f} · {time.time()-t:.0f}s")
        print(f"[{name}] {len(feature_cols)} feature cols, {len(leaky)} label-side cols held back")
        print(f"[{name}] -> {target.relative_to(root).as_posix()}")

    print("done")


if __name__ == "__main__":
    main()
