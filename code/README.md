# Buy or Wait? — AI-Powered Financial Decision Agent

**HackerRank Orchestrate (September 2026)** · Solo entry

An agentic, **neuro-symbolic** system that answers one question per request in
`dataset/requests.csv`: *can this user safely afford this expense now, with a plan,
later, or not at all?* For every request it outputs the safe pay amount,
affordability status, recommended payment method, a dated payment plan, the earliest
safe full-payment date, required spending changes, and a grounded explanation.

## Design Philosophy: Neuro-Symbolic Split

LLMs are excellent at **perception** and unreliable at **arithmetic**. This system
therefore confines all model calls to the perception edge and keeps every financial
decision in deterministic, verifiable Python:

| Layer | Technology | Responsibility |
|---|---|---|
| Perception | Gemini (vision + multilingual NLP) | Read receipt images; parse message overrides |
| Reasoning | Pure Python / pandas | Ledger reconstruction, 90-day simulation, plan ranking |

No LLM call ever performs arithmetic, date math, or plan selection.

## Pipeline Architecture

```
dataset/*.csv ──► Data Loader ──► Ledger Builder ──► 90-Day Forecast Engine
images/*.png  ──► Gemini Vision ─┘                        │
messages.csv  ──► Gemini NLP ────┘                        ▼
                                              Capacity Curve (max safe pay/day)
                                                          │
                              Plan Selector ◄─────────────┘
                       (full / partial / installments / wait)
                                                          ▼
                              Ranked decision ──► dataset/output.csv
```

1. **Data Loader** — ingests all CSVs, parses dates, de-duplicates exact duplicate
   event records, and normalizes currencies with the fixed dated exchange rates
   (closest rate on/before settlement date, inverse-pair fallback).
2. **Perception Layer (runtime AI)** —
   *Vision:* events with a blank amount are resolved via `images.csv` →
   `dataset/media/images/<id>.png`; Gemini extracts `{currency, amount}`.
   *NLP:* multilingual payroll/event messages are parsed into structured overrides
   (`salary_update` with effective date, `event_delay`). Pending/unconfirmed
   information (e.g., unapproved bonuses) is deliberately ignored.
3. **Ledger Builder** — drops `failed`/`cancelled`/`unrealized` records; reserves
   pending debits; ignores pending credits; detects recurrence **only when history
   supports it** (≥2 settled occurrences with regular gaps) and projects those
   occurrences across the forecast window unless already explicitly scheduled.
4. **90-Day Forecast Engine** — simulates the balance day-by-day from
   `request_date` to `request_date + 90d`, then computes a **capacity curve**:
   a reverse running minimum of `(balance − minimum_balance_to_keep)`, so
   `caps[i]` = the largest single payment safe on day `i`.
   * `amount_safe_to_pay = min(caps[0], requested_amount)`
   * `earliest_date_for_full_payment` = first day with `caps[i] ≥ requested`
     (empty when never safe inside the window).
   * `plan_is_safe()` re-simulates the balance with a candidate plan's payments
     subtracted; `STRICT_HORIZON` rejects payments outside the verifiable window
     (the "financially safer interpretation" rule).
5. **Plan Selector** — builds eligible candidates from the user's
   `payment_methods_user_will_consider`, `max_installment_months`, and the
   request's `allows_partial_payment`. Installment plans must exactly match a
   supplied option in `request_payment_options.csv`; partial plans are exactly two
   payments summing to the requested amount, completed by the deadline. Safe
   candidates are ranked by the official tie-breakers:
   **deadline completion → no spending changes → lowest total cost → earliest
   start → fewest payments → lowest `payment_option_id`.**
6. **Output Writer** — emits `dataset/output.csv` in the exact required schema,
   with sample-matching number formatting and templated, grounded explanations.

## Repository Layout

```
code/
├── main.py                  # Entry point (full pipeline)
├── README.md                # This file (copy)
├── evaluation/
│   └── usage_report.md      # Required token-usage report
├── image_cache.json         # Generated: cached vision extractions
├── msg_cache.json           # Generated: cached message parses
└── usage_stats.json         # Generated: live token accounting
dataset/                     # Inputs + media/ (not shipped in code.zip)
requirements.txt
README.md
```

## Setup

**Prerequisites:** Python 3.10+, pip, and (optionally) a Google AI API key.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets (repo root). NEVER commit this file.
#    Create `.env` with:
#      GEMINI_API_KEY=<your_key>
#      GEMINI_MODEL=gemini-3.6-flash   # optional override
```

## Run

```bash
python code/main.py
```

* Reads all inputs from `dataset/`.
* Writes predictions to `dataset/output.csv`.
* Writes token accounting to `code/usage_stats.json`.
* First run performs perception calls (rate-limit aware, with retries and sleeps);
  every result is cached, so **subsequent runs are fully deterministic and make
  zero API calls**.

**Graceful degradation:** with no API key and no caches, the pipeline still runs
end-to-end on structured data alone; perception-dependent fields simply fall back
to safe defaults.

## Security

* Secrets are read **only** from environment variables via `python-dotenv`.
* `.env` is excluded from version control (`.gitignore`) and from `code.zip`.
* Logs and caches never contain keys, tokens, or credentials.

## Token Usage & Cost

See `code/evaluation/usage_report.md` (final-run summary) and
`code/usage_stats.json` (live counters). Perception runs at `temperature=0` for
reproducibility.

## Known Limitations & Future Work

* Free-tier quota capped successful message parses on the first run; failed parses
  are cached as no-ops so runs never crash (quota-aware batching is future work).
* `spending_changes_needed` currently emits `none`; a search over stop/reduce
  combinations on flexible categories is the planned extension.
* Recurrence detection uses gap regularity heuristics rather than time-series
  decomposition.