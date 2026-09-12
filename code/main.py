import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import json, re

import numpy as np
from datetime import timedelta

from google import genai

# --- 0. SETUP & SECURITY ---
# Load API key securely from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: No GEMINI_API_KEY found in .env file!")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "code" # We will save output.csv in the code/ folder as per standard

# --- 1. DATA LOADER LAYER ---
def load_all_data():
    """Loads all CSVs into a dictionary of Pandas DataFrames."""
    print("Loading datasets...")
    data = {
        "requests": pd.read_csv(DATASET_DIR / "requests.csv"),
        "profiles": pd.read_csv(DATASET_DIR / "financial_profiles.csv"),
        "events": pd.read_csv(DATASET_DIR / "financial_events.csv"),
        "payment_options": pd.read_csv(DATASET_DIR / "request_payment_options.csv"),
        "exchange_rates": pd.read_csv(DATASET_DIR / "exchange_rates.csv"),
        "messages": pd.read_csv(DATASET_DIR / "messages.csv"),
        "images": pd.read_csv(DATASET_DIR / "images.csv"),
    }
    
    # Convert date columns to datetime objects for easy math
    for df_name in ["requests", "events", "exchange_rates", "payment_options"]:
        for col in data[df_name].columns:
            if "date" in col:
                data[df_name][col] = pd.to_datetime(data[df_name][col], errors='coerce')
                
    return data

# --- 2. MULTIMODAL LAYER (VISION) ---
CACHE_PATH = BASE_DIR / 'code' / 'image_cache.json'
USAGE = {"model": "gemini-2.0-flash", "calls": 0, "input_tokens": 0, "output_tokens": 0}

def extract_amount_from_image(image_id):
    """Uses Gemini Vision to extract missing amounts from receipts (cached)."""
    if not image_id or client is None:
        return None, None
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    if image_id in cache:
        return cache[image_id]['amount'], cache[image_id]['currency']
    path = DATASET_DIR / 'media' / 'images' / f'{image_id}.png'
    if not path.exists():
        return None, None
    from google.genai import types
    resp = client.models.generate_content(
        model=USAGE["model"],
        contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type='image/png'),
                  'Extract the primary transaction amount and 3-letter currency code from this '
                  'document. Reply JSON only: {"currency": "XXX", "amount": 0.00}'],
        config=types.GenerateContentConfig(temperature=0.0))
    USAGE["calls"] += 1
    USAGE["input_tokens"] += getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
    USAGE["output_tokens"] += getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
    amt = cur = None
    m = re.search(r'\{.*\}', resp.text or '', re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            amt, cur = float(obj.get("amount")), obj.get("currency")
        except Exception:
            pass
    if amt is None:
        m2 = re.search(r'([A-Z]{3})\s*([\d][\d,]*(?:\.\d+)?)', resp.text or '')
        if m2:
            cur, amt = m2.group(1), float(m2.group(2).replace(',', ''))
    cache[image_id] = {"amount": amt, "currency": cur}
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return amt, cur
# --- HELPER: CURRENCY CONVERSION ---
def convert_currency(amount, from_curr, to_curr, date, rates_df):
    """Converts amount to home currency using the closest past exchange rate."""
    if from_curr == to_curr or pd.isna(amount):
        return amount
    
    # Find rates for this pair
    pair_rates = rates_df[
        (rates_df['from_currency'] == from_curr) & 
        (rates_df['to_currency'] == to_curr)
    ]
    
    if pair_rates.empty:
        # Try inverse rate if direct isn't found
        pair_rates = rates_df[
            (rates_df['from_currency'] == to_curr) & 
            (rates_df['to_currency'] == from_curr)
        ]
        if pair_rates.empty:
            return amount # Fallback if no rate exists
        # Get closest past rate
        valid_rates = pair_rates[pair_rates['rate_date'] <= date].sort_values('rate_date', ascending=False)
        if valid_rates.empty: return amount
        rate = valid_rates.iloc[0]['rate']
        return amount / rate # Inverse math
    else:
        valid_rates = pair_rates[pair_rates['rate_date'] <= date].sort_values('rate_date', ascending=False)
        if valid_rates.empty: return amount
        rate = valid_rates.iloc[0]['rate']
        return amount * rate

# --- 3. LEDGER BUILDER LAYER (v2) ---
def build_user_ledger(user_id, request_date, data, horizon=90):
    """Filters events and projects ALL recurring occurrences inside the 90-day window."""
    profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
    home_curr = profile['home_currency']
    rates = data['exchange_rates']
    end = request_date + timedelta(days=horizon)

    ev = data['events'][data['events']['user_id'] == user_id].copy()
    ev = ev[~ev['status'].isin(['failed', 'cancelled', 'unrealized'])]
    blank = ev['amount'].isna()
    if blank.any():
        img_map = dict(zip(data['images']['related_event_id'], data['images']['image_id']))
        for idx in ev.index[blank]:
            amt, cur = extract_amount_from_image(img_map.get(ev.at[idx, 'event_id']))
            if amt is not None:
                ev.at[idx, 'amount'] = amt
                if pd.isna(ev.at[idx, 'currency']) and cur:
                    ev.at[idx, 'currency'] = cur
    ev['amount_home'] = ev.apply(
        lambda r: convert_currency(r['amount'], r['currency'], home_curr, r['settlement_date'], rates), axis=1)

    # Future one-offs. Rule: reserve pending debits, ignore pending credits.
    future_events = ev[ev['settlement_date'] >= request_date].copy()
    future_events = future_events[~((future_events['direction'] == 'credit') & (future_events['status'] == 'pending'))]

    # Recurrence detection: only when history supports it (>=2 settled occurrences, regular gaps)
    past = ev[(ev['settlement_date'] < request_date) & (ev['status'] == 'settled')]
    future_descs = set(future_events['description'])
    projections = []
    for desc, g in past.groupby('description'):
        if desc in future_descs:
            continue  # already explicitly scheduled; don't double count
        g = g.sort_values('settlement_date')
        hist = g['settlement_date'].tolist()
        if len(hist) < 2:
            continue
        gaps = [(hist[i+1] - hist[i]).days for i in range(len(hist)-1)]
        if max(gaps) - min(gaps) > 10:
            continue  # irregular -> not recurring
        freq = max(1, int(round(sum(gaps) / len(gaps))))
        amt = g['amount_home'].mean()
        direction = g['direction'].iloc[0]
        nxt = hist[-1] + timedelta(days=freq)
        while nxt < request_date:
            nxt += timedelta(days=freq)
        while nxt <= end:
            projections.append({'date': nxt, 'amount': amt, 'direction': direction})
            nxt += timedelta(days=freq)
    return future_events, projections


# --- 4. THE 90-DAY FORECAST ENGINE (v2: balance curve + capacity curve) ---
def build_daily_balance(profile, ledger_data, request_date, horizon=90):
    """Simulates balance day-by-day WITHOUT the request payment."""
    future_events, projections = ledger_data
    deltas = [0.0] * (horizon + 1)

    fe = future_events[future_events['settlement_date'] <= request_date + timedelta(days=horizon)]
    for _, e in fe.iterrows():
        idx = (e['settlement_date'] - request_date).days
        deltas[idx] += -e['amount_home'] if e['direction'] == 'debit' else e['amount_home']
    for p in projections:
        idx = (p['date'] - request_date).days
        deltas[idx] += -p['amount'] if p['direction'] == 'debit' else p['amount']

    balances, bal = [], float(profile['current_available_balance'])
    for d in deltas:
        bal += d
        balances.append(bal)
    dates = [request_date + timedelta(days=i) for i in range(horizon + 1)]
    return dates, balances, float(profile['minimum_balance_to_keep'])


def capacity_curve(balances, min_bal):
    """caps[i] = max single payment safe on day i (balance never dips below min afterwards)."""
    caps, running = [0.0]*len(balances), float('inf')
    for i in range(len(balances)-1, -1, -1):
        running = min(running, balances[i] - min_bal)
        caps[i] = max(0.0, running)
    return caps


STRICT_HORIZON = True  # flip to False after calibration if samples show long installment plans


def plan_is_safe(payments, dates, balances, min_bal, request_date, horizon=90):
    """True if balance stays >= min_bal on every forecast day after applying plan payments."""
    bal = list(balances)
    for d, amt in payments:
        if d < request_date:
            return False
        idx = (d - request_date).days
        if idx > horizon:
            if STRICT_HORIZON:
                return False
            continue
        for t in range(idx, horizon+1):
            bal[t] -= amt
    return all(b >= min_bal - 1e-6 for b in bal)


# --- 5. PLAN SELECTOR & ORCHESTRATOR (v2) ---
def evaluate_plans(request, profile, forecast, options):
    (dates, balances, min_bal), caps = forecast
    req_date = request['request_date']
    deadline = request['desired_completion_date']
    requested = float(request['requested_amount'])
    horizon = len(dates) - 1

    considered = set(str(profile['payment_methods_user_will_consider']).split('|'))
    max_months = profile['max_installment_months']
    safe_today = round(min(caps[0], requested), 2)

    efp_date = next((dates[i] for i, c in enumerate(caps) if c >= requested - 1e-6), None)
    efp_str = efp_date.strftime('%Y-%m-%d') if efp_date is not None else ""

    candidates = []

    # FULL PAYMENT
    if 'full_payment' in considered and caps[0] >= requested - 1e-6:
        candidates.append(dict(method='full_payment', payments=[(req_date, requested)],
                               total=requested, start=req_date, option_id=None,
                               by_deadline=req_date <= deadline))

    # PARTIAL PAYMENT (exactly two payments)
    if ('partial_payment' in considered
            and str(request['allows_partial_payment']).lower() == 'true'
            and 0 < safe_today < requested
            and efp_date is not None and efp_date <= deadline):
        pays = [(req_date, safe_today), (efp_date, round(requested - safe_today, 2))]
        if plan_is_safe(pays, dates, balances, min_bal, req_date, horizon):
            candidates.append(dict(method='partial_payment', payments=pays, total=requested,
                                   start=req_date, option_id=None, by_deadline=True))

    # INSTALLMENTS (must match a supplied option + user preferences)
    if 'installments' in considered:
        for _, opt in options.iterrows():
            if opt['payment_method'] != 'installments':
                continue
            if pd.notna(max_months) and int(opt['number_of_payments']) > int(max_months):
                continue
            pays, d = [], opt['first_payment_date']
            for _ in range(int(opt['number_of_payments'])):
                pays.append((d, float(opt['payment_amount'])))
                d = d + timedelta(days=int(opt['payment_frequency_days']))
            if pays[0][0] < req_date:
                continue
            if not plan_is_safe(pays, dates, balances, min_bal, req_date, horizon):
                continue
            candidates.append(dict(method='installments', payments=pays,
                                   total=float(opt['total_payable_amount']), start=pays[0][0],
                                   option_id=opt['payment_option_id'], by_deadline=pays[-1][0] <= deadline))

    # WAIT (full payment becomes safe later)
    if 'full_payment' in considered and efp_date is not None and efp_date > req_date:
        candidates.append(dict(method='wait', payments=[(efp_date, requested)], total=requested,
                               start=efp_date, option_id=None, by_deadline=efp_date <= deadline))

        base = dict(amount_safe_to_pay=fmt(safe_today), spending_changes_needed='none',
                earliest_date_for_full_payment=efp_str)

    if not candidates:
        base.update(affordability_status='affordable_later' if efp_date else 'not_affordable',
                    recommended_payment_method='not_recommended', payment_plan='none',
                    decision_explanation=f"Safe capacity today is {safe_today} of {requested}; "
                                         f"full payment safe from {efp_str or 'beyond forecast'}.")
        return base

    # Official ranking: deadline, no changes, min total, earliest start, fewest payments, option id
    candidates.sort(key=lambda c: (0 if c['by_deadline'] else 1, round(c['total'], 2),
                                   c['start'], len(c['payments']), c['option_id'] or ''))
    best = candidates[0]

    status = {'full_payment': 'affordable_now', 'partial_payment': 'affordable_with_plan',
              'installments': 'affordable_with_plan', 'wait': 'affordable_later'}[best['method']]
    plan = 'none' if not best['payments'] else '|'.join(
        f"{d.strftime('%Y-%m-%d')}:{fmt(a)}" for d, a in best['payments'])
    base.update(affordability_status=status, recommended_payment_method=best['method'],
                payment_plan=plan,
                                  decision_explanation=explain(None, profile['home_currency'], requested,
                                                 min_bal, safe_today, [], efp_str))

    return base


# --- 6. MAIN EXECUTION LOOP ---
def main():
    data = load_all_data()
    
    # Prepare the output DataFrame based on the required columns
    output_columns = [
        "request_id", "amount_safe_to_pay", "affordability_status", 
        "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", 
        "spending_changes_needed", "decision_explanation"
    ]
    data['events'] = data['events'].drop_duplicates(
        subset=[c for c in data['events'].columns if c != 'event_id'])
    results = []
    (BASE_DIR / 'code' / 'usage_stats.json').write_text(json.dumps(USAGE, indent=2))
    print(f"Evaluating {len(data['requests'])} requests...")
    
    # Loop through every request
    for index, row in tqdm(data['requests'].iterrows(), total=len(data['requests'])):
        request_id = row['request_id']
        user_id = row['user_id']
        
        # 1. Get User Profile
        profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
        #2
        ledger_data = build_user_ledger(user_id, row['request_date'], data)
        #3
        forecast = (build_daily_balance(profile, ledger_data, row['request_date']),)
        #4
        forecast = (forecast[0], capacity_curve(forecast[0][1], forecast[0][2]))
        #5
        options = data['payment_options'][data['payment_options']['request_id'] == request_id]
        #6
        decision = evaluate_plans(row, profile, forecast, options)
        
        # Append to results
        result_row = {"request_id": request_id}
        result_row.update(decision)
        results.append(result_row)
        
    # Save to output.csv
    output_df = pd.DataFrame(results, columns=output_columns)
    output_path = DATASET_DIR / "output.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Success! Saved predictions to {output_path}")

# ---------------------------JSON Section -------------------------------
def fmt(x):
    
    x = round(float(x), 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip('0').rstrip('.')

def fmt_comma(x):
    x = round(float(x), 2)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}".rstrip('0').rstrip('.')

def _long_date(d):
    return f"{d.day} {d.strftime('%B %Y')}"

def explain(method, cur, req, min_bal, safe, payments, efp_str):
    C = lambda x: f"{cur} {fmt_comma(x)}"
    if method == 'full_payment':
        return f"Pay {C(req)} today. This leaves at least {C(min_bal)} available over the next 90 days."
    if method == 'installments':
        return f"Use {len(payments)} installments of {C(payments[0][1])}, starting {_long_date(payments[0][0])}. This leaves at least {C(min_bal)} available."
    if method == 'wait':
        return f"Wait until {_long_date(payments[0][0])}, then pay {C(req)} in full. Paying sooner would put the {C(min_bal)} minimum at risk."
    if method == 'partial_payment':
        return f"Pay {C(payments[0][1])} today and {C(payments[1][1])} on {_long_date(payments[1][0])}. This leaves at least {C(min_bal)} available."
    if efp_str:
        return f"Safe capacity today is {C(safe)} of {C(req)}; full payment becomes safe on {efp_str}, but no accepted payment method fits safely."
    return f"Safe capacity today is {C(safe)} of {C(req)}; full payment is not safe within the next 90 days."

if __name__ == "__main__":
    main()