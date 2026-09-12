import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

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
def extract_amount_from_image(image_id):
    """Uses Gemini Vision to extract missing amounts from receipts."""
    # TODO: Implement image reading and Gemini API call
    # For now, return None so the script runs without crashing
    return None

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

# --- 3. LEDGER BUILDER LAYER ---
def build_user_ledger(user_id, request_date, data):
    """Filters events and projects recurring bills for the 90-day forecast."""
    profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
    home_curr = profile['home_currency']
    rates = data['exchange_rates']
    
    user_events = data['events'][data['events']['user_id'] == user_id].copy()
    
    # 1. Filter out failed, cancelled, and unrealized events
    user_events = user_events[~user_events['status'].isin(['failed', 'cancelled', 'unrealized'])]
    
    # 2. Convert all amounts to home currency
    user_events['amount_home'] = user_events.apply(
        lambda row: convert_currency(row['amount'], row['currency'], home_curr, row['settlement_date'], rates), 
        axis=1
    )
    
    # 3. Separate into Future One-Offs and Past History (for recurrence detection)
    future_events = user_events[user_events['settlement_date'] >= request_date].copy()
    
    # Rule: Do not count pending credits
    future_events = future_events[~((future_events['direction'] == 'credit') & (future_events['status'] == 'pending'))]
    
    # 4. Detect Recurring Expenses (The Hackathon Trick)
    # If an event happened in the 90 days prior, assume it happens again in the next 90 days.
    past_events = user_events[
        (user_events['settlement_date'] < request_date) & 
        (user_events['settlement_date'] >= request_date - timedelta(days=90)) &
        (user_events['status'] == 'settled') &
        (user_events['direction'] == 'debit')
    ]
    
    recurring_projections = []
    for desc, group in past_events.groupby('description'):
        if len(group) >= 1: # It happened recently, project it forward
            last_date = group['settlement_date'].max()
            avg_amount = group['amount_home'].mean()
            
            # Project to next occurrence (simple 30-day cycle for monthly bills)
            next_date = last_date + timedelta(days=30)
            while next_date < request_date:
                next_date += timedelta(days=30)
                
            if next_date < request_date + timedelta(days=90):
                recurring_projections.append({
                    'date': next_date, 
                    'amount': avg_amount, 
                    'direction': 'debit'
                })
                
    return future_events, recurring_projections

# --- 4. THE 90-DAY FORECAST ENGINE (THE MATH) ---
def calculate_90_day_forecast(profile, ledger_data, request_date):
    """Simulates 90 days to find the maximum safe payment today."""
    start_bal = profile['current_available_balance']
    min_bal = profile['minimum_balance_to_keep']
    future_events, recurring_projections = ledger_data
    
    # Track the lowest the balance drops relative to the minimum balance
    max_shortfall = 0.0
    running_bal = start_bal
    
    for day_offset in range(91):
        current_day = request_date + timedelta(days=day_offset)
        
        # 1. Apply Future One-Off Events for this day
        day_events = future_events[future_events['settlement_date'] == current_day]
        for _, ev in day_events.iterrows():
            if ev['direction'] == 'debit':
                running_bal -= ev['amount_home']
            else:
                running_bal += ev['amount_home']
                
        # 2. Apply Recurring Projections for this day
        for proj in recurring_projections:
            if proj['date'] == current_day:
                if proj['direction'] == 'debit':
                    running_bal -= proj['amount']
                    
        # 3. Check for shortfall
        if running_bal < min_bal:
            shortfall = min_bal - running_bal
            if shortfall > max_shortfall:
                max_shortfall = shortfall
                
    # The max safe to pay is the current buffer minus the worst shortfall we saw
    safe_buffer = start_bal - min_bal
    amount_safe_to_pay = max(0.0, safe_buffer - max_shortfall)
    
    return amount_safe_to_pay

# --- 5. PLAN SELECTOR & ORCHESTRATOR ---
def evaluate_plans(request, profile, amount_safe_to_pay, payment_options):
    """Decides the best payment method based on the safe amount."""
    req_amount = request['requested_amount']
    
    # Default fallback
    decision = {
        "amount_safe_to_pay": amount_safe_to_pay,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": f"Safe to pay {amount_safe_to_pay} today, but request is {req_amount}."
    }
    
    # Scenario 1: Can afford in full right now!
    if amount_safe_to_pay >= req_amount:
        decision["affordability_status"] = "affordable_now"
        decision["recommended_payment_method"] = "full_payment"
        decision["payment_plan"] = f"{request['request_date'].strftime('%Y-%m-%d')}:{req_amount}"
        decision["earliest_date_for_full_payment"] = request['request_date'].strftime('%Y-%m-%d')
        decision["decision_explanation"] = f"User has enough safe buffer to pay {req_amount} in full today."
        return decision

    # Scenario 2: Can afford installments?
    # Check if any payment option fits within the safe buffer
    for _, opt in payment_options.iterrows():
        if opt['payment_method'] == 'installments' and opt['payment_amount'] <= amount_safe_to_pay:
            # Build the installment plan string
            plan_str = []
            pay_date = opt['first_payment_date']
            for i in range(int(opt['number_of_payments'])):
                plan_str.append(f"{pay_date.strftime('%Y-%m-%d')}:{opt['payment_amount']}")
                pay_date += timedelta(days=int(opt['payment_frequency_days']))
                
            decision["affordability_status"] = "affordable_with_plan"
            decision["recommended_payment_method"] = "installments"
            decision["payment_plan"] = "|".join(plan_str)
            decision["decision_explanation"] = f"Cannot pay in full, but can afford {opt['payment_amount']} installments."
            return decision

    return decision


# --- 6. MAIN EXECUTION LOOP ---
def main():
    data = load_all_data()
    
    # Prepare the output DataFrame based on the required columns
    output_columns = [
        "request_id", "amount_safe_to_pay", "affordability_status", 
        "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", 
        "spending_changes_needed", "decision_explanation"
    ]
    results = []
    
    print(f"Evaluating {len(data['requests'])} requests...")
    
    # Loop through every request
    for index, row in tqdm(data['requests'].iterrows(), total=len(data['requests'])):
        request_id = row['request_id']
        user_id = row['user_id']
        
        # 1. Get User Profile
        profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
        
        # 2. Build Ledger (Data up to request_date)
        ledger_data = build_user_ledger(user_id, row['request_date'], data)
        
        # 3. Forecast 90 days (Returns the max safe amount to pay today)
        amount_safe = calculate_90_day_forecast(profile, ledger_data, row['request_date'])
        
        # 4. Get Payment Options for this request
        options = data['payment_options'][data['payment_options']['request_id'] == request_id]
        
        # 5. Evaluate and decide
        decision = evaluate_plans(row, profile, amount_safe, options)
        
        # Append to results
        result_row = {"request_id": request_id}
        result_row.update(decision)
        results.append(result_row)

    # Save to output.csv
    output_df = pd.DataFrame(results, columns=output_columns)
    output_path = DATASET_DIR / "output.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Success! Saved predictions to {output_path}")

if __name__ == "__main__":
    main()