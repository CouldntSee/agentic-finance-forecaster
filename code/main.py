import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
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
    for df_name in ["requests", "events", "exchange_rates"]:
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

# --- 3. LEDGER BUILDER LAYER ---
def build_user_ledger(user_id, request_date, data):
    """Filters events and calculates the user's actual financial state on request_date."""
    # TODO: Filter events for this specific user
    # TODO: Convert foreign currencies to home_currency using exchange_rates
    # TODO: Handle linked_event_id and blank amounts (call extract_amount_from_image)
    # TODO: Separate into: One-off past, One-off pending, Recurring bills, Confirmed income
    return {} 

# --- 4. THE 90-DAY FORECAST ENGINE (THE MATH) ---
def calculate_90_day_forecast(user_profile, ledger, request_date):
    """
    Simulates the user's bank balance day-by-day for 90 days.
    This is the most critical function in the entire hackathon.
    """
    # TODO: Start with current_available_balance
    # TODO: Loop day by day from request_date to request_date + 90 days
    # TODO: Subtract recurring bills, add confirmed salary
    # TODO: Check if balance ever drops below minimum_balance_to_keep
    # TODO: Return the daily balance array and the safe daily surplus
    return []

# --- 5. PLAN SELECTOR & ORCHESTRATOR ---
def evaluate_plans(request, profile, forecast, payment_options):
    """
    Determines amount_safe_to_pay, affordability_status, and best payment plan.
    Ranks plans based on the hackathon's strict tie-breaking rules.
    """
    # TODO: Calculate max safe to pay today
    # TODO: Check if full payment, partial, or installments are safe
    # TODO: Use Gemini here ONLY to generate the `decision_explanation` text
    return {
        "amount_safe_to_pay": 0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": "Skeleton output."
    }

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
        ledger = build_user_ledger(user_id, row['request_date'], data)
        
        # 3. Forecast 90 days
        forecast = calculate_90_day_forecast(profile, ledger, row['request_date'])
        
        # 4. Get Payment Options for this request
        options = data['payment_options'][data['payment_options']['request_id'] == request_id]
        
        # 5. Evaluate and decide
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

if __name__ == "__main__":
    main()