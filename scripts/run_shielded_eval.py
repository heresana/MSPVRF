import os
import sys
import json
import time
import csv
import pandas as pd
from tqdm import tqdm

# Add the src/ directory to the Python path to import your pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pipeline import run_pipeline

# ── Configuration ────────────────────────────────────────────────────────
MODEL_NAME = "llama3.2:latest"

def run_shielded_eval():
    # ── Path Setup (Using Absolute Anchoring) ──────────────────────────────
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(PROJECT_ROOT, "data", "adversarial_suite")
    indirect_path = os.path.join(data_dir, "indirect_injections.json")
    direct_path = os.path.join(data_dir, "direct_jailbreaks.json")
    benign_path = os.path.join(data_dir, "benign_prompts.json")


    output_dir = os.path.join(PROJECT_ROOT, "results", "logs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "shielded_raw_outputs.csv")

    # ── Load and Slice Data (150 per file) ─────────────────────────────────
    all_prompts = []

    if os.path.exists(indirect_path):
        with open(indirect_path, 'r', encoding='utf-8') as f:
            indirect_data = json.load(f)
            all_prompts.extend(indirect_data)

    if os.path.exists(direct_path):
        with open(direct_path, 'r', encoding='utf-8') as f:
            direct_data = json.load(f)
            all_prompts.extend(direct_data)
    if os.path.exists(benign_path):
        with open(benign_path, 'r', encoding='utf-8') as f:
            benign_data = json.load(f)
            all_prompts.extend(benign_data)  # Cap at 150

    if not all_prompts:
        print("❌ Error: No datasets found in data/adversarial_suite/.")
        return

    print(f"Total dataset size : {len(all_prompts)} prompts (capped at 150 each)")
    print(f"Target Model       : {MODEL_NAME} (Shielded via GuardPrompt)")
    print("🚀 Starting Shielded Evaluation...\n")

    results = []

    # ── Initialize CSV with Headers ───────────────────────────────────────────
    # 🟢 NEW: Added redaction_reason to track the specific structural defenses
    fieldnames = [
        "model_name", "attack_type", "category", "user_intent", 
        "external_context", "model_output", "generation_time_sec", 
        "status", "plane1_decision", "plane2_decision", 
        "redaction_reason", "module1a_score", "module1b_score"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"✅ Ready to save incrementally to:\n   -> {os.path.abspath(output_file)}\n")

    # ── Main Eval Loop ────────────────────────────────────────────────────────
    for item in tqdm(all_prompts, desc=f"Testing {MODEL_NAME} (Shielded)"):
        attack_type = item.get("attack_type", "unknown")
        user_prompt = item.get("user_intent", "")
        context_data = item.get("external_context", "")
        source_cat = item.get("source", "unknown")

        try:
            start_time = time.time()
            
            # 🟢 FIX: Pass intent and context strictly as distinct parameters
            pipeline_response = run_pipeline(user_intent=user_prompt, external_context=context_data)
            
            generation_time = time.time() - start_time
            
            final_text = pipeline_response.get("final_output", "")
            ctx_data = pipeline_response.get("context", {})

            if "ERROR: TIMEOUT" in final_text:
                status = "Failed - Timeout"
            elif "ERROR:" in final_text:
                status = "Failed - Exception"
            else:
                status = "Success"

        except Exception as e:
            generation_time = time.time() - start_time
            final_text = f"PIPELINE EXCEPTION: {str(e)}"
            status = "Failed - Exception"
            ctx_data = {}

        # Extract the new specific redaction reason from Plane 2
        plane2_notes = ctx_data.get("plane2_notes", {})
        redaction_reason = plane2_notes.get("redaction_reason", "N/A")

        # ── Build and Save the Row Immediately ────────────────────────────────
        row = {
            "model_name": MODEL_NAME,
            "attack_type": attack_type,
            "category": source_cat,
            "user_intent": user_prompt,
            "external_context": context_data,
            "model_output": final_text,
            "generation_time_sec": round(generation_time, 3),
            "status": status,
            "plane1_decision": ctx_data.get("plane1_decision", "ERROR"),
            "plane2_decision": ctx_data.get("plane2_decision", "N/A"),
            "redaction_reason": redaction_reason,
            "module1a_score": ctx_data.get("score_1A", 0),
            "module1b_score": ctx_data.get("score_1B", 0.0)
        }

        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        results.append(row)

    # ── Final Summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n✅ Shielded evaluation complete! Saved to: {os.path.abspath(output_file)}\n")
    
    if not df.empty:
        p1_blocks = sum(df["plane1_decision"] == "BLOCK")
        p2_redacts = sum(df["plane2_decision"] == "REDACT")
        allowed = sum((df["plane1_decision"] != "BLOCK") & (df["plane2_decision"] == "ALLOW"))
        
        print("🛡️ GuardPrompt Intervention Summary:")
        print(f"  Total Prompts         : {len(df)}")
        print(f"  Blocked at Plane 1    : {p1_blocks}")
        print(f"  Redacted at Plane 2   : {p2_redacts}")
        print(f"  Passed through safely : {allowed}")
        print(f"\nAvg generation time (including firewall overhead): {df['generation_time_sec'].mean():.3f}s")
    else:
        print("⚠️ No results were processed to summarize.")

if __name__ == "__main__":
    run_shielded_eval()