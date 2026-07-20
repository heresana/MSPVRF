import os
import json
import time
import csv
import ollama
import pandas as pd
from tqdm import tqdm

def run_model_baseline():
    # ── Path Setup (Using Absolute Anchoring) ──────────────────────────────
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(PROJECT_ROOT, "data", "adversarial_suite")
    indirect_path = os.path.join(data_dir, "indirect_injections.json")
    direct_path = os.path.join(data_dir, "direct_jailbreaks.json")
    benign_path = os.path.join(data_dir, "benign_prompts.json")
    
    output_dir = os.path.join(PROJECT_ROOT, "results", "logs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "baseline_raw_outputs.csv")
    
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
            all_prompts.extend(benign_data)
    if not all_prompts:
        print("❌ Error: No datasets found in data/adversarial_suite/.")
        return

    models_to_test = [
        "llama3.2:latest"
    ]
    
    print(f"Total dataset size: {len(all_prompts)} prompts (capped at 150 each).")
    print(f"Target model list: {models_to_test}")
    print(f"Total matrix iterations: {len(all_prompts) * len(models_to_test)}\n")
    
    # ── Initialize CSV with Headers ────────────────────────────────────────
    fieldnames = [
        "model_name", "attack_type", "category", "user_intent", 
        "external_context", "model_output", "generation_time_sec", "status"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
    print(f"✅ Ready to save incrementally to:\n   -> {os.path.abspath(output_file)}\n")
    
    results = []
    
    # ── Main Eval Loop ─────────────────────────────────────────────────────
    for model_name in models_to_test:
        print(f"🚀 Starting evaluation loop for model: {model_name}")
        
        for item in tqdm(all_prompts, desc=f"Testing {model_name}"):
            attack_type = item.get("attack_type", "unknown")
            user_prompt = item.get("user_intent", "")
            context_data = item.get("external_context", "")
            source_cat = item.get("source", "unknown")
            
            if "indirect" in attack_type:
                combined_prompt = (
                    f"Instruction: {user_prompt}\n\n"
                    f"Source Document Context:\n\"\"\"\n{context_data}\n\"\"\""
                )
            else:
                combined_prompt = user_prompt
                
            try:
                start_time = time.time()
                
                response = ollama.generate(
                    model=model_name,
                    prompt=combined_prompt,
                    options={
                        'temperature': 0.0,
                        'num_predict': 1024
                    }
                )
                
                generation_time = time.time() - start_time
                generated_text = response.get('response', '').strip()
                
                if not generated_text:
                    status = "Failed - Empty Response"
                    generated_text = "NULL_OUTPUT_DETECTED"
                else:
                    status = "Success"
                    
            except Exception as e:
                generation_time = 0.0
                generated_text = f"RUNTIME EXCEPTION: {str(e)}"
                status = "Failed - Exception"

            # ── Build and Save the Row Immediately ─────────────────────────
            row = {
                "model_name": model_name,
                "attack_type": attack_type,
                "category": source_cat,
                "user_intent": user_prompt,
                "external_context": context_data,
                "model_output": generated_text,
                "generation_time_sec": round(generation_time, 3),
                "status": status
            }

            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            # Keep in memory for the final completion message
            results.append(row)

    print(f"\n✅ Multi-model baseline complete! Saved to:\n{os.path.abspath(output_file)}")

if __name__ == "__main__":
    run_model_baseline()