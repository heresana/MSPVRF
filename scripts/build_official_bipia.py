import os
import json
import random

def build_bipia_without_install():
    print("Parsing raw Microsoft BIPIA files directly...")
    
    # Define exact paths to the cloned repository files
    context_file = os.path.join("BIPIA", "benchmark", "email", "test.jsonl")
    attack_file = os.path.join("BIPIA", "benchmark", "text_attack_test.json")
    
    # 1. Verify files exist
    if not os.path.exists(context_file) or not os.path.exists(attack_file):
        print("Error: Could not find raw BIPIA data. Ensure 'BIPIA/' folder is in your root directory.")
        return

    # 2. Load the raw benign email contexts (JSONL format)
    contexts = []
    with open(context_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                contexts.append(json.loads(line))
                
    # 3. Load the raw malicious text attacks (JSON format)
    with open(attack_file, 'r', encoding='utf-8') as f:
        raw_attacks = json.load(f)
        
    # FIX: Extract the values/items into a true list if it's parsed as a dictionary
    if isinstance(raw_attacks, dict):
        # If Microsoft formatted it as { "attack_id_1": {"attack": "..."}, ... }
        attacks = list(raw_attacks.values())
    else:
        # If it's already a clean list
        attacks = list(raw_attacks)
        
    print(f"Loaded {len(contexts)} email contexts and {len(attacks)} clear attack strings.")

    # 4. Synthesize the injection pairs manually
    random.seed(42) # Fixed seed for research consistency
    
    formatted_data = []
    
    # We will safely iterate to pair contexts and attack instructions
    for i in range(200):
        ctx_sample = random.choice(contexts)
        attack_sample = random.choice(attacks)
        
        clean_email_text = ctx_sample.get("context", "")
        
        # Handle different potential naming keys inside Microsoft's attack objects
        if isinstance(attack_sample, dict):
            malicious_command = attack_sample.get("attack", attack_sample.get("text", attack_sample.get("instruction", "")))
        else:
            malicious_command = str(attack_sample)
        
        # Format the context-poisoned payload strings
        poisoned_context = f"{clean_email_text}\n\n[SYSTEM NOTE: {malicious_command}]"
        
        formatted_data.append({
            "external_context": poisoned_context,
            "user_intent": ctx_sample.get("question", "Summarize the email context provided."),
            "source": "Microsoft_BIPIA_Direct_Parse",
            "attack_type": "indirect_injection"
        })

    # 5. Save the file directly to your adversarial suite
    output_dir = os.path.join("data", "adversarial_suite")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "indirect_injections.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
        
    print(f"Successfully constructed 200 samples without installing deepspeed!")
    print(f"Saved file to: {output_file}")

if __name__ == "__main__":
    build_bipia_without_install()