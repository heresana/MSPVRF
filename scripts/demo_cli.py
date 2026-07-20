import time
import sys
import os
import json

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import run_pipeline

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("==================================================")
    print("             GuardPrompt Demo CLI                 ")
    print("==================================================")
    print(f"{Colors.RESET}")

def main():
    print_banner()
    print(f"{Colors.CYAN}Type 'quit' or 'exit' at any prompt to stop.{Colors.RESET}\n")

    while True:
        # 1. Gather Inputs
        print(f"{Colors.BOLD}--- New Request ---{Colors.RESET}")
        user_intent = input(f"{Colors.BLUE}Prompt > {Colors.RESET}").strip()
        
        if user_intent.lower() in ['quit', 'exit']:
            break
            
        print(f"{Colors.BLUE}External content (Press Enter to skip) > {Colors.RESET}", end="")
        external_context = input().strip()

        if external_context.lower() in ['quit', 'exit']:
            break

        print(f"\n{Colors.WARNING}⚙️  Processing through GuardPrompt Pipeline...{Colors.RESET}")
        
        # 2. Execute Pipeline
        start_time = time.time()
        try:
            response = run_pipeline(user_intent=user_intent, external_context=external_context)
        except Exception as e:
            print(f"\n{Colors.FAIL}Pipeline Exception: {str(e)}{Colors.RESET}\n")
            continue
            
        latency = time.time() - start_time
        
        final_output = response.get("final_output", "")
        ctx = response.get("context", {})
        
        p1_decision = ctx.get("plane1_decision", "UNKNOWN")
        p2_decision = ctx.get("plane2_decision", "SKIPPED")
        p2_notes = ctx.get("plane2_notes", {})
        
        print(f"\n{Colors.BOLD}📊 Execution Telemetry ({latency:.2f}s){Colors.RESET}")
        print("-" * 50)
        
        # Plane 1 Stats
        p1_color = Colors.FAIL if p1_decision == "BLOCK" else Colors.GREEN
        print(f"Plane 1 (Gatekeeper) : {p1_color}{p1_decision}{Colors.RESET}")
        if p1_decision != "UNKNOWN":
            print(f"  ├─ Lexical (1A)    : Score {ctx.get('score_1A', 0)} " 
                  f"(Cat: {ctx.get('matched_category', 'None')})")
            print(f"  └─ Semantic (1B)   : Score {ctx.get('score_1B', 0.0):.3f}")

        # Plane 2 Stats
        if p1_decision != "BLOCK":
            p2_color = Colors.FAIL if p2_decision == "REDACT" else Colors.GREEN
            print(f"Plane 2 (Firewall)   : {p2_color}{p2_decision}{Colors.RESET}")
            
            if p2_decision == "REDACT":
                reason = p2_notes.get("redaction_reason", "Unknown Policy Violation")
                print(f"  ├─ Redaction Reason: {Colors.FAIL}{reason}{Colors.RESET}")
                
            if external_context:
                l1_boundary = p2_notes.get("layer1", {}).get("boundary", {})
                shift = l1_boundary.get("hijack_score", 0.0)
                print(f"  └─ Attention Shift : {shift:.4f} " 
                      f"(> 0.20 indicates context hijack)")
        
        print("-" * 50)
        
        output_color = Colors.FAIL if (p1_decision == "BLOCK" or p2_decision == "REDACT") else Colors.GREEN
        print(f"{Colors.BOLD}🤖 Final Output:{Colors.RESET}")
        print(f"{output_color}{final_output}{Colors.RESET}\n")

if __name__ == "__main__":
    main()