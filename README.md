# SecureGuard (MSPVRF)
### Dual-Plane Prompt Injection Defense Framework for Locally-Hosted LLMs

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-unlicensed-lightgrey.svg)
![Status](https://img.shields.io/badge/status-active--research-yellow.svg)

Summer Internship Project — CS-RVU-CY-SI-26-28
School of Computer Science and Engineering, RV University, Bangalore

---

## Problem

LLM applications are vulnerable to prompt injection, jailbreaks, and policy-evasion attacks that bypass safety alignment. Our own baseline testing on locally-hosted open-weight models confirms this at scale:

| Target Model | Attack Vector | Attack Success Rate (ASR) |
|---|---|---|
| Llama 3.2 (3B) | Direct Jailbreak | 94.8% |
| Llama 3.2 (3B) | Indirect Injection | 98.5% |
| DeepSeek-R1 (1.5B) | Direct Jailbreak | 98.2% |
| DeepSeek-R1 (1.5B) | Indirect Injection | 98.5% |

Existing single-layer defenses (keyword filters, isolated classifiers) leave significant residual risk, and defenses are typically studied in isolation rather than as integrated, cross-layer architectures. GuardPrompt addresses this gap with a two-plane pipeline that inspects both the **input** before inference and the **output** after generation.

## Architecture — MSPVRF

```
INPUT SURFACES (User Prompt / RAG Context)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ PLANE 1: Pre-Inference Input Gatekeeper                   │
│                                                             │
│  Module 1A: Lexical Filter                                 │
│   - Regex blacklists (override, roleplay, RAG-poison)       │
│   - Base64 / hex de-obfuscation                             │
│   - Binary OR decision → score_1A ∈ {0,1}                  │
│                                                             │
│  Module 1B: Semantic Classifier                             │
│   - MiniLM sentence embeddings (384-d)                      │
│   - Calibrated Logistic Regression → P(malicious)           │
│   - Roleplay cosine-similarity check                        │
│   - Fusion: score_1B = max(prob, 0.9 × roleplay_sim)         │
│                                                             │
│  Orchestrator: score_1A = 1 → immediate BLOCK.               │
│  Otherwise, 3-way threshold on score_1B:                     │
│   > 0.85 BLOCK | 0.60–0.85 FLAG | < 0.60 PASS                │
└───────────────────────────────────────────────────────────┘
        │ PASS / FLAG
        ▼
   BASE LLM EXECUTION (Llama 3.2 / DeepSeek, via Ollama)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ PLANE 2: Post-Inference Output Firewall                    │
│                                                             │
│  Module 2A: Compliance Evaluator                            │
│   - Injection/boundary checks (prompt–output drift)          │
│   - Jailbreak checks (canary token leak, harmful regex)      │
│   - Persona checks (leak regex + drift)                      │
│   - Conditional LLM-as-Judge (narrow authority-override case)│
│                                                             │
│  Module 2B: Data Sanitization Layer                          │
│   - Fallback template injection on flag                      │
│   - Redaction reason logged (priority-ranked)                │
│   - System-prompt hashing before log                         │
└───────────────────────────────────────────────────────────┘
        │
   ┌────┴─────┐
   ▼          ▼
 ALLOW      REDACT
(safe text) (fallback template)
```

**Design note:** `user_intent` and `external_context` (RAG-style content) are threaded separately through the pipeline all the way to the point of LLM generation — never pre-concatenated — so that indirect injection detection remains structurally possible at every layer.

## Results (Llama 3.2, 3B)

| Configuration | Attack Type | FC (%) | PB (%) | SNR (%) | ASR (%) |
|---|---|---|---|---|---|
| Unprotected Baseline | Direct Jailbreak | 38.5 | 1.2 | 0.0 | **39.8** |
| Unprotected Baseline | Indirect Injection | 35.0 | 2.5 | 0.0 | **37.5** |
| Shielded (MSPVRF) | Direct Jailbreak | 26.2 | 0.0 | 0.0 | **26.2** |
| Shielded (MSPVRF) | Indirect Injection | 0.0 | 0.0 | 0.0 | **0.0** |

- Indirect Injection ASR: 37.5% → **0.0%**
- Direct Jailbreak ASR: 39.8% → **26.2%**
- Benign False Positive Rate: 24.5% → **18.0%**
- Net latency: **−0.85s** (2.02s → 1.16s) — driven by early blocking of malicious prompts

**Known limitations:**
- Benign/encoded prompts (e.g. base64, hex) see a 2–3x latency penalty under the current pipeline
- 18% residual FPR and 26.2% residual direct-jailbreak ASR remain
- Drift/harmful-content detection relies on cosine similarity and regex — weaker, more evadable signals than a learned classifier
- Plane 2 currently has no unified risk score; the decision is a boolean OR across sub-checks
- Plane 2 is force-overridden to PASS when `external_context` is empty — the firewall currently only actively gates RAG-style requests (non-RAG evaluation gap, on the roadmap)

## Datasets

**Training & Validation (Module 1B):**
- `deepset/prompt-injections` — 662 samples (546 train / 116 test, EN+DE); shorter, mixed-language prompts (mean length ~19 words)
- `jackhhao` (jailbreak-classification) — larger, longer-form roleplay/persona-jailbreak samples (mean length ~203 words)
- Synthetic — custom-generated context-poisoning samples simulating indirect injection via hidden instructions embedded in support tickets, chat logs, etc. (mean length ~30 words, low variance)
- Combined pool spans three stylistically distinct sources; vocabulary overlap analysis between deepset and jackhhao shows a Jaccard similarity of 0.082 (987 shared words out of 2,022 / 10,932 unique malicious-class vocab respectively) — the low overlap indicates the sources are diverse enough that merging them should improve classifier generalization rather than introduce redundancy

**Evaluation Benchmark (locked, zero leakage):**
- JailbreakBench (direct) — 200 prompts sampled from 200 behaviors, NeurIPS 2024
- BIPIA (indirect) — 200 prompts sampled from 86,250 across 5 real-world scenarios, 50 attack types
- Synthetic Benign — 200 prompts
- Total: 600 prompts per model, held blind until final baseline evaluation

## Repository Structure

```
GuardPrompt/
├── data/
│   ├── adversarial_suite/     # direct jailbreaks, indirect injections, prompt leaking, benign queries for FPR measurement
│   └── classifier_splits/     # train / val / test for Module 1B  
├── src/
│   ├── config.py
│   ├── gatekeeper.py          # Plane 1 orchestration
│   ├── pattern_filter.py      # Module 1A: regex, Base64/hex de-obfuscation
│   ├── intent_classifier.py   # Module 1B: embedding + classifier inference
│   ├── preprocess_module1b.py # embedding generation for training/eval
│   ├── firewall.py            # Plane 2 orchestration
│   ├── context.py             # RequestContext (user_intent / external_context)
│   ├── models.py              # OllamaClient / VLLMClient
│   └── pipeline.py            # Input → Gatekeeper → Base LLM → Firewall
├── scripts/
│   ├── train_intent_model.py  # trains Module 1B classifier
│   ├── tune_thresholds.py     # sweeps val set for per-layer thresholds
│   ├── run_baseline_eval.py   # unshielded evaluation
│   ├── run_shielded_eval.py   # full-pipeline evaluation
│   ├── demo_gui.py            # Streamlit demo
│   └── demo_cli.py            # CLI demo with colored telemetry
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_result_analysis.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Requirement Specification

**Software:**
- Python 3.10+
- Ollama (local LLM runtime, GGUF quantized models)
- scikit-learn (Module 1B classifier), sentence-transformers (MiniLM embeddings)
- pandas / NumPy for data prep
- Jupyter Notebook + Matplotlib for exploration and results analysis

**Hardware:**
- GPU recommended (≥8GB VRAM) for LLM inference + classifier training
- 16 GB RAM, ~10–15 GB storage per model (1B–7B parameter range)

## Setup

```bash
git clone https://github.com/<your-username>/GuardPrompt.git
cd GuardPrompt
pip install -r requirements.txt

# pull the base model via Ollama
ollama pull llama3.2
```

## Usage

Train the Module 1B classifier:
```bash
python scripts/train_intent_model.py --data_dir data/classifier_splits/processed --out_dir results/models
```

Run baseline (unshielded) evaluation:
```bash
python scripts/run_baseline_eval.py
```

Run shielded (full pipeline) evaluation:
```bash
python scripts/run_shielded_eval.py
```

Try the interactive demo:
```bash
streamlit run scripts/demo_gui.py
# or
python scripts/demo_cli.py
```

## Team

| Name | ID |
|---|---|
| Sana Tasneem | GL4035 |
| Rehan Mokashi | 25261008 |
| Laxmi M L | 2SD23IS023 |
| Aryan Kaushik | 04214813124 |

**Project Guide:** Prof. Sarasvathi V, Professor, School of Computer Science and Engineering, RV University, Bangalore

## References

1. P. Jaiswal, A. Pratap, S. Saraswati, H. Kasyap, and S. Tripathy, "Analysis of LLMs Against Prompt Injection and Jailbreak Attacks," arXiv, 2025.
2. A. Alzahrani, "PromptGuard: A Structured Framework for Injection Resilient Language Models," arXiv, 2025.
3. X. Sheng and Q. Jiang, "Threats and Defenses for Large Language Models: A Survey," arXiv, 2025.
4. N. Alshammari and M. Alsaleh, "Detecting Prompt Injection Attacks in Generative AI Systems: A Hybrid SIEM and One-Class SVM Framework," *Electronics*, 2026.
5. Alamsabi et al., "Embedding-Based Detection of Indirect Prompt Injection Attacks in Large Language Models Using Semantic Context Analysis," *Algorithms*, vol. 19, p. 92, 2026.
6. Podpora et al., "LLM Firewall Using Validator Agent for Prevention Against Prompt Injection Attacks," *Applied Sciences*, vol. 16, p. 85, 2025.
7. I. Maloyan and D. Namiot, "Prompt Injection Attacks on Agentic Coding Assistants: A Systematic Analysis of Vulnerabilities in Skills, Tools, and Protocol Ecosystems," arXiv, 2026.
8. Duarte et al., "A Systematic Review of Prompt Injection Attacks on Large Language Models: Trends, Taxonomy, Evaluation, Defenses, and Opportunities," *IEEE Access*, vol. 14, 2026.

## Scope

**In scope:** text-based defense for locally-hosted open-weight LLMs (1.5B–8B), covering direct jailbreaks, roleplay-wrapped jailbreaks, and indirect (context-embedded) injections.

**Out of scope (this phase):** multimodal (image) attacks, agentic/MCP-based attack surfaces, adaptive white-box attacks.
