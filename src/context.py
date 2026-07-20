"""
GuardPrompt - Shared Request Context
--------------------------------------
Single object that threads risk information from Plane 1 (Gatekeeper) through
the base LLM call into Plane 2 (Policy Auditor), so Plane 2 can use Plane 1's
suspicion level to decide when to escalate to Layer 4 (LLM-as-judge) instead
of treating every request as equally blank-slate.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestContext:
    prompt: str                                    # original user input
    deobfuscated_prompt: str = ""                  # after Module 1A's decode step

    # Plane 1 results
    score_1A: int = 0                              # 0 or 1
    matched_category: Optional[str] = None          # "override" | "roleplay" | None
    score_1B: float = 0.0                           # 0..1
    classifier_prob: float = 0.0
    roleplay_sim: float = 0.0
    plane1_decision: str = "PASS"                   # "BLOCK" | "FLAG" | "PASS"

    # Base LLM output - only populated if plane1_decision != "BLOCK"
    llm_output: Optional[str] = None

    # Plane 2 results - filled in by policy_checker / misalignment / firewall
    plane2_decision: Optional[str] = None           # "ALLOW" | "REDACT"
    escalated_to_judge: bool = False
    plane2_notes: dict = field(default_factory=dict)

    def to_log_dict(self) -> dict:
        """What gets written to results/logs. Drop 'prompt' here if you want
        lighter logs - scores and decisions are what matter for eval."""
        return {
            "score_1A": self.score_1A,
            "matched_category": self.matched_category,
            "score_1B": round(self.score_1B, 4),
            "plane1_decision": self.plane1_decision,
            "plane2_decision": self.plane2_decision,
            "escalated_to_judge": self.escalated_to_judge,
        }
