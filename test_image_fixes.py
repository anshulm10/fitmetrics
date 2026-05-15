"""
Smoke tests for the three image fixes:
  1. Greeting with image_path bypasses intro
  2. Low-confidence CLIP sets image_identification_note, clears matched name
  3. Exact progression lookup in context_fusion for cross_modal
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import router_node, route_by_query_type, _build_user_prompt
from src.agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    s: AgentState = {
        "query": "test",
        "query_type": "",
        "image_path": None,
        "retrieved_text_context": [],
        "exercise_context_records": [],
        "generation_retrieved_text": None,
        "retrieved_image_context": [],
        "show_images": False,
        "matched_exercise_name": None,
        "identified_exercise": None,
        "exercise_confidence": None,
        "image_identification_note": None,
        "node_timings": {},
        "recall_at_3": None,
        "injury_context": [],
        "progression_context": [],
        "tool_calls_log": [],
        "final_response": "",
        "conversation_history": [],
        "skip_injury_lookup": False,
    }
    s.update(overrides)
    return s


# ── Test 1: greeting + image_path → NOT a greeting route ──────────────────────
print("=== Test 1: greeting + image bypasses intro ===")
state = _base_state(query="hey", image_path="data/raw/images/bench_press/mid.png")
result = router_node(state)
assert result["query_type"] != "greeting", f"Expected non-greeting, got {result['query_type']}"
print(f"  query_type={result['query_type']}  PASS")

# Even if somehow query_type ended up as greeting, _build_user_prompt should not produce intro
state2 = _base_state(query="hey", query_type="greeting", image_path="data/raw/images/bench_press/mid.png")
prompt = _build_user_prompt(state2)
assert "GREETING / INTRODUCTION RULES" not in prompt, "Should not produce greeting prompt when image_path is set"
assert "Introduction" not in prompt.split("\n")[0], "Should not start with intro"
print(f"  _build_user_prompt skips greeting block when image_path set  PASS")

# ── Test 2: low-confidence note short-circuits prompt ─────────────────────────
print("\n=== Test 2: low-confidence note short-circuits prompt ===")
state3 = _base_state(
    query="what exercise is this?",
    query_type="cross_modal",
    image_path="data/raw/images/bench_press/mid.png",
    image_identification_note="The uploaded image could not be confidently matched (confidence: 0.18).",
)
prompt3 = _build_user_prompt(state3)
assert "could not be confidently matched" in prompt3, "Expected low-confidence message in prompt"
assert "personal bests" not in prompt3.lower(), "Should not include progression template"
print(f"  Low-confidence note present in prompt  PASS")
print(f"  No progression template leaked  PASS")

# ── Test 3: matched + empty progression → 'no personal best' message ──────────
print("\n=== Test 3: matched exercise with empty progression → no-personal-best message ===")
state4 = _base_state(
    query="what is this?",
    query_type="cross_modal",
    image_path="data/raw/images/bench_press/mid.png",
    matched_exercise_name="Bench Press",
    progression_context=[],
)
prompt4 = _build_user_prompt(state4)
assert "No personal best recorded yet for Bench Press" in prompt4, (
    f"Expected no-personal-best message, got prompt:\n{prompt4[:400]}"
)
print(f"  No-personal-best message injected  PASS")

# ── Test 4: matched + has progression → real numbers used, not fallback ───────
print("\n=== Test 4: matched exercise with real progression data ===")
state5 = _base_state(
    query="what is this?",
    query_type="cross_modal",
    image_path="data/raw/images/bench_press/mid.png",
    matched_exercise_name="Bench Press",
    progression_context=["Lift history for Bench Press. best_weight_kg=80.0, best_reps=5"],
)
prompt5 = _build_user_prompt(state5)
assert "No personal best" not in prompt5, "Should not show fallback when data present"
assert "80.0" in prompt5, "Should contain actual weight data"
print(f"  Real progression data included, no fallback  PASS")

# ── Test 5: greeting without image → produces intro ───────────────────────────
print("\n=== Test 5: greeting without image → intro produced ===")
state6 = _base_state(query="hey", query_type="greeting", image_path=None)
prompt6 = _build_user_prompt(state6)
assert "GREETING / INTRODUCTION RULES" in prompt6
print(f"  Greeting without image → intro prompt  PASS")

print("\nAll tests passed.")
