"""
LangGraph StateGraph for the multimodal fitness agent.

Graph topology
--------------
START → router
router  →  (conditional fan-out based on query_type)
    greeting                                      → generation (no retrieval)
    factual_retrieval                             → text_retrieval + progression_analysis
    cross_modal                                   → image_retrieval + text_retrieval
    analytical                                    → text_retrieval + progression_analysis
    personalized_followup (no injury keywords)    → text_retrieval + progression_analysis
    personalized_followup (with injury keywords)  → text_retrieval + progression_analysis + injury_lookup
All retrieval nodes → context_fusion → generation → END.

injury_lookup ONLY fires when the query explicitly mentions pain/injury
terminology (see _INJURY_KEYWORD_PATTERN). Every node appends its own name
to tool_calls_log so callers can audit the full execution path from
AgentState.tool_calls_log.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.agent.router import QueryRoute, QueryRouter
from src.agent.state import AgentState
from src.agent.tools import InjuryMemoryTool, StrengthProgressionTool
from src.config import cfg, user_profile
from src.retrieval.clip_classifier import (
    NN_EXACT_MATCH_THRESHOLD,
    classify_exercise as _clip_classify,
)
from src.retrieval.search import (
    _client as _chroma_client,
    get_images_by_exercise_label,
    search_exercise_by_text,
    search_lift_records_by_text,
    search_similar_exercise_image,
)

_CHROMA_PATH = cfg.chroma.persist_path

# Pre-warm the ChromaDB client at import time so the SharedSystemClient
# global registry is populated *before* LangGraph forks parallel nodes that
# all want to read from the same persistent DB.  Without this, the parallel
# fan-out (text_retrieval + progression_analysis + injury_lookup) races on
# `chromadb._identifier_to_system` and one of the threads dies with
# `Could not connect to tenant default_tenant`.
_chroma_client(_CHROMA_PATH)

# ── LLM prompts ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are FitSupport, a personal training coach. Follow these rules strictly:\n\n"
    "GREETING / INTRODUCTION RULES:\n"
    "- If the user asks what you can do, who you are, or says hello/hi/hey, "
    "respond with exactly this style (keep the line breaks and the bullet list):\n\n"
    "  I'm FitSupport, your personal training assistant. I know your workout "
    "history, personal bests, and injury context. Ask me things like:\n"
    "  - What weight should I use for hack squat today?\n"
    "  - My knee hurts, what can I train?\n"
    "  - [upload an image] What exercise is this and what's my best on it?\n"
    "  - I did chest yesterday, what should I train today?\n\n"
    "- Do not retrieve, invent, or reference any context for greetings. "
    "Just introduce yourself.\n\n"
    "SAFETY RULES (never break these):\n"
    "- If user mentions pain, stiffness, or discomfort today, "
    "always reduce weight by 15-20% or suggest an alternative\n"
    "- Never recommend an exercise the user already did today\n"
    "- Always acknowledge the injury/discomfort explicitly first\n\n"
    "KNEE PAIN RULES — if user mentions knee pain/hurt/sore today:\n"
    "- NEVER recommend: leg press, squats, hack squat, lunges, "
    "or any knee-dominant movement\n"
    "- ALWAYS recommend: upper body work, seated cable rows, lat pulldowns, "
    "hip thrusts (if pain is only at knee flexion), lying leg curls at light "
    "weight, upper body push/pull movements\n"
    "- Say explicitly, without starting the response with 'Given your': \"with your knee pain today, let's keep legs out of "
    "it entirely. Here's what you can do instead: [upper body options]\"\n\n"
    "CONTEXT RULES:\n"
    "- If user already did squats + leg press today, their quads are fatigued "
    "— do not recommend more quad-dominant work\n"
    "- Suggest what's MISSING from today's session, not more of the same\n\n"
    "STYLE RULES:\n"
    "- Maximum 3-4 sentences per response (greetings are the only exception — "
    "use the exact greeting block above)\n"
    "- Never use bullet points or numbered lists (except inside the greeting block)\n"
    "- Lead with the actual recommendation immediately\n"
    "- NEVER start your response with 'Given your recent performance' "
    "or any variation of it. Never start with 'Given your...'. "
    "Vary your opening naturally. Examples: "
    "'Your hack squat is at 140kg x 6 so...', "
    "'Based on where you're at with hack squat...', "
    "'You hit 140kg last session —', "
    "'For hack squat today,', "
    "'Looking at your progression,'. "
    "Start mid-thought like a coach would, not like a report.\n"
    "- Reference their real numbers casually e.g. 'you hit 140kg last time so try 142.5kg today'\n"
    "- Never say 'No injury conflicts were found'\n"
    "- Never say 'based on retrieved information'\n"
    "- Never end with generic disclaimers\n"
    "- Sound like you actually know them\n\n"
    "EXAMPLE:\n"
    "User: 'did squats, leg press, hamstring curl, knee stiff'\n"
    "BAD: 'do back squat at 142.5kg'\n"
    "GOOD: 'quads and hamstrings are covered, knee is stiff so skip anything heavy. "
    "Maybe finish with seated calf raises or some light hip thrusts focusing on the stretch "
    "— keep it under 100kg today given the knee.'"
)


# ── Greeting & injury keyword detection ────────────────────────────────────────

_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|hola|howdy|greetings|"
    r"good\s+(morning|afternoon|evening|day))\b",
    re.IGNORECASE,
)
_CAPABILITY_PATTERN = re.compile(
    r"\b("
    r"what\s+can\s+you\s+do|"
    r"who\s+are\s+you|"
    r"what\s+are\s+you|"
    r"how\s+do\s+you\s+work|"
    r"how\s+can\s+you\s+help|"
    r"what\s+do\s+you\s+do|"
    r"introduce\s+yourself|"
    r"tell\s+me\s+about\s+yourself"
    r")\b",
    re.IGNORECASE,
)
_INJURY_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"pain|painful|hurt|hurts|hurting|sore|soreness|stiff|stiffness|"
    r"injury|injured|injuries|discomfort|ache|aches|aching|"
    r"strain|strained|sprain|sprained|tweak|tweaked|"
    r"avoid|careful|"
    r"bad\s+(knee|shoulder|back|hip|elbow|wrist|ankle|neck)"
    r")\b",
    re.IGNORECASE,
)


def _is_greeting_query(query: str) -> bool:
    """Return True for pure greetings/capability questions only.

    Allows simple combinations like "hey what can you do?", but rejects
    queries that add a real task, e.g. "hey what is this exercise?".
    """
    q = (query or "").strip()
    if not q:
        return False
    q_clean = re.sub(r"[!?.\s]+", " ", q.lower()).strip()
    q_clean = re.sub(r"[,;:]+", " ", q_clean)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    greeting_terms = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "hola",
        "howdy",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
    }
    if q_clean in greeting_terms:
        return True
    if _CAPABILITY_PATTERN.fullmatch(q_clean):
        return True
    greeting_prefixes = sorted(greeting_terms, key=len, reverse=True)
    for greeting in greeting_prefixes:
        if q_clean.startswith(f"{greeting} "):
            remainder = q_clean[len(greeting) :].strip()
            return bool(_CAPABILITY_PATTERN.fullmatch(remainder))
    return False


def _has_injury_keywords(query: str) -> bool:
    """Return True if the query explicitly mentions pain/injury terminology."""
    return bool(_INJURY_KEYWORD_PATTERN.search(query or ""))

# ── Runtime overrides ──────────────────────────────────────────────────────────
# Use set_active_model() / set_top_k_override() to change at runtime (UI, eval).

_active_model: str | None = None
_top_k_override: int | None = None
_GEMINI_SELECTORS = {"gemini", "google", "google-gemini"}
_OLLAMA_SELECTORS = {"qwen", "quen", "quinn", "ollama", "local"}

# Once Ollama is confirmed unreachable (first failed call), we skip further
# network attempts in the same process to avoid long timeouts in batch runs.
_ollama_confirmed_down: bool = False


def set_active_model(model: str | None) -> None:
    """Override the active LLM model. Pass None to revert to cfg.llm.primary_model."""
    global _active_model
    _active_model = model


def set_top_k_override(top_k: int | None) -> None:
    """Override the retrieval top_k. Pass None to revert to cfg.retrieval.top_k."""
    global _top_k_override
    _top_k_override = top_k


def _get_model() -> str:
    return _active_model if _active_model is not None else cfg.llm.primary_model


def _get_provider(model: str) -> str:
    selector = model.strip().lower()
    if selector in _GEMINI_SELECTORS or selector.startswith("gemini"):
        return "gemini"
    if _active_model is None:
        return cfg.llm.provider
    return "ollama"


def _get_provider_model(model: str, provider: str) -> str:
    selector = model.strip().lower()
    if provider == "gemini":
        if _active_model is None or selector in _GEMINI_SELECTORS:
            return cfg.llm.gemini_model
        return model
    if _active_model is None or selector in _OLLAMA_SELECTORS:
        return cfg.llm.primary_model
    return model


def _get_top_k() -> int:
    return _top_k_override if _top_k_override is not None else cfg.retrieval.top_k


# ── Ollama integration ─────────────────────────────────────────────────────────

def call_ollama(
    user_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str | None = None,
    timeout: int = 60,
) -> str:
    """Call the Ollama /api/chat endpoint and return the assistant response text.

    Pass ``model`` explicitly to switch between cfg.llm.primary_model and
    cfg.llm.secondary_model — this is the designated model-comparison helper.

    Parameters
    ----------
    user_prompt : str
        Fully-formatted user turn including all retrieved context.
    system_prompt : str
        System instructions placed before the user turn.
    model : str
        Ollama model tag, e.g. "llama3.1:8b" or "qwen2.5:7b".
    base_url : str | None
        Ollama base URL; defaults to cfg.llm.ollama_base_url.
    timeout : int
        Socket timeout in seconds (default 60).  Lower values let the eval
        harness fail fast when Ollama is unreachable.
    """
    _base = base_url or cfg.llm.ollama_base_url
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    global _ollama_confirmed_down
    if _ollama_confirmed_down:
        return f"[Ollama unavailable — start Ollama and retry. Endpoint: {_base}]"

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data["message"]["content"])
    except urllib.error.URLError as exc:
        # Connection refused / DNS failure — Ollama is not running at all.
        _ollama_confirmed_down = True
        return (
            f"[Ollama connection error — is Ollama running at {_base}?]\n"
            f"Error details: {exc}"
        )
    except (TimeoutError, OSError) as exc:
        # Model load timeout or transient OS error — don't permanently blacklist.
        return (
            f"[Ollama response timeout after {timeout}s — model may still be loading.]\n"
            f"Error details: {exc}"
        )
    except (KeyError, json.JSONDecodeError) as exc:
        return f"[Ollama response parse error: {exc}]"


def check_ollama(base_url: str | None = None) -> bool:
    """Return True if the Ollama endpoint is reachable (3-second probe)."""
    global _ollama_confirmed_down
    _base = base_url or cfg.llm.ollama_base_url
    req = urllib.request.Request(f"{_base}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3):
            _ollama_confirmed_down = False
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# ── Gemini integration ─────────────────────────────────────────────────────────

def call_gemini(
    user_prompt: str,
    system_prompt: str,
    model: str,
    timeout: int = 60,
) -> str:
    """Call Gemini's generateContent endpoint and return the response text."""
    api_key = os.getenv(cfg.llm.gemini_api_key_env)
    if not api_key:
        return (
            f"[Gemini API key missing — set {cfg.llm.gemini_api_key_env} "
            "in your local .env file.]"
        )

    payload = json.dumps(
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return f"[Gemini API error {exc.code}: {detail}]"
    except urllib.error.URLError as exc:
        return f"[Gemini connection error: {exc.reason}]"
    except (TimeoutError, OSError) as exc:
        return f"[Gemini response timeout after {timeout}s: {exc}]"
    except json.JSONDecodeError as exc:
        return f"[Gemini response parse error: {exc}]"

    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(str(part.get("text", "")) for part in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        return f"[Gemini response parse error: {exc}]"


def _build_user_prompt(state: AgentState) -> str:
    """Assemble the user-turn prompt from all accumulated context fields in state."""

    def _fmt(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "None"

    # Greetings bypass the retrieval template ONLY when no image was uploaded.
    # An uploaded image always takes priority — even with a simple greeting text.
    if state.get("query_type") == "greeting" and not state.get("image_path"):
        return (
            f"The user said: {state['query']!r}\n\n"
            "Follow the GREETING / INTRODUCTION RULES in your system prompt. "
            "Introduce yourself as FitSupport using the exact example list. "
            "Do not invent workout numbers, weights, or retrieval results."
        )

    # If CLIP confidence was too low, short-circuit: just relay the note to the
    # user.  Do not attempt any weight recommendation without a known exercise.
    image_note = (state.get("image_identification_note") or "").strip()
    if image_note:
        return (
            f"Query: {state['query']}\n\n"
            f"Image identification result: {image_note}\n\n"
            "Relay this message to the user exactly as written. "
            "Do not guess an exercise name or suggest weights."
        )

    parts: list[str] = []

    history = state.get("conversation_history") or []
    if history:
        history_lines = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )
        parts.append(
            f"Previous conversation:\n{history_lines}\n\n"
            f"The user is now asking: {state['query']}\n\n"
            "This is a follow-up - do not repeat previous advice. "
            "If you already recommended hack squat, don't recommend it again. "
            "Answer the NEW question directly using the conversation context above."
        )

    image_context = state.get("retrieved_image_context", []) if state.get("show_images") else []
    matched_exercise_name = (state.get("matched_exercise_name") or "").strip()

    # Known working weights derived from the user's lift history — used as
    # additional context when ChromaDB returns no progression records for the
    # identified exercise (e.g. due to name mismatches between image labels
    # and lift-record exercise names).
    _KNOWN_WEIGHTS: dict[str, int] = {
        "Hack Squat": 140,
        "Leg Press": 230,
        "Hip Thrust": 160,
        "Barbell Squat": 105,
        "Dead Lift": 80,
        "Bench Press": 35,
        "Incline Machine Press": 30,
        "Lat Pulldown": 72,
    }

    if matched_exercise_name:
        confidence = state.get("exercise_confidence")
        conf_str = f" (confidence: {confidence:.0%})" if confidence is not None else ""
        parts.append(
            f"The user uploaded an image. Based on visual analysis, they appear to be "
            f"performing: {matched_exercise_name}{conf_str}. "
            "Use this exact exercise name when responding — do not guess a different exercise."
        )
        # Inject known working weight when progression_context won't cover it
        known_weight = _KNOWN_WEIGHTS.get(matched_exercise_name)
        if known_weight is not None and not (state.get("progression_context") or []):
            parts.append(
                f"User's current working weight for {matched_exercise_name}: {known_weight} kg."
            )

    # Progression context — use targeted data when an exercise was image-identified.
    # If the image was identified but no lift records exist, tell the LLM explicitly.
    progression_ctx = state.get("progression_context", []) or []
    is_cross_modal = state.get("query_type") == QueryRoute.CROSS_MODAL.value
    if matched_exercise_name and is_cross_modal and not progression_ctx:
        progression_display = [
            f"No personal best recorded yet for {matched_exercise_name}."
        ]
    else:
        progression_display = progression_ctx

    parts.append(
        f"Query: {state['query']}\n\n"
        f"Retrieved exercises:\n{_fmt(state.get('retrieved_text_context', []))}\n\n"
        f"User's personal bests for relevant exercises:\n"
        f"{_fmt(progression_display)}\n\n"
        f"Use these exact numbers when recommending weight. "
        f"If bench press best is 35 kg × 6, recommend 35–37.5 kg. "
        f"Always reference actual numbers, never say 'increase gradually'.\n\n"
        f"Injury context:\n{_fmt(state.get('injury_context', []))}\n\n"
        f"Image context:\n{_fmt(image_context)}\n\n"
        "Provide a specific, personalized recommendation."
    )

    return "\n\n".join(parts)


# ── private helpers ────────────────────────────────────────────────────────────

def _extract_docs(records: list[dict[str, Any]]) -> list[str]:
    """Return the document string from each retrieval record."""
    return [str(r.get("document", "")) for r in records if r.get("document")]


# ── node functions ─────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify the query and write query_type into state.

    Detects pure greetings/capability questions first (no retrieval needed),
    otherwise delegates to the rule-based QueryRouter so routing is
    deterministic and reproducible across evaluation runs.
    """
    t0 = time.perf_counter()
    query = state["query"]
    image_path = state.get("image_path")

    # Greetings/capability questions with no image attached short-circuit
    # to generation — no retrieval is needed and any retrieved context
    # would just confuse the introduction response.
    if not image_path and _is_greeting_query(query):
        return {
            "query_type": "greeting",
            "node_timings": {"router": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["router"],
        }

    routed = QueryRouter().route(query, image_path=image_path)
    return {
        "query_type": routed.route.value,
        "node_timings": {"router": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["router"],
    }


def text_retrieval_node(state: AgentState) -> dict:
    """Retrieve top-k exercise documents and the demo image(s) for the top hit.

    After ranking text documents, we look up the top exercise's demo images
    (start/mid/finish frames) by exact ``exercise_label`` match.  This lets
    the UI render the matching exercise visually without requiring the user
    to upload an image first.
    """
    t0 = time.perf_counter()
    records = search_exercise_by_text(
        state["query"],
        top_k=_get_top_k(),
        chroma_path=_CHROMA_PATH,
    )

    image_docs: list[str] = []
    if state.get("query_type") != QueryRoute.CROSS_MODAL.value:
        for rec in records:
            exercise_name = str((rec.get("metadata") or {}).get("exercise_name", "")).strip()
            if not exercise_name:
                continue
            image_records = get_images_by_exercise_label(
                exercise_name, chroma_path=_CHROMA_PATH, limit=3
            )
            if image_records:
                image_docs.extend(_extract_docs(image_records))
                break

    return {
        "retrieved_text_context": _extract_docs(records),
        "retrieved_image_context": image_docs,
        "node_timings": {"text_retrieval": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["text_retrieval"],
    }


def image_retrieval_node(state: AgentState) -> dict:
    """Identify the exercise in an uploaded image, then retrieve matching docs.

    New flow (CLIP zero-shot classification):
      1. Run clip_classifier.classify_exercise() on the uploaded image.
      2. If confidence >= 0.25:
           - Fetch demo images from ChromaDB by exact exercise_label match.
           - Fetch exercise text docs by exercise name for generation context.
           - Store matched_exercise_name + exercise_confidence in state so
             context_fusion_node can do a targeted lift-record lookup.
      3. If confidence < 0.25:
           - Fall back to the original image-embedding nearest-neighbour search.

    When no image is uploaded the node falls back to a text-based image lookup.
    """
    t0 = time.perf_counter()
    image_path = state.get("image_path")

    # ── No image uploaded: look up demo images for the query text ─────────────
    if not image_path:
        records = search_exercise_by_text(state["query"], top_k=1, chroma_path=_CHROMA_PATH)
        image_docs: list[str] = []
        for rec in records:
            exercise_name = str((rec.get("metadata") or {}).get("exercise_name", "")).strip()
            if not exercise_name:
                continue
            img_recs = get_images_by_exercise_label(exercise_name, chroma_path=_CHROMA_PATH, limit=3)
            image_docs.extend(_extract_docs(img_recs))
            break
        return {
            "retrieved_image_context": image_docs,
            "show_images": True,
            "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["image_retrieval"],
        }

    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {
            "retrieved_image_context": [],
            "show_images": True,
            "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["image_retrieval"],
        }

    _CONFIDENCE_THRESHOLD = 0.25

    # ── Step 1a: NN image-embedding search (runs before CLIP to check for near-
    # exact matches — e.g. user uploaded one of the indexed demo frames) ────────
    nn_records = search_similar_exercise_image(path, top_k=1, chroma_path=_CHROMA_PATH)
    if nn_records:
        nn_top = nn_records[0]
        nn_score = float(nn_top.get("score", 0))
        nn_meta = nn_top.get("metadata") or {}
        nn_exercise = str(
            nn_top.get("exercise_name")
            or nn_meta.get("exercise_name")
            or nn_meta.get("exercise_label")
            or ""
        ).strip()
        if nn_score >= NN_EXACT_MATCH_THRESHOLD and nn_exercise:
            # Near-exact image match — skip CLIP and trust the NN result directly.
            # This handles the case where the user uploads an image that is
            # essentially identical to an indexed demo frame.
            img_recs = get_images_by_exercise_label(nn_exercise, chroma_path=_CHROMA_PATH, limit=3)
            text_recs = search_exercise_by_text(nn_exercise, top_k=_get_top_k(), chroma_path=_CHROMA_PATH)
            return {
                "retrieved_text_context": _extract_docs(text_recs),
                "retrieved_image_context": _extract_docs(img_recs),
                "show_images": True,
                "matched_exercise_name": nn_exercise,
                "exercise_confidence": nn_score,
                "image_identification_note": None,
                "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
                "tool_calls_log": ["image_retrieval"],
            }
    else:
        nn_score, nn_exercise = 0.0, ""

    # ── Step 1b: CLIP zero-shot classification ─────────────────────────────────
    try:
        identified_exercise, confidence = _clip_classify(path)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[image_retrieval] CLIP classifier failed (%s); falling back to embedding search", exc
        )
        identified_exercise, confidence = "", 0.0

    # ── Step 2a: High-confidence CLIP path ────────────────────────────────────
    if identified_exercise and confidence >= _CONFIDENCE_THRESHOLD:
        # Fetch demo images by exact exercise_label (e.g. "Hack Squat")
        img_recs = get_images_by_exercise_label(
            identified_exercise, chroma_path=_CHROMA_PATH, limit=3
        )
        # Fetch exercise text docs so the generation node gets exercise details
        text_recs = search_exercise_by_text(
            identified_exercise, top_k=_get_top_k(), chroma_path=_CHROMA_PATH
        )
        return {
            "retrieved_text_context": _extract_docs(text_recs),
            "retrieved_image_context": _extract_docs(img_recs),
            "show_images": True,
            "matched_exercise_name": identified_exercise,
            "exercise_confidence": confidence,
            "image_identification_note": None,
            "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["image_retrieval"],
        }

    # ── Step 2b: Low-confidence — fall back to image-embedding NN search ──────
    # Reuse the nn_records already fetched in Step 1a.
    records = nn_records if nn_records else search_similar_exercise_image(path, top_k=1, chroma_path=_CHROMA_PATH)

    if not records:
        return {
            "retrieved_image_context": [],
            "show_images": True,
            "matched_exercise_name": "",
            "exercise_confidence": None,
            "image_identification_note": (
                "The uploaded image could not be matched to any exercise in your library "
                "(no results returned). Please describe the exercise or upload a clearer image."
            ),
            "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["image_retrieval"],
        }

    top = records[0]
    fallback_score = float(top.get("score", 0))
    metadata = top.get("metadata") or {}
    fallback_exercise = str(
        top.get("exercise_name")
        or metadata.get("exercise_name")
        or metadata.get("exercise_label")
        or ""
    ).strip()

    if fallback_score < _CONFIDENCE_THRESHOLD:
        return {
            "retrieved_image_context": [],
            "show_images": False,
            "matched_exercise_name": "",
            "exercise_confidence": None,
            "image_identification_note": (
                f"The uploaded image could not be confidently identified "
                f"(confidence: {max(confidence, fallback_score):.0%}). "
                "Please describe the exercise or upload a clearer image."
            ),
            "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
            "tool_calls_log": ["image_retrieval"],
        }

    return {
        "retrieved_image_context": _extract_docs(records),
        "show_images": True,
        "matched_exercise_name": fallback_exercise,
        "exercise_confidence": None,
        "image_identification_note": None,
        "node_timings": {"image_retrieval": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["image_retrieval"],
    }


def injury_lookup_node(state: AgentState) -> dict:
    """Load injury-memory records that match body-part terms in the query."""
    t0 = time.perf_counter()
    result = InjuryMemoryTool().run(state["query"], top_k=_get_top_k())
    return {
        "injury_context": _extract_docs(result.records),
        "node_timings": {"injury_lookup": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["injury_lookup"],
    }


def progression_analysis_node(state: AgentState) -> dict:
    """Retrieve personal strength-progression records via a ChromaDB metadata filter.

    Filters ``fitness_text`` by ``record_type == "lift_record"`` so the search
    runs over the user's personal strength corpus only — the generic exercise
    library can never crowd out their actual numbers.

    Falls back to the CSV-based ``StrengthProgressionTool`` if ChromaDB returns
    nothing (e.g. before the index has been built).
    """
    t0 = time.perf_counter()
    records = search_lift_records_by_text(
        state["query"],
        top_k=_get_top_k(),
        chroma_path=_CHROMA_PATH,
    )
    if not records:
        result = StrengthProgressionTool().run(state["query"], top_k=_get_top_k())
        records = result.records

    return {
        "progression_context": _extract_docs(records),
        "node_timings": {"progression_analysis": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["progression_analysis"],
    }


def context_fusion_node(state: AgentState) -> dict:
    """Join point after all parallel retrieval branches complete.

    For cross_modal queries with a confident image match, runs a targeted
    exact-filter lift-record lookup here (rather than in progression_analysis_node,
    which runs in parallel with image_retrieval_node and therefore cannot read
    the matched_exercise_name that image_retrieval_node writes).
    """
    t0 = time.perf_counter()
    result: dict = {
        "node_timings": {"context_fusion": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["context_fusion"],
    }

    matched = (state.get("matched_exercise_name") or "").strip()
    is_cross_modal = state.get("query_type") == QueryRoute.CROSS_MODAL.value
    if matched and is_cross_modal:
        # Exact lookup: only this exercise's lift records, matched by name.
        records = search_lift_records_by_text(
            matched,
            top_k=_get_top_k(),
            chroma_path=_CHROMA_PATH,
            exercise_name=matched,
        )
        if records:
            result["progression_context"] = _extract_docs(records)
        # If empty, _build_user_prompt will emit the "no personal best" message.

    result["node_timings"]["context_fusion"] = (time.perf_counter() - t0) * 1000
    return result


def generation_node(state: AgentState) -> dict:
    """Call the configured LLM to generate a personalised fitness coaching response.

    Uses _get_model() which respects any active model override, defaulting to
    cfg.llm.primary_model.  All four context buckets from AgentState are
    injected into the user prompt so the LLM has full retrieval context.
    """
    t0 = time.perf_counter()
    model = _get_model()
    provider = _get_provider(model)
    provider_model = _get_provider_model(model, provider)
    user_prompt = _build_user_prompt(state)
    if provider == "gemini":
        response = call_gemini(user_prompt, _SYSTEM_PROMPT, provider_model, timeout=300)
    else:
        # 300 s covers cold-start model loading (4-5 GB models can take 2-3 min).
        response = call_ollama(user_prompt, _SYSTEM_PROMPT, provider_model, timeout=300)
    return {
        "final_response": response,
        "node_timings": {"generation": (time.perf_counter() - t0) * 1000},
        "tool_calls_log": ["generation"],
    }


# ── conditional routing ────────────────────────────────────────────────────────

def route_by_query_type(state: AgentState) -> list[str]:
    """Map query_type to the set of retrieval nodes to activate (fan-out).

    Routing rules (Issue #2 fix — injury_lookup only fires on injury keywords):
      - greeting                                  → ["generation"]   (skip ALL retrieval)
      - factual_retrieval                         → text_retrieval + progression_analysis
      - cross_modal                               → image_retrieval + text_retrieval
      - analytical                                → text_retrieval + progression_analysis
      - personalized_followup w/ injury keywords  → text_retrieval + progression_analysis + injury_lookup
      - personalized_followup w/o injury keywords → text_retrieval + progression_analysis

    Returns a list so LangGraph runs all listed nodes in parallel within the
    same superstep.  Nodes share state via the ADD reducer.
    """
    qt = state["query_type"]
    query = state.get("query", "")

    if qt == "greeting":
        return ["generation"]
    if qt == QueryRoute.CROSS_MODAL:
        return ["image_retrieval", "text_retrieval"]
    if qt == QueryRoute.FACTUAL_RETRIEVAL:
        return ["text_retrieval", "progression_analysis"]
    if qt == QueryRoute.ANALYTICAL:
        return ["text_retrieval", "progression_analysis"]
    if qt == QueryRoute.PERSONALIZED_FOLLOWUP:
        if _has_injury_keywords(query):
            return ["text_retrieval", "progression_analysis", "injury_lookup"]
        return ["text_retrieval", "progression_analysis"]
    return ["text_retrieval", "progression_analysis"]


# ── graph construction ─────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Construct, wire, and compile the fitness LangGraph StateGraph.

    Returns the compiled graph object, ready to call .invoke() on.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("text_retrieval", text_retrieval_node)
    graph.add_node("image_retrieval", image_retrieval_node)
    graph.add_node("injury_lookup", injury_lookup_node)
    graph.add_node("progression_analysis", progression_analysis_node)
    graph.add_node("context_fusion", context_fusion_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_by_query_type)

    for retrieval_node in (
        "text_retrieval",
        "image_retrieval",
        "injury_lookup",
        "progression_analysis",
    ):
        graph.add_edge(retrieval_node, "context_fusion")

    graph.add_edge("context_fusion", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


# Module-level compiled graph — import and call .invoke() directly.
compiled_graph = build_graph()
graph = compiled_graph  # alias for ergonomic imports: from src.agent.graph import graph


def run_graph(
    query: str,
    image_path: str | None = None,
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """Run the compiled graph for a single query and return the final AgentState.

    Parameters
    ----------
    query : str
        User question.
    image_path : str | None
        Optional path to a query image for cross_modal queries.
    conversation_history : list[dict] | None
        Recent chat exchanges for follow-up context, e.g. the last 3 pairs of
        {"role": "user"|"assistant", "content": "..."} dicts.
    """
    initial: AgentState = {
        "query": query,
        "query_type": "",
        "image_path": image_path,
        "retrieved_text_context": [],
        "retrieved_image_context": [],
        "show_images": False,
        "matched_exercise_name": None,
        "exercise_confidence": None,
        "image_identification_note": None,
        "node_timings": {},
        "recall_at_3": None,
        "injury_context": [],
        "progression_context": [],
        "tool_calls_log": [],
        "final_response": "",
        "conversation_history": conversation_history or [],
    }
    t0 = time.perf_counter()
    state = compiled_graph.invoke(initial)
    node_timings = dict(state.get("node_timings", {}))
    node_timings["total"] = (time.perf_counter() - t0) * 1000
    state["node_timings"] = node_timings
    return state


def run_graph_with_model(
    query: str,
    model: str,
    image_path: str | None = None,
    top_k: int | None = None,
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """Run the graph with a specific LLM model (and optional top_k override).

    This is the recommended entry point for comparing cfg.llm.primary_model
    against cfg.llm.secondary_model.  Both overrides are always reset after
    the call, even if an exception is raised.

    Parameters
    ----------
    query : str
        User question.
    model : str
        Ollama model tag to use, e.g. cfg.llm.primary_model or
        cfg.llm.secondary_model.
    image_path : str | None
        Optional path to a query image.
    top_k : int | None
        Override retrieval depth; None keeps cfg.retrieval.top_k.
    conversation_history : list[dict] | None
        Recent chat exchanges for follow-up context.
    """
    set_active_model(model)
    set_top_k_override(top_k)
    try:
        return run_graph(query, image_path=image_path, conversation_history=conversation_history)
    finally:
        set_active_model(None)
        set_top_k_override(None)
