from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.router import QueryRoute, QueryRouter
from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder
from retrieval.search import search_exercise_by_text, search_similar_exercise_image


@dataclass
class ToolResult:
    tool_name: str
    records: list[dict[str, Any]]


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


class TextRetrievalTool:
    name = "text_retrieval"

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self.embedder = embedder

    def run(self, query: str, top_k: int = 3) -> ToolResult:
        return ToolResult(
            self.name,
            search_exercise_by_text(
                query,
                top_k=top_k,
                chroma_path=ROOT / "data/chroma",
                embedder=self.embedder,
            ),
        )


class ImageRetrievalTool:
    name = "image_retrieval"

    def __init__(self, embedder: ImageEmbedder | None = None) -> None:
        self.embedder = embedder

    def run(self, image_path: str | None, top_k: int = 3) -> ToolResult:
        if not image_path:
            return ToolResult(self.name, [])
        path = Path(image_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            return ToolResult(self.name, [])
        return ToolResult(
            self.name,
            search_similar_exercise_image(
                path,
                top_k=top_k,
                chroma_path=ROOT / "data/chroma",
                embedder=self.embedder,
            ),
        )


class InjuryMemoryTool:
    name = "injury_memory"

    def __init__(self, injuries_dir: Path | None = None) -> None:
        self.injuries_dir = injuries_dir or ROOT / "data/raw/injuries"

    def run(self, query: str, top_k: int = 3) -> ToolResult:
        records: list[dict[str, Any]] = []
        terms = [t for t in ("knee", "shoulder", "back", "hip", "pain", "injury", "recovery") if t in query.lower()]
        if not terms:
            terms = ["injury", "pain", "recovery"]

        for path in sorted(self.injuries_dir.glob("*")):
            if not path.is_file():
                continue
            payload = ""
            if path.suffix.lower() == ".json":
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    try:
                        payload = json.dumps(json.loads(text), ensure_ascii=False)
                    except json.JSONDecodeError:
                        payload = text
            elif path.suffix.lower() in {".txt", ".md", ".csv"}:
                payload = path.read_text(encoding="utf-8", errors="ignore")
            if payload and _contains_any(payload, terms):
                records.append(
                    {
                        "id": f"injury::{path.stem}",
                        "score": 1.0,
                        "document": payload[:500],
                        "metadata": {"source_path": str(path), "record_type": "injury_memory"},
                    }
                )

        if not records and "knee" in query.lower():
            records.append(
                {
                    "id": "injury::knee_default",
                    "score": 0.5,
                    "document": "No detailed knee note found; apply conservative knee-friendly filtering.",
                    "metadata": {"record_type": "injury_memory", "source_path": "default"},
                }
            )
        return ToolResult(self.name, records[:top_k])


class StrengthProgressionTool:
    name = "strength_progression"

    def __init__(self, strength_csv: Path | None = None) -> None:
        self.strength_csv = strength_csv or ROOT / "data/raw/lifts/strength.csv"
        self.exercise_library_csv = ROOT / "data/raw/metadata/exercise_library.csv"

    def _library_context(self) -> dict[str, str]:
        if not self.exercise_library_csv.is_file():
            return {}
        df = pd.read_csv(self.exercise_library_csv)
        context: dict[str, str] = {}
        for _, row in df.iterrows():
            exercise = str(row.get("exercise_name", "")).strip()
            if not exercise:
                continue
            context[exercise] = " ".join(
                str(row.get(col, "") or "")
                for col in ("movement_pattern", "equipment", "primary_muscles", "difficulty")
            )
        return context

    def run(self, query: str, top_k: int = 3) -> ToolResult:
        if not self.strength_csv.is_file():
            return ToolResult(self.name, [])
        df = pd.read_csv(self.strength_csv)
        terms = [t for t in query.lower().replace("-", " ").split() if len(t) > 2]
        library_context = self._library_context()
        records: list[dict[str, Any]] = []
        for i, row in df.iterrows():
            exercise = str(row.get("exercise_name", ""))
            notes = str(row.get("notes", ""))
            blob = f"{exercise} {notes} {library_context.get(exercise, '')}".lower()
            score = sum(1 for term in terms if term in blob)
            if "glute" in query.lower() and "glute" in blob:
                score += 2
            if "quad" in query.lower() and "quad" in blob:
                score += 2
            if "back" in query.lower() and ("lat" in blob or "row" in blob or "pull" in blob):
                score += 2
            if score > 0 or any(word in query.lower() for word in ("strength", "baseline", "pr", "progress")):
                records.append(
                    {
                        "id": f"strength::{i}",
                        "score": float(score),
                        "document": (
                            f"{exercise}: best_weight_kg={row.get('best_weight_kg', '')}, "
                            f"best_reps={row.get('best_reps', '')}, notes={notes}"
                        ),
                        "metadata": {
                            "record_type": "strength_progression",
                            "exercise_name": exercise,
                            "best_weight_kg": "" if pd.isna(row.get("best_weight_kg")) else str(row.get("best_weight_kg")),
                            "best_reps": "" if pd.isna(row.get("best_reps")) else str(row.get("best_reps")),
                            "source_path": str(self.strength_csv),
                        },
                    }
                )
        records.sort(key=lambda item: item["score"], reverse=True)
        return ToolResult(self.name, records[:top_k])


class FitnessToolRouter:
    def __init__(
        self,
        *,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.router = QueryRouter()
        self.text = TextRetrievalTool(embedder=text_embedder)
        self.image = ImageRetrievalTool(embedder=image_embedder)
        self.injury = InjuryMemoryTool()
        self.strength = StrengthProgressionTool()

    def run(self, query: str, image_path: str | None = None, top_k: int = 3) -> dict[str, Any]:
        routed = self.router.route(query, image_path=image_path)
        results: list[ToolResult] = []

        if routed.route == QueryRoute.FACTUAL_RETRIEVAL:
            results.append(self.text.run(query, top_k=top_k))
        elif routed.route == QueryRoute.CROSS_MODAL:
            results.append(self.text.run(query, top_k=top_k))
            results.append(self.image.run(image_path, top_k=top_k))
        elif routed.route == QueryRoute.ANALYTICAL:
            results.append(self.strength.run(query, top_k=top_k))
            results.append(self.text.run(query, top_k=top_k))
        elif routed.route == QueryRoute.PERSONALIZED_FOLLOWUP:
            if any(term in query.lower() for term in ("strength", "baseline", "pr", "profile")):
                results.append(self.strength.run(query, top_k=top_k))
                results.append(self.text.run(query, top_k=top_k))
                results.append(self.injury.run(query, top_k=top_k))
            else:
                results.append(self.text.run(query, top_k=top_k))
                results.append(self.injury.run(query, top_k=top_k))
                results.append(self.strength.run(query, top_k=top_k))

        flattened: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in results:
            for record in result.records:
                key = str(record.get("id", ""))
                if key and key in seen:
                    continue
                seen.add(key)
                record = {**record, "tool_name": result.tool_name}
                flattened.append(record)

        return {
            "route": routed.route.value,
            "rationale": routed.rationale,
            "tool_results": [r.__dict__ for r in results],
            "records": flattened[: max(top_k, 3) * 3],
        }

