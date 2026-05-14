"""Verify fitness_images index: counts, name mappings, and self-match confidence."""
from __future__ import annotations

from pathlib import Path

from src.retrieval.search import _client, search_similar_exercise_image
from src.config import cfg


def main() -> None:
    chroma_path = cfg.chroma.persist_path
    client = _client(chroma_path)

    # ── 1. fitness_images count and per-exercise breakdown ─────────────────────
    img_col = client.get_or_create_collection(cfg.chroma.image_collection)
    total = img_col.count()
    result = img_col.get(include=["metadatas"])
    metas = result.get("metadatas", []) or []

    exercise_counts: dict[str, int] = {}
    for m in metas:
        name = str(m.get("exercise_name") or m.get("exercise_label") or "(none)").strip()
        exercise_counts[name] = exercise_counts.get(name, 0) + 1

    print(f"fitness_images total records: {total}")
    print(f"Unique exercises indexed    : {len(exercise_counts)}")
    print()
    print("Per-exercise image counts:")
    for name, cnt in sorted(exercise_counts.items()):
        print(f"  {cnt:>2}  {name}")

    # ── 2. fitness_text exercise names ─────────────────────────────────────────
    text_col = client.get_or_create_collection(cfg.chroma.text_collection)
    text_result = text_col.get(include=["metadatas"])
    text_metas = text_result.get("metadatas", []) or []
    text_exercise_names: set[str] = set()
    for m in text_metas:
        n = str(m.get("exercise_name", "")).strip()
        if n:
            text_exercise_names.add(n)

    print()
    print(f"fitness_text exercise names ({len(text_exercise_names)}):")
    for n in sorted(text_exercise_names):
        print(f"  {n}")

    # ── 3. Folder → exercise_name mapping check ────────────────────────────────
    image_root = Path("data/raw/images")
    folder_names = sorted(f.name for f in image_root.iterdir() if f.is_dir())
    indexed_names_lower = {n.lower(): n for n in exercise_counts}

    print()
    print("Folder <-> indexed-name mapping:")
    mismatches: list[str] = []
    for folder in folder_names:
        expected = folder.replace("_", " ").title()
        if expected.lower() in indexed_names_lower:
            print(f"  OK  {folder!r:30s} → {expected!r}")
        else:
            # fuzzy: find closest
            candidates = [n for n in exercise_counts if folder.replace("_", " ") in n.lower()]
            hint = f" (closest: {candidates})" if candidates else ""
            print(f"  !!  {folder!r:30s} → {expected!r} NOT FOUND in index{hint}")
            mismatches.append(folder)

    # ── 4. Self-match test per exercise ────────────────────────────────────────
    print()
    print("Self-match confidence test:")
    all_pass = True
    for folder in folder_names:
        folder_path = image_root / folder
        img: Path | None = None
        for stem in ("mid", "start", "finish"):
            for ext in (".jpeg", ".jpg", ".png"):
                c = folder_path / (stem + ext)
                if c.is_file():
                    img = c
                    break
            if img:
                break
        if img is None:
            print(f"  SKIP  {folder}  (no image file found)")
            continue

        rows = search_similar_exercise_image(img, top_k=3)
        if not rows:
            print(f"  FAIL  {folder}  (no results returned)")
            all_pass = False
            continue

        top = rows[0]
        matched = str(
            top.get("exercise_name")
            or (top.get("metadata") or {}).get("exercise_name")
            or (top.get("metadata") or {}).get("exercise_label")
            or "?"
        ).strip()
        score = float(top.get("score", 0))
        expected = folder.replace("_", " ").title()
        score_ok = score >= 0.3
        name_ok = matched.lower() == expected.lower()
        status = "PASS" if (score_ok and name_ok) else "FAIL"
        if not (score_ok and name_ok):
            all_pass = False
        name_flag = f"  <-- NAME MISMATCH (got {matched!r})" if not name_ok else ""
        score_flag = "  <-- LOW SCORE" if not score_ok else ""
        print(
            f"  {status}  {folder:30s}  score={score:.3f}  expected={expected!r}"
            f"{name_flag}{score_flag}"
        )

    print()
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}): {mismatches}")
    else:
        print("No folder/name mismatches.")
    print("Self-match result:", "ALL PASS" if all_pass else "SOME FAILURES — review above")


if __name__ == "__main__":
    main()
