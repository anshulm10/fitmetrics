from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import chromadb


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import cfg
from embeddings.image_embedder import ImageEmbedder, build_image_records
from embeddings.text_embedder import TextEmbedder, build_text_records
from retrieval.search import search_exercise_by_text, search_similar_exercise_image


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def _prepare_collections(client: chromadb.PersistentClient):
    """Drop and recreate both ChromaDB collections for a clean rebuild."""
    for collection_name in (cfg.chroma.text_collection, cfg.chroma.image_collection):
        try:
            client.delete_collection(collection_name)
            _log("INDEX", f"reset stale collection={collection_name}")
        except Exception:
            pass
    text_col = client.get_or_create_collection(cfg.chroma.text_collection)
    image_col = client.get_or_create_collection(cfg.chroma.image_collection)
    return text_col, image_col


def _upsert_records(
    collection,
    records: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    include_documents: bool = True,
) -> int:
    if not records:
        return 0
    ids = [r["id"] for r in records]
    metadatas = [r["metadata"] for r in records]
    if include_documents:
        documents = [r["text"] for r in records]
    else:
        documents = [r["metadata"].get("source_path", "") for r in records]
    collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    return len(records)


def _verify_collection(collection, expected_count: int, collection_name: str) -> None:
    actual_count = collection.count()
    if actual_count != expected_count:
        raise RuntimeError(
            f"{collection_name} count mismatch: expected={expected_count} actual={actual_count}"
        )
    ids = collection.get(include=[])["ids"]
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise RuntimeError(f"{collection_name} has duplicate ids: {duplicate_count}")
    _log("INDEX", f"verified {collection_name}: count={actual_count} duplicate_ids=0")


def build_indexes() -> dict[str, int]:
    metadata_csv = ROOT / "data" / "raw" / "metadata" / "exercise_library.csv"
    lifts_dir = ROOT / "data" / "raw" / "lifts"
    images_root = ROOT / "data" / "raw" / "images"
    chroma_dir = cfg.chroma.persist_path
    chroma_dir.mkdir(parents=True, exist_ok=True)

    _log("EMBED TEXT", f"model={cfg.embeddings.text_model} source={metadata_csv}")
    text_records = build_text_records(metadata_csv=metadata_csv, lifts_dir=lifts_dir)
    text_embedder = TextEmbedder()
    text_embeddings = text_embedder.embed_texts([r["text"] for r in text_records])
    _log("EMBED TEXT", f"records={len(text_records)} vectors={len(text_embeddings)}")

    _log("EMBED IMAGE", f"model={cfg.embeddings.image_model} source={images_root}")
    image_records = build_image_records(images_root=images_root, metadata_csv=metadata_csv)
    image_embedder = ImageEmbedder()
    image_embeddings = image_embedder.embed_images([Path(r["image_path"]) for r in image_records])
    _log("EMBED IMAGE", f"records={len(image_records)} vectors={len(image_embeddings)}")

    _log("INDEX", f"path={chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    text_col, image_col = _prepare_collections(client)
    text_count = _upsert_records(text_col, text_records, text_embeddings, include_documents=True)
    image_count = _upsert_records(image_col, image_records, image_embeddings, include_documents=False)
    _log("INDEX", f"{cfg.chroma.text_collection}={text_count} {cfg.chroma.image_collection}={image_count}")
    _verify_collection(text_col, text_count, cfg.chroma.text_collection)
    _verify_collection(image_col, image_count, cfg.chroma.image_collection)

    _log("SEARCH", "text query: knee friendly quad exercise")
    rows_1 = search_exercise_by_text(
        "knee friendly quad exercise", chroma_path=chroma_dir, embedder=text_embedder
    )
    _log("SEARCH", f"top={rows_1[:3]}")

    _log("SEARCH", "text query: upper chest press")
    rows_2 = search_exercise_by_text("upper chest press", chroma_path=chroma_dir, embedder=text_embedder)
    _log("SEARCH", f"top={rows_2[:3]}")

    _log("SEARCH", "text query: lat focused back movement")
    rows_3 = search_exercise_by_text(
        "lat focused back movement", chroma_path=chroma_dir, embedder=text_embedder
    )
    _log("SEARCH", f"top={rows_3[:3]}")

    sample_image = next((images_root.rglob("*.jpeg")), None)
    if sample_image is None:
        sample_image = next((images_root.rglob("*.jpg")), None)
    if sample_image is None:
        sample_image = next((images_root.rglob("*.png")), None)
    if sample_image:
        _log("SEARCH", f"image query: {sample_image}")
        img_rows = search_similar_exercise_image(
            sample_image, chroma_path=chroma_dir, embedder=image_embedder
        )
        _log("SEARCH", f"top={img_rows[:3]}")
    else:
        _log("SEARCH", "no sample image found for image retrieval test")

    return {"text_records": text_count, "image_records": image_count}


if __name__ == "__main__":
    stats = build_indexes()
    print("Done:", stats)

