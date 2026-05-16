from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from State import State
from param import Param

_EMBED_MODEL: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
	global _EMBED_MODEL
	if _EMBED_MODEL is None:
		_EMBED_MODEL = SentenceTransformer(Param.EMBEDDING_MODEL)
	return _EMBED_MODEL


def remove_duplicate(state: State):
	search_results = state.get("search_results") or []
	ltm_results = state.get("ltm_results") or []
	stm_results = state.get("stm_results") or []

	print("[remove_duplicate] input counts:", {
		"search": len(search_results),
		"ltm": len(ltm_results),
		"stm": len(stm_results),
	})

	items: List[Tuple[str, object, str]] = []

	for entry in search_results:
		text = ""
		if isinstance(entry, dict):
			text = str(entry.get("body", "")).strip()
		else:
			text = str(entry).strip() if entry is not None else ""
		if text:
			items.append(("search", entry, text))

	for entry in ltm_results:
		text = ""
		if isinstance(entry, dict):
			text = str(entry.get("text", "")).strip()
		else:
			text = str(entry).strip() if entry is not None else ""
		if text:
			items.append(("ltm", entry, text))

	for entry in stm_results:
		text = ""
		if isinstance(entry, dict):
			text = str(entry.get("message", "")).strip()
		else:
			text = str(entry).strip() if entry is not None else ""
		if text:
			items.append(("stm", entry, text))

	if not items:
		print("[remove_duplicate] no items, skip dedup")
		return {
			"context": state.get("context"),
			"search_results": search_results,
			"ltm_results": ltm_results,
			"stm_results": stm_results,
		}

	texts = [item[2] for item in items]
	model = _get_embedder()
	embeddings = model.encode(texts, normalize_embeddings=True)
	vectors = np.asarray(embeddings, dtype=np.float32)
	similarity_matrix = vectors @ vectors.T

	keep_indices: List[int] = []
	threshold = Param.DUPLICATE_SIMILARITY_THRESHOLD
	for i in range(len(texts)):
		is_duplicate = False
		for j in keep_indices:
			if similarity_matrix[i, j] >= threshold:
				is_duplicate = True
				break
		if not is_duplicate:
			keep_indices.append(i)

	kept_items = [items[i] for i in keep_indices]
	unique_texts = [item[2] for item in kept_items]
	context = "\n".join(unique_texts)

	filtered_search: List[str] = []
	filtered_ltm: List[object] = []
	filtered_stm: List[object] = []
	for source, entry, _ in kept_items:
		if source == "search":
			filtered_search.append(entry)
		elif source == "ltm":
			filtered_ltm.append(entry)
		else:
			filtered_stm.append(entry)

	print("[remove_duplicate] output counts:", {
		"search": len(filtered_search),
		"ltm": len(filtered_ltm),
		"stm": len(filtered_stm),
	})

	return {
		"context": context,
		"search_results": filtered_search,
		"ltm_results": filtered_ltm,
		"stm_results": filtered_stm,
	}
