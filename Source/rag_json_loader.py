import json
from typing import Dict


def load_reversed_jsonl(jsonl_path: str) -> Dict:
    # Doc tung dong JSONL va gom tat ca key/value vao 1 dict duy nhat
    mapping: Dict = {}
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue

            if isinstance(item, dict):
                # Neu la dict co 1 cap key/value duy nhat thi merge truc tiep
                if len(item) == 1:
                    mapping.update(item)
                    continue
                # Neu co cap key/value ro rang thi lay theo cap do
                if "key" in item and "value" in item:
                    mapping[item["key"]] = item["value"]
                    continue

    return mapping
