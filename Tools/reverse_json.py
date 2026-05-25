import json

def reverse_jsonl(inputFile, outputFile):
    with open(inputFile, "r", encoding="utf-8") as fin, \
        open(outputFile, "w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line.strip())
            reversed_dict = {}

            for correct, wrong_list in data.items():
                for wrong in wrong_list:
                    reversed_dict[wrong] = correct

            fout.write(json.dumps(reversed_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    inputFile = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\word.jsonl"
    outputFile = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\reversed_word.jsonl"
    reverse_jsonl(inputFile, outputFile)