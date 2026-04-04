from vncorenlp import VnCoreNLP

annotator = VnCoreNLP(r"P:\Program Files\Python313\AI_assistance\Model\VnCoreNLP-1.1.1.jar", annotators="wseg")

text = "li"
tokens = annotator.tokenize(text)
print(tokens)