from g2p_en import G2p
import nltk

# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('cmudict')
# nltk.download('punkt')

g2p = G2p()

def reStrip(listWord):
    result = []
    for word in listWord:
        data = word.rstrip("012")
        result.append(data)
    return result
print(reStrip(g2p("Lionel Messi")))