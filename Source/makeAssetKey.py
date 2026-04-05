from g2p_en import G2p
import json
from itertools import combinations, product
import unicodedata
from tqdm import tqdm
import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('cmudict')
# nltk.download('punkt')
import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool
import os

SUPPORT_LETTER_SET = {
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x"
}

def reStrip(listWord):
    result = []
    for word in listWord:
        if "0" in word or "1" in word or "2" in word:
            data = word.rstrip("012").lower()
            result.append(data)
            result.append(" ")
        else:
            result.append(word.lower())
    return result


def givenCombo(listIndex):
    listCombo = []
    for i in range(1, len(listIndex) + 1):
        for combo in combinations(listIndex, i):
            listCombo.append(combo)
    return listCombo

_TONE_CACHE = {}

@lru_cache(maxsize=10000)
def remove_tone(text):
    if text in _TONE_CACHE:
        return _TONE_CACHE[text]
    
    text = text.lower()
    # ✅ NFD normalize
    text = unicodedata.normalize('NFD', text)
    # ✅ Sử dụng regex compiled thay vì list comprehension
    text = re.sub(r'[\u0300-\u036f]', '', text)  # Unicode diacritical marks
    text = text.replace('đ', 'd')
    
    _TONE_CACHE[text] = text
    return text

def removeSupportLetter(text):
    # Nếu khi split có các từ đứng 1 mình thuộc supportLetter thì xóa ra khỏi từ
    return " ".join(word for word in text.split() if word not in SUPPORT_LETTER_SET)

# Cơ chế tính điểm
# Có nghĩa hết: 1 điểm
# Không có nghĩa: 0 điểm
# Bonus cộng thêm 0.05 * độ dài từ (Khi đã đúng hết mới được cộng bonus)
# => Từ càng dài càng đúng thì điểm càng cao
# Cộng lại chia cho số lượng chữ
def caculateScore(inputWord, listSyllables):
    point = 0
    listCheckVN = inputWord.split(" ")
    for word in listCheckVN:
        if word in listSyllables:
            point += 1
    if point == len(listCheckVN) and len(listCheckVN) > 3:
        point += 0.05
    return point // len(listCheckVN) if len(listCheckVN) > 0 else 0

def makeAssetKeyValue(phoneme_list, listSyllables, data, maxLen):
    # Remove stress number
    tempList = reStrip(phoneme_list)

    # Tìm các vị trí có thể thay
    replace_positions = []

    for i, p in enumerate(tempList):
        if p in data:
            replace_positions.append(i)
    # Sinh tổ hợp vị trí
    if len(replace_positions) > 8:
        # Chỉ lấy 5 vị trí có nhiều lựa chọn nhất
        replace_positions = sorted(
            replace_positions, 
            key=lambda i: len(data[tempList[i]]), 
            reverse=True
        )[:5]
    position_combos = givenCombo(replace_positions)
    result = set()

    minPoint = float('inf')
    # max : 32 combo
    for combo in position_combos:
        # lấy danh sách lựa chọn cho từng vị trí
        choices = []
        for pos in combo:
            phoneme = tempList[pos]
            choices.append(data[phoneme])
            # sinh tổ hợp từ
        for wordCombo in product(*choices):
            # Báo hiệu cho rằng từ sau khi ghép nối thực sự có nghĩa
            copyData = tempList.copy()

            for i in range(len(combo)):
                index = combo[i]
                copyData[index] = wordCombo[i]
            
            # Gộp list các từ sau quy đổi âm tiết thành 1 khối
            dataName = removeSupportLetter(remove_tone("".join(copyData)))
            # Lấy các từ sau khi gộp lại coi từ được gộp có thực sự có nghĩa không
            point = caculateScore(dataName, listSyllables)
            if len(result) < maxLen:
                dictTemp = (dataName, point)
                result.add(dictTemp)
                if point < minPoint:
                    minPoint = point
                    minWord = dictTemp
            else:
                if point > minPoint:
                    result.discard(minWord)
                    dictTemp = (dataName, point)
                    result.add(dictTemp)
                    minWord = min(result, key=lambda x: x[1])
            if point >= 1:
                break
    return [item[0] for item in result]

def init_worker(syllables_arg, data_arg):
    global g2p, syllables, data
    g2p = G2p()
    syllables = syllables_arg
    data = data_arg

def process_single_word(word):
    phoneme_list = list(g2p(word))
    return {word: makeAssetKeyValue(phoneme_list, syllables, data, maxLen=25)}

if __name__ == "__main__":
    # Load mapping
    # G2P
    g2p = G2p()
    with open("../Database/GPTMapping.json", encoding="utf-8") as f:
        data = json.load(f)
    
    print("Mapping table completed")
    
    # Load name playlist
    wordBucket = []
    with open("../bucket.txt","r",encoding= "utf-8") as f:
        for line in f:
            wordBucket.append(line.strip())
    print("Word bucket completed")

    # Load syllables
    with open("../common-vietnamese-syllables.txt", encoding="utf-8") as f:
        syllables = set(remove_tone(line.strip()) for line in f)
    print("Vietnamese word completed")
    
    with Pool(
        processes=os.cpu_count() - 1,
        initializer=init_worker,
        initargs=(syllables, data)
    ) as pool:
        results = []
        with tqdm(total=len(wordBucket), desc="Processing", unit="word") as pbar:
            for result in pool.imap_unordered(process_single_word, wordBucket):
                results.append(result)
                pbar.update(1)
    
    with open("../word.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
