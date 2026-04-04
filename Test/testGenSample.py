from g2p_en import G2p
import json
from itertools import combinations

g2p = G2p()

def reStrip(listWord):
    result = []
    for word in listWord:
        data = word.rstrip("012").lower()
        result.append(data)
    return result

def processingList(listData):
    dictIndex = {}
    for idx, value in enumerate(listData):
        if value in dictIndex:
            dictIndex[value].append(idx)
        else:
            dictIndex[value] = [idx]
    return dictIndex

def givenCombo(listIndex):
    listCombo = []
    for i in range(1, len(listIndex) + 1):
        for combo in combinations(listIndex, i):
            listCombo.append(combo)
    return listCombo

with open(r"P:\Program Files\Python313\AI_assistance\Model\Database\mapping.json", encoding= "utf-8") as f:
    data = json.load(f)

print("".join(g2p("Lionel Messi")))
tempList = reStrip(list(g2p("Lionel Messi")))
dictIndex = processingList(tempList)

result = []
# Value là list các vị trí
for key, value in dictIndex.items():
    if key in data:
        # Nếu từ đó nằm trong database, duyệt qua các bộ từ có thể thay thế
        # Mỗi từ có thể thay thế thì có 1 bộ tổ hợp có thể thay thế
        listCombo = givenCombo(value)
        for word in data[key]:
            # Duyệt qua lần lượt các bộ combo
            for combo in listCombo:
                combo = list(combo)
                copyData = tempList.copy()
                # Thay thế các từ trong word gốc dựa vào vị trí của combo
                for pos in combo:
                    copyData[pos] = word
                copyData = "".join(copyData)
                result.append(copyData)

print(result)


# Xây dựng bảng mapping: Dựa trên cách đọc tiếng Anh phiên âm sang tiếng Việt, ví dụ m eh: m é t, r o: r ô
# Với mỗi từ mới, không chỉ là 1 từ có thể ghép vào nhiều chỗ mà còn là nhiều từ ghép vào nhiều chỗ
# Mỗi đoạn có trọng âm thì thay đoạn đấy bằng dấu cách vd: LAY1AH0NAH0L MEH1SIY0 -> LAY AH NAH L MEH SIY -> li ô ne me si

