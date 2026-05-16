import chromadb
import sys
import os
import json
from GitModel.AI_assistant.Source.paramInit import initAudioPlayerParams

# Thêm parent directory vào sys.path để import tokenSaving
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class queryEmbedWord:
    def __init__(self):
        self.__client = chromadb.PersistentClient(path= initAudioPlayerParams().chunk_database)
        self.__collection = self.__client.get_collection(name= initAudioPlayerParams().db_check_chunk)
        self.__inputJSON = initAudioPlayerParams().inputJSON
        self.__dictResult = {}
        try:
            with open(self.__inputJSON, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.__dictResult.update(data)
        except Exception as e:
            print("Error loading JSON file:", e)
        
        self.__threshold = initAudioPlayerParams().threshold_check_chunk
    
    def query(self, queryText, resultSpellCheck):
        # Nếu có từ sai, truy vấn từ đúng chính tả trong ChromaDB và thay thế
        result = self.__collection.query(
            query_texts=[resultSpellCheck],
            n_results=1,
            include=["documents", "distances"]
        )
        fixedWord, distance = result["documents"][0][0], result["distances"][0][0]

        if distance > self.__threshold:
            # Nếu khoảng cách lớn hơn ngưỡng, coi như không có từ nào phù hợp → trả về query gốc
            return queryText
        else:
            fixedWord = self.__dictResult.get(fixedWord)
            outputString = queryText.replace(resultSpellCheck, fixedWord)
            return outputString