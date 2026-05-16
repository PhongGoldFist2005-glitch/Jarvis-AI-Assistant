import sys
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core import prompts # Ép model theo 1 prompts mình mong muốn
from langchain_core.output_parsers import PydanticOutputParser # Dùng để check có cần RAG không khá hay.
# from langchain.agents import create_tool_calling_agent, AgentExecutor # Dùng cho KB memory
from paramInit import initAudioPlayerParams
from GitModel.AI_assistant.RAG.pydanticFirst import checkResponse
from GitModel.AI_assistant.RAG.pydanticOuput import formatOutput
from queryChunk import queryEmbedWord
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Thêm parent directory vào sys.path để import tokenSaving
from tokenSaving import tokenizer

class modelAssistant:
    def __init__(self):
        tokenOb = tokenizer()
        self.__model = ChatGoogleGenerativeAI(
            model= tokenOb.model_name,
            google_api_key= tokenOb.gemini_token
        )
        self.checkParser = PydanticOutputParser(pydantic_object=checkResponse)
        self.outputParser = PydanticOutputParser(pydantic_object=formatOutput)
    
    def checkInput(self, prompt, cache_text):
        if cache_text == "" or cache_text is None:
            cache_text = "Không có thông tin nào trong bộ nhớ cache."
        checkPrompt = prompts.ChatPromptTemplate.from_messages(
            initAudioPlayerParams().checkPrompFunction(parser = self.checkParser)
        ).format_messages(question = prompt, context = cache_text)
        fullResponse = self.basicQuery(
            checkPrompt,
            type=1
        )
        parsed = self.checkParser.parse(fullResponse.content)
        return parsed.needRAG, parsed.text_need_fix
    
    # Chunking dùng trong streaming data
    def handleChunk(self, chunk):
        pass
        
    def useStreamModel(self, prompt, needRag, text_need_fix):
        pass
    
    # Chạy model với prompt đã được format sẵn, trả về kết quả stream
    # Cache text lấy nội dung từ cache trong class STM tồn tại trong audioPlayer.
    def run(self, prompt, cache_text):
        needRag, text_need_fix = self.checkInput(prompt, cache_text)
        print(f"Need RAG: {needRag}, Text need fix: {text_need_fix}")
        yield from self.useStreamModel(prompt, needRag, text_need_fix)

    def basicQuery(self, prompt, type):
        # Loại 0 là dành cho câu hỏi bình thường
        if type == 0:
            my_prompt = prompts.ChatPromptTemplate.from_messages(
                initAudioPlayerParams().outputFormat(parser = self.outputParser)
            ).format_messages(context = "", question = prompt)
        
        # Loại 1 là dành cho việc check needRAG và lỗi chính tả
        if type == 1:
            my_prompt = prompts.ChatPromptTemplate.from_messages(
                initAudioPlayerParams().checkPrompFunction(parser = self.checkParser)
            ).format_messages(question = prompt)
        
        try:
            response = self.__model.invoke(my_prompt)
        except Exception as e:
            print(f"Error during basic query: {e}")
        return response


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(
        model= "gemini-2.5-flash",
        google_api_key= tokenizer().gemini_token
    )
    response = llm.invoke("What is the capital of France?")
    print(response)