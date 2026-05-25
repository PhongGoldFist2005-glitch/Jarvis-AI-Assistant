from collections import deque
from datetime import datetime

class Short_Term_Memory:
    def __init__(self):
        self.context_window = deque()
        self.window_max_size = 50
    def insert_into_window(self, summary_message):
        if len(self.context_window) >= self.window_max_size:
            self.context_window.popleft()
        content = {
            "message": summary_message,
            "timestamp": datetime.now()
        }
        self.context_window.append(content)
        return content
    def clear_window(self):
        self.context_window.clear()
    
    def query_window(self, query_type= 2):
        if query_type == 0:
            return list(self.context_window)
        if query_type == 1:
            return list(self.context_window)[-5:]
        
        return list(self.context_window)[-10:]