from collections import deque
from typing import Dict, List, Optional, Any
import time
import json


class SessionMem:
    """
    Lớp SessionMem sử dụng Sliding Window để lưu trữ top k cuộc hội thoại gần nhất.
    Duy trì thứ tự thời gian và cung cấp ngữ cảnh tốt cho LLM.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Khởi tạo SessionMem với sliding window.
        
        Args:
            max_size (int): Số lượng hội thoại tối đa giữ trong memory (mặc định 10)
        """
        self.max_size = max_size
        self.window = deque(maxlen=max_size)  # Sliding window tự động xóa item cũ
        self.session_id = self._generate_session_id()
        self.created_at = time.time()
        self.last_updated = time.time()

    # ==================== ADD/STORE ====================
    def add_conversation(self, user_question: str, chatbot_answer: str, 
                        instructions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Thêm một cuộc hội thoại vào sliding window.
        
        Args:
            user_question (str): Câu hỏi của người dùng
            chatbot_answer (str): Câu trả lời của chatbot
            instructions (Optional[Dict]): Các instructions liên quan
        
        Returns:
            Dict: Conversation entry vừa được thêm
        """
        conversation = {
            'id': len(self.window),
            'user_question': user_question,
            'chatbot_answer': chatbot_answer,
            'instructions': instructions or {},
            'timestamp': time.time()
        }
        
        self.window.append(conversation)
        self.last_updated = time.time()
        
        return conversation

    def add_batch_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Thêm nhiều cuộc hội thoại cùng lúc.
        
        Args:
            conversations (List[Dict]): Danh sách conversations
                Mỗi item phải có: user_question, chatbot_answer, (optional) instructions
        
        Returns:
            List[Dict]: Danh sách conversations vừa được thêm
        """
        added = []
        for conv in conversations:
            result = self.add_conversation(
                user_question=conv.get('user_question', ''),
                chatbot_answer=conv.get('chatbot_answer', ''),
                instructions=conv.get('instructions', {})
            )
            added.append(result)
        
        return added

    # ==================== RETRIEVE/QUERY ====================
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Lấy tất cả các cuộc hội thoại trong sliding window (theo thứ tự thời gian).
        
        Returns:
            List[Dict]: Danh sách tất cả conversations
        """
        return list(self.window)

    def get_recent_conversations(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Lấy `count` cuộc hội thoại gần nhất.
        
        Args:
            count (int): Số lượng conversations cần lấy
        
        Returns:
            List[Dict]: Danh sách conversations gần nhất
        """
        if count <= 0:
            return []
        
        return list(self.window)[-count:]

    def get_conversation_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Lấy một cuộc hội thoại theo vị trí index.
        
        Args:
            index (int): Vị trí trong window (0 = cũ nhất, -1 = mới nhất)
        
        Returns:
            Optional[Dict]: Conversation nếu tồn tại, None nếu không
        """
        try:
            return list(self.window)[index]
        except IndexError:
            return None

    def search_by_keyword(self, keyword: str, search_in: str = 'all') -> List[Dict[str, Any]]:
        """
        Tìm kiếm cuộc hội thoại theo từ khóa.
        
        Args:
            keyword (str): Từ khóa tìm kiếm
            search_in (str): Tìm trong 'all', 'user_question', 'chatbot_answer'
        
        Returns:
            List[Dict]: Danh sách conversations chứa từ khóa
        """
        keyword = keyword.lower()
        results = []
        
        for conv in self.window:
            if search_in == 'user_question':
                if keyword in conv['user_question'].lower():
                    results.append(conv)
            elif search_in == 'chatbot_answer':
                if keyword in conv['chatbot_answer'].lower():
                    results.append(conv)
            else:  # 'all'
                if (keyword in conv['user_question'].lower() or 
                    keyword in conv['chatbot_answer'].lower()):
                    results.append(conv)
        
        return results

    def get_context_window(self, current_index: int = -1, window_size: int = 3) -> str:
        """
        Lấy context window từ vị trí hiện tại (gồm conversations trước và sau).
        Hữu ích để cung cấp ngữ cảnh cho LLM.
        
        Args:
            current_index (int): Vị trí hiện tại (-1 = mới nhất)
            window_size (int): Kích thước context window
        
        Returns:
            str: Formatted context string
        """
        conversations = list(self.window)
        
        if not conversations:
            return ""
        
        # Điều chỉnh index
        if current_index < 0:
            current_index = len(conversations) + current_index
        
        # Tính range
        start = max(0, current_index - window_size)
        end = min(len(conversations), current_index + window_size + 1)
        
        context_convs = conversations[start:end]
        
        # Format context
        context_parts = []
        for conv in context_convs:
            context_parts.append(
                f"User: {conv['user_question']}\n"
                f"Bot: {conv['chatbot_answer']}"
            )
        
        return "\n---\n".join(context_parts)

    # ==================== UPDATE ====================
    def update_last_conversation(self, chatbot_answer: Optional[str] = None, 
                                instructions: Optional[Dict] = None) -> bool:
        """
        Cập nhật cuộc hội thoại mới nhất (hữu ích nếu cần sửa lại answer hoặc instructions).
        
        Args:
            chatbot_answer (Optional[str]): Câu trả lời mới
            instructions (Optional[Dict]): Instructions mới
        
        Returns:
            bool: True nếu cập nhật thành công
        """
        try:
            if len(self.window) == 0:
                return False
            
            # Lấy item cuối cùng
            conv = list(self.window)[-1]
            
            if chatbot_answer:
                conv['chatbot_answer'] = chatbot_answer
            if instructions:
                conv['instructions'].update(instructions)
            
            self.last_updated = time.time()
            return True
        except Exception as e:
            print(f"Lỗi khi cập nhật conversation: {e}")
            return False

    # ==================== DELETE ====================
    def clear(self):
        """
        Xóa toàn bộ sliding window.
        """
        self.window.clear()
        self.last_updated = time.time()

    def remove_oldest(self, count: int = 1) -> int:
        """
        Xóa `count` cuộc hội thoại cũ nhất.
        
        Args:
            count (int): Số lượng conversations cần xóa
        
        Returns:
            int: Số conversations đã xóa
        """
        removed = 0
        for _ in range(min(count, len(self.window))):
            self.window.popleft()
            removed += 1
        
        self.last_updated = time.time()
        return removed

    # ==================== EXPORT/IMPORT ====================
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Xuất session memory thành dictionary (hữu ích để save/restore).
        
        Returns:
            Dict: Toàn bộ dữ liệu session
        """
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'max_size': self.max_size,
            'conversations': list(self.window)
        }

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Xuất session memory thành JSON format.
        
        Args:
            filepath (Optional[str]): Đường dẫn file để save (nếu None, chỉ return JSON string)
        
        Returns:
            str: JSON string
        """
        data = self.export_to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                print(f"Đã save session vào: {filepath}")
            except Exception as e:
                print(f"Lỗi khi save session: {e}")
        
        return json_str

    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Nhập session memory từ dictionary.
        
        Args:
            data (Dict): Dictionary chứa session data
        
        Returns:
            bool: True nếu import thành công
        """
        try:
            self.session_id = data.get('session_id', self.session_id)
            self.created_at = data.get('created_at', self.created_at)
            self.max_size = data.get('max_size', self.max_size)
            
            self.window = deque(
                data.get('conversations', []),
                maxlen=self.max_size
            )
            self.last_updated = time.time()
            return True
        except Exception as e:
            print(f"Lỗi khi import session: {e}")
            return False

    def import_from_json(self, filepath: str) -> bool:
        """
        Nhập session memory từ JSON file.
        
        Args:
            filepath (str): Đường dẫn JSON file
        
        Returns:
            bool: True nếu import thành công
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.import_from_dict(data)
        except Exception as e:
            print(f"Lỗi khi import từ JSON: {e}")
            return False

    # ==================== UTILITY ====================
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về session memory.
        
        Returns:
            Dict: Thông tin thống kê
        """
        return {
            'session_id': self.session_id,
            'total_conversations': len(self.window),
            'max_capacity': self.max_size,
            'utilization': f"{(len(self.window) / self.max_size * 100):.1f}%",
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at)),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_updated)),
            'duration': f"{(self.last_updated - self.created_at):.1f}s"
        }

    def print_history(self, count: Optional[int] = None):
        """
        In ra lịch sử hội thoại theo định dạng dễ đọc.
        
        Args:
            count (Optional[int]): Số conversations cần in (None = tất cả)
        """
        conversations = self.get_recent_conversations(count) if count else self.get_all_conversations()
        
        if not conversations:
            print("Chưa có cuộc hội thoại nào.")
            return
        
        print("\n" + "="*80)
        print(f"SESSION HISTORY ({len(conversations)} conversations)")
        print("="*80)
        
        for i, conv in enumerate(conversations):
            timestamp = time.strftime('%H:%M:%S', time.localtime(conv['timestamp']))
            print(f"\n[{i+1}] {timestamp}")
            print(f"👤 User: {conv['user_question']}")
            print(f"🤖 Bot: {conv['chatbot_answer']}")
            
            if conv['instructions']:
                print(f"📋 Instructions: {conv['instructions']}")
        
        print("\n" + "="*80 + "\n")

    def _generate_session_id(self) -> str:
        """
        Tạo session ID duy nhất.
        
        Returns:
            str: Session ID
        """
        return f"session_{int(time.time() * 1000)}"

    def get_session_id(self) -> str:
        """Lấy session ID."""
        return self.session_id

    def get_size(self) -> int:
        """Lấy số lượng conversations hiện tại."""
        return len(self.window)

    def is_full(self) -> bool:
        """Kiểm tra xem sliding window có đầy không."""
        return len(self.window) == self.max_size
