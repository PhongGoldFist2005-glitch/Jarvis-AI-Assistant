# AI_assistant

Trợ lý AI tiếng Việt cho hội thoại trực tiếp, hướng tới phản hồi nhanh, tự nhiên, và có ngữ cảnh. Dự án tập trung vào trải nghiệm trò chuyện realtime, kết hợp bộ nhớ ngắn hạn/dài hạn, tìm kiếm ngoài, và xử lý giọng nói để cải thiện chất lượng hội thoại. Đồng thời Agent có thể đóng vai trò như 1 trợ lý AI smart home.

## Tính năng chính
- Hội thoại realtime với ngữ cảnh và phản hồi ngắn gọn, rõ ràng.
- Bộ nhớ ngắn hạn (STM) và dài hạn (LTM) để giữ thông tin người dùng.
- Tìm kiếm ngoài khi cần thông tin cập nhật (thời tiết, tin tức, tỷ giá, ...).
- Xử lý giọng nói: STT (PhoWhisper) + wake-word (hey_jarvis) + edge_tts.
- RAG.

## Cấu trúc thư mục
- RAG: lõi điều phối RAG, gồm các node và graph.
- Source: các module điều khiển audio, wake-word, model.
- Database: dữ liệu mapping, JSONL, và chroma_db.
- PhoWhisper-tiny: model STT.
- qdrant_store: dữ liệu Qdrant local.

## Luồng hội thoại (tóm tắt)
Graph trong [RAG/Graph.py](RAG/Graph.py) điều phối các bước hội thoại:
- Preprocessing + Cache: [RAG/Cache.py](RAG/Cache.py)
- Planner: [RAG/planner.py](RAG/planner.py)
- Retrieval STM/LTM: [RAG/retrievel_STM.py](RAG/retrievel_STM.py), [RAG/retrieval_LTM.py](RAG/retrieval_LTM.py)
- Search: [RAG/Search.py](RAG/Search.py)
- Remove duplicate: [RAG/remove_dup.py](RAG/remove_dup.py)
- LLM generation: [RAG/LLM_gen.py](RAG/LLM_gen.py)
- Update STM/LTM + Refresh: [RAG/updateSTM.py](RAG/updateSTM.py), [RAG/updateLTM.py](RAG/updateLTM.py), [RAG/refreshState.py](RAG/refreshState.py)

## Yêu cầu môi trường
- Python 3.13
- Qdrant (cho LTM) tại http://localhost:6333/
- Các thư viện chính: langgraph, langchain, sentence-transformers, qdrant-client, ddgs

## Cài đặt nhanh
1. Cài dependencies (tham khảo lib.txt nếu có):
	- `pip install -r lib.txt`
2. Bật Qdrant local:
	- Ví dụ Docker: `docker run -p 6333:6333 qdrant/qdrant`
3. Cấu hình API key cho LLM (Gemini) trong file cấu hình nội bộ của bạn.

## Chạy demo RAG
Script demo: [RAG/demo_graph_run.py](RAG/demo_graph_run.py)

Chạy:
```
python RAG/demo_graph_run.py
```

Ghi chú:
- Demo sử dụng file JSONL mapping tại `Database/reversed_word.jsonl`.
- Nếu LTM chưa có Qdrant đang chạy, sẽ báo lỗi kết nối.

## Hiển thị graph
Notebook: [RAG/Graph_view.ipynb](RAG/Graph_view.ipynb)

Chạy các cell để xuất ảnh graph sang `RAG/graph_nx.png`.

## License
Chưa thiết lập.
