🧠 Kiến trúc RAG cho AI Assistant Tiếng Việt (Tối ưu tốc độ & bộ nhớ dài hạn)
1. 🎯 Mục tiêu hệ thống

Xây dựng một hệ thống RAG (Retrieval-Augmented Generation) cho AI Assistant tiếng Việt với các khả năng:

Ghi nhớ ngữ cảnh hội thoại trong phiên hiện tại
Ghi nhớ lịch sử hội thoại đa phiên
Tích lũy và lưu trữ tri thức chắt lọc từ external sources
Đảm bảo tốc độ phản hồi real-time
Hạn chế tối đa việc dùng LLM cho filtering → ưu tiên BM25 / Vector Search / Heuristic
Có cơ chế đánh giá chất lượng thông tin trước khi:
Lưu trữ
Đưa vào context cho LLM!

Note:
Giải thích các tech words: [text](https://chatgpt.com/c/69e0f67d-3898-839c-af87-afbd3ebfc148)
Lộ trình: [text](https://claude.ai/chat/f572de21-9e28-417b-a93a-b8e14f45735d)
Tất cả các bước cập nhật các db thì đều sẽ được cập nhật song song và chạy ngầm khi có nguồn thông tin mới.
![alt text](image.png)
Shorterm: Lưu trữ theo time slide window, trả về time slide đó
Longterm: Lưu trữ có giới hạn theo top K và đánh giá chất lượng thông tin trước khi lưu trữ. Trả về thông tin tích góp theo thời gian và chất lượng top m (m < K) để đưa vào context cho LLM.>
KB: Lưu trữ tri thức được chắt lọc đủ lớn được đánh giá kỹ càng về do các phương pháp và LLM đánh giá. Trả về thông tin có liên quan nhất đến ngữ cảnh.