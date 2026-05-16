Đặc tả các node:
RAG: sử dụng function calling để biết có cần dùng RAG hay không.
    STM: Short-term-memory:
    Mục đích: Lưu nội dung tạm thời với 1 kích thước bộ nhớ nhất định.
        - 1. Có 1 context window (tương tự như 1 đoạn chat) để lưu trữ thông tin tạm thời. [{}]
        - 2. Có cơ chế insert vào window
            Nó sẽ kiểm tra liệu rằng window đã đầy chưa, nếu đầy thì sẽ xóa thông tin cũ nhất để insert thông tin mới vào window. Còn nếu chưa đầy thì sẽ insert thông tin mới vào window.
            Insert summary info
        - 3. Có cơ chế clear window.
            Khi clear thì sẽ xóa hết thông tin trong window.
        - 4. Có cơ chế truy vấn:
            Truy vấn sẽ trả về thông tin trong window. Bao gồm 2 loại truy vấn: (Cần cơ chế function calling để xác định loại truy vấn nào được sử dụng)
                - Truy vấn toàn bộ window: trả về toàn bộ thông tin trong window.
                - Truy vấn theo top 5: Với câu hỏi không đề cập nhiều đến lịch sử hội thoại
                - Truy vấn theo top 10: Với câu hỏi có đề cập đến lịch sử hội thoại
    
    LTM: Long-term-memory: Update in the background (Sau 1 thời gian nhất định từ cuộc trò chuyện gần nhất & phải có thông tin mới)
    Mục đích: Lưu trữ thông tin mang tính cá nhân hóa.
        - 1. Cần có cơ chế check liệu có cần lưu thông tin này vào LTM không ?, fix thêm trong type json của output
        - 2. Có cơ chế insert vào vector db. Sử dụng type để phân biệt loại thông tin nào sẽ được lưu vào LTM. Đầu vào: text, type
            Thông tin insert được LLM đánh giá là có giá trị để lưu vào LTM.
            Thông tin insert sẽ là thông tin được suy ra từ STM
            ! Yêu cầu lưu luôn thông tin thời gian cập nhật
        - 3. Có cơ chế truy vấn
            Truy vấn sẽ trả về thông tin liên quan đến người dùng từ vector db.
            Có 1 biến để lưu trữ thông tin truy vấn trả về trong phần init
        - 4. Có cơ chế xóa các thông tin liên quan mà người dùng yêu cầu xóa. (Sử dụng cơ chế vector search để tìm các thông tin liên quan đến yêu cầu xóa của người dùng, sau đó xóa chúng khỏi vector db. Yêu cầu có 1 ngưỡng để xác định các thông tin liên quan đến yêu cầu xóa của người dùng.(Cái này sử dụng đầu vào truyền vào))
        Note: Sử dụng Qdrant với host là : http://localhost:6333/, collection_names sẽ là Long Term Memory, vector size sẽ là 384, distance sẽ là cosine. Sử dụng mô hình embedding là "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    Search:
    Mục đích: Research từ nguồn dữ liệu ở bên ngoài để trả lời câu hỏi của người dùng.
        - 1. Cần có cơ chế check liệu có cần truy vấn RAG không ?, Đã có
        - 2. Có cơ chế truy vấn RAG:
            Truy vấn sẽ trả về thông tin liên quan đến câu hỏi của người dùng từ nguồn dữ liệu bên ngoài.
flow:
