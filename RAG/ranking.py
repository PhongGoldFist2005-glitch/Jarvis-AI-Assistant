# Lớp ranking sẽ là lớp tổng hợp lại cả BM25 và vector search để đánh giá mức độ liên quan của document với query, từ đó đưa ra thứ tự ưu tiên khi trả về kết quả.
# Phương pháp của class:
# Lấy cả KB, LongTermMem, ShortTermMem để đánh giá mức độ liên quan của document với query.
# Đối với KB, LongTermMem sẽ dùng vector search và BM25 để chọn top 3 document và thứ tự ranking của chúng
# Đối với ShortTerm sẽ lấy document trong slide window hiện tại (3), ranking là dựa trên thứ tự gần nhất (gần nhất sẽ được ưu tiên hơn)
# Tính MMR (Cân bằng độ liên quan và đa dạng), RRF(Gộp dựa trên ranking không dựa vào điểm số) dựa vào query của người dùng và từ top 3 của từng KB, LongTermMem, ShortTermMem (Điểm dựa trên thứ tự gần nhất) để đánh giá mức độ liên quan của document với query.
# MMR dùng alpha là 0.5, còn RRF dùng k là 60 (điểm sẽ giảm dần theo thứ tự gần nhất)
# Đánh giá trên 1 cái ngưỡng và chọn ra các document có điểm số cao hơn ngưỡng đó để trả về cho LLM, nếu không có document nào vượt qua ngưỡng thì sẽ trả về 1 document có điểm số cao nhất.
# Trong init cần có sẵn 3 thread có sẵn để xử lý song song cho 3 nguồn KB, LongTermMem, ShortTermMem để tăng tốc độ đánh giá và trả về kết quả nhanh hơn.
