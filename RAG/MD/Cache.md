Vấn đề mình đang gặp phải ở đây là gì ?
    - Chưa hoàn toàn nắm được toàn bộ ý tưởng RAG của mình.
        - Có một số khối chưa nắm rõ được ý tưởng triển khai
         - Khối nào ?
            - Khối cache, việc nối từ processing text sang khối cache liệu đã hợp lý hay chưa ?
                Việc nối từ processing text sang khối cache là không hiểu quả vì sẽ phải tốn 2 lần prompt trước khi lần prompt chính thức được lấy về.
                - Cách khắc phục sẽ là gì ?
                    - 
        - Chưa nắm được các khối sau liệu có vấn đề nữa hay không ?
            - Chưa đọc và triển khai thử
    - Chưa thực sự chốt được cái state sẽ được triển khai thì trong đó nó sẽ có những cái gì ?
        - Chưa thực sự nắm rõ mục đích của khối state được thực hiện trong langgraph
            - Chưa thực sự nắm được langgraph
             - Chỉ có cách triển khai bản đơn giản demo
    - Code đang thực sự loạn vì chưa nắm rõ là full pipeline nó sẽ thực hiện các thành phần như thế nào ?.
        - Cụ thể hơn đi: Ban đầu mình ko định triển khai langgraph thì việc RAG maybe nó sẽ có thể kết hợp với phần backend chỗ voice. Nhưng khiển khai langraph yêu cầu không chỉ đơn giản gọi mô hình kết hợp Text để tạo output mà yêu cầu phần chatbot nằm ở phía sau hoặc phía trong đồ thị
        - Cần thống nhất về full pipeline 1 lần nữa
    - Mình không chắc chắn triển khai như vậy liệu có hoạt động được hay không ?
        
    - Một sô vấn đề phụ làm bị loạn:
        - Chưa hiểu rõ trước đó mình làm gì cho lắm
            - Dùng chat gpt tóm tắt hộ
        - Chưa biết cách viết file MD bố cục sao cho AI có thể hiểu và triển khai theo ý mình nói lắm
        - Chưa format lại được kiểu viết Code của Ai cho lắm

    Vấn đề hiện tại:
        - Context và Prompt cho phần function calling sẽ khác so với Context của phần RAG thực sự
        - Không rõ cái cách fix context và prompt mới như thế nào
            - Nó sẽ bị ghi đè, lúc return cần phải đảm bảo là nó phải được merge trước để tránh mất thông tin