- Ý tưởng
    - **Tên dự án:** Hệ thống Bản sao số Điều khiển Tín hiệu Giao thông Thông minh (Smart Traffic Digital Twin).
    - **Mô tả:** Xây dựng một hệ thống đèn tín hiệu giao thông minh hoạt động dựa trên cơ chế Học tăng cường (Reinforcement Learning - RL). Thay vì lập trình cứng thời gian đếm ngược như đèn xanh/đỏ truyền thống, hệ thống cho phép các nút giao thông tự động "học" cách điều phối dòng xe thông qua quá trình thử-và-sai (trial and error) liên tục trong môi trường giả lập.
    - **Giải thích:** bài toán giao thông được ánh xạ trực tiếp qua 4 thành phần:
        - **Tác tử (Agent):** Mỗi ngã tư đường là một "tác tử AI" độc lập.
        - **Quan sát (State):** Tác tử "nhìn" vào môi trường để đếm xem hiện tại mỗi làn đường đang có bao nhiêu chiếc xe đang xếp hàng chờ.
        - **Hành động (Action):** Dựa vào những gì nhìn thấy, tác tử ra quyết định: *Bật đèn xanh cho hướng đi nào tiếp theo?*
        - **Phần thưởng (Reward):** Đây là cách ta "dạy" tác tử. Nếu quyết định của tác tử giúp giải tỏa được đám đông (giảm chênh lệch xe giữa đầu vào và đầu ra), nó sẽ được **cộng điểm**. Ngược lại, nếu quyết định sai lầm khiến xe kẹt dài hơn, nó bị **trừ điểm**.
        
        → **Kết quả:** Qua hàng ngàn lần chạy thử nghiệm, mạng Nơ-ron của tác tử sẽ tự đúc kết ra một **Chính sách (Policy)** tối ưu nhất để luôn đạt được điểm số cao nhất (tương đương với việc không bao giờ để xảy ra kẹt xe).
        
    - **Kiến trúc sử dụng:** Sử dụng thuật toán **Deep Q-Network (DQN)** độc lập cho từng ngã tư. Điểm đặc biệt là áp dụng cơ chế **Chia sẻ tham số (Parameter Sharing)**: thay vì mỗi ngã tư tự học một mình, tất cả các ngã tư sẽ dùng chung một "bộ não" Nơ-ron để chia sẻ kinh nghiệm cho nhau, giúp hệ thống khôn lên cực kỳ nhanh.
    - Ý nghĩa thực tế
        1. Đối với Hệ thống vận hành giao thông:
            - **Bài toán:** Các hệ thống đèn cảm biến (Actuated) hiện tại chỉ phản ứng một cách thiển cận (thấy xe thì bật xanh), dễ dẫn đến việc đẩy chỗ kẹt từ ngã tư này sang ngã tư khác.
            - **Giá trị của RL:** RL có chỉ số **Discount Factor ($\gamma$)**. Nó giúp tác tử không chỉ nhìn vào phần thưởng trước mắt, mà còn biết tính toán chuỗi hành động dài hạn. Tác tử RL biết hy sinh lợi ích nhỏ cục bộ để tạo ra một "làn sóng xanh" (Green Wave) thông suốt cho toàn bộ trục đường chính.
        2. Đối với Kỹ sư quy hoạch:
            - **Bài toán:** Thiếu dữ liệu mô phỏng động để xem xét thiết kế đường sá.
            - **Giá trị của RL:** Trong quá trình huấn luyện tác tử RL, hệ thống liên tục sinh ra các tệp log dữ liệu khổng lồ (Offline Dataset). Kỹ sư có thể dùng chính tập dữ liệu này để phân tích xem nút thắt cổ chai nằm ở đâu mà ngay cả AI tốt nhất cũng không giải quyết được, từ đó quyết định chính xác vị trí cần xây thêm cầu vượt hay mở rộng làn đường.
- Paper tham khảo
    - 3 nền tảng học thuật cốt lõi
        - [**PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network (Wei et al., 2019)](https://faculty.ist.psu.edu/jessieli/Publications/2019-KDD-presslight.pdf):** Đề xuất kiến trúc Independent DQN (mỗi giao lộ là một agent độc lập) kết hợp Parameter Sharing giữa các agent. Hàm phần thưởng được thiết kế dựa trên ý tưởng MaxPressure (tối đa hóa “áp lực” giao thông giữa các pha). Đây là một trong những thuật toán kinh điển và hiệu quả cao cho Multi-Agent Traffic Signal Control, thường được dùng làm baseline mạnh.
            - Giải thích paper
                - Tóm tắt
                    
                    **1. Vấn đề của các phương pháp cũ mà bài báo chỉ ra:**
                    
                    - **Các mô hình RL cũ:** Thường thiết kế hàm phần thưởng (reward) và trạng thái (state) theo cảm tính (heuristic), ví dụ như kết hợp độ dài hàng đợi và thời gian chờ. Điều này làm mô hình học rất chậm và hiệu suất không ổn định.
                    - **Phương pháp Max-Pressure truyền thống:** Tuy có nền tảng toán học tốt để tối đa hóa lưu lượng, nhưng nó lại là một thuật toán "tham lam" (greedy), chỉ tính toán ngắn hạn dẫn đến tối ưu cục bộ.
                    
                    **2. Giải pháp của PressLight:** PressLight tạo ra một Agent (tác nhân AI) điều khiển đèn giao thông tại mỗi nút giao với thiết kế chặt chẽ như sau:
                    
                    - **Thiết kế Hàm phần thưởng (Reward):** Thay vì dùng các chỉ số cảm tính, PressLight sử dụng trực tiếp **áp lực (pressure)** làm hàm phần thưởng.
                        - Phần thưởng ri=−Pi (âm áp lực của nút giao).
                        - Áp lực Pi được tính bằng tổng chênh lệch mật độ xe giữa làn vào (incoming) và làn ra (outgoing).
                        - *Sự đột phá:* Bài báo đã chứng minh bằng toán học rằng việc RL tối đa hóa phần thưởng này (tức là giảm thiểu áp lực) tương đương với việc tối đa hóa lưu lượng thông xe và giảm thiểu thời gian di chuyển của toàn mạng lưới.
                    - **Thiết kế Trạng thái (State):** Để AI ra quyết định, nó chỉ cần quan sát một tập hợp thông tin gọn gàng nhưng đầy đủ (dựa trên chuỗi Markov của MP), bao gồm:
                        - Pha đèn hiện tại.
                        - Số lượng xe trên các làn đầu ra (để biết đường phía trước có kẹt không).
                        - Số lượng xe trên từng đoạn của làn đầu vào.
                    - **Thuật toán học:** Sử dụng **Deep Q-Network (DQN)** để AI tự học qua quá trình thử - sai. Nhờ có hàm phần thưởng dựa trên Max-Pressure, mô hình không bị "lạc lối" và hội tụ nhanh hơn.
                    
                    **3. Kết quả đạt được:**
                    
                    - **Hiệu suất vượt trội:** PressLight vượt qua các phương pháp truyền thống (như thời gian cố định, Max-Pressure gốc, Làn sóng xanh - GreenWave) và cả các mô hình RL tân tiến khác về chỉ số giảm thời gian di chuyển (travel time) trung bình.
                    - **Tự động tạo "Làn sóng xanh" (Green Wave):** Một ưu điểm cực lớn của hệ thống này là các nút giao độc lập có thể tự học cách phối hợp với nhau để tạo ra "làn sóng xanh" – giúp các đoàn xe đi qua hàng loạt ngã tư liên tiếp mà không gặp đèn đỏ, dù không hề được cài đặt quy luật từ trước
                - Giải thích pressure, max-pressure control
                    
                    **1. "Pressure" (Áp lực) là gì?**
                    
                    - **Áp lực của một hướng di chuyển (movement pressure):** Là sự chênh lệch về mật độ phương tiện giữa làn đường đi vào (incoming lane) ngã tư và làn đường đi ra (outgoing lane) khỏi ngã tư đó. Mật độ này bằng số lượng xe hiện tại chia cho sức chứa tối đa của làn đường. Hiểu đơn giản, nếu các làn có sức chứa như nhau, áp lực chính là số lượng xe đang chờ ở làn vào trừ đi số lượng xe ở làn ra.
                    - **Áp lực của một ngã tư (intersection pressure):** Là tổng giá trị tuyệt đối áp lực của tất cả các hướng di chuyển tại ngã tư đó.
                    - **Ý nghĩa thực tế:** Áp lực đại diện cho mức độ mất cân bằng (disequilibrium) của dòng xe. Áp lực tại một ngã tư càng lớn nghĩa là sự phân bổ xe cộ càng mất cân đối (ví dụ: xe dồn ứ ở đầu vào nhưng làn đầu ra lại trống rỗng).
                    1. **Chiến lược "Max Pressure Control"** 
                        
                        **"Movement Pressure"** (áp lực dòng phương tiện) là chỉ số đo lường mức độ khẩn cấp cần giải phóng xe cho một hướng đi cụ thể tại nút giao.
                        
                        Chỉ số này được tính bằng sự chênh lệch lượng xe giữa đầu vào và đầu ra của ngã tư:
                        
                        P=Qin−Qout
                        
                        - **Qin (Upstream Queue):** Số xe đang xếp hàng chờ đi vào ngã tư.
                        - **Qout (Downstream Queue):** Số xe đang ùn ứ ở đoạn đường ngay phía sau ngã tư (hướng thoát).
                        
                        **Nguyên lý hoạt động:**
                        
                        - **P>0:** Làn chờ đông xe, nhưng đường thoát phía trước đang trống. Hệ thống sẽ tính toán và ưu tiên bật **đèn xanh** cho hướng này.
                        - **P≤0:** Đường thoát phía trước đã kẹt cứng (hết chỗ chứa). Lúc này, dù làn vào có xe chờ, hệ thống cũng **không bật đèn xanh** để tránh làm kẹt xe chéo, gây tê liệt toàn bộ ngã tư (gridlock).
        - [**Intelligent traffic signal control based on reinforcement learning: a survey (Hang Xiao, Huale Li, Zhaobin Wang et al.):**](https://link.springer.com/article/10.1007/s10462-026-11530-9) Đây là khảo sát toàn diện về Reinforcement Learning cho Traffic Signal Control (TSC). Paper cung cấp các thông số Hyperparameters chuẩn xác nhất cho các thuật toán phổ biến như DQN (và các biến thể), nhằm đảm bảo hội tụ ổn định và tái lập kết quả dễ dàng hơn.
            - Tóm tắt paper
                
                **1. Động lực và Cách RL mô hình hóa bài toán giao thông**
                
                - **Tại sao lại dùng RL?** Các phương pháp điều khiển đèn giao thông truyền thống (như cài đặt thời gian cố định, kiểm soát thích ứng bằng các luật cố định) không đủ linh hoạt để đối phó với sự phức tạp và thay đổi liên tục của giao thông đô thị hiện đại. RL và đặc biệt là Deep RL (DRL) cho phép hệ thống tự học chiến lược thích ứng thông qua việc tương tác với môi trường.
                - **Mô hình hóa (MDP):** Bài toán giao thông được định nghĩa qua 3 yếu tố:
                    - **Trạng thái (State):** Độ dài hàng đợi, thời gian chờ, số lượng xe, hoặc thậm chí là hình ảnh của ngã tư.
                    - **Hành động (Action):** Chọn giữ hay chuyển pha đèn, điều chỉnh thời lượng đèn xanh, hoặc kết hợp cả hai.
                    - **Phần thưởng (Reward):** Các mục tiêu tối ưu như giảm thiểu thời gian chờ, tối đa hóa lưu lượng (throughput), giảm hàng đợi, hoặc giảm "áp lực" (như trong PressLight).
                
                **2. Phân loại các phương pháp RL trong điều khiển giao thông** Bài báo chia quá trình phát triển của RL thành 2 nhóm quy mô chính:
                
                - **Điều khiển nút giao thông đơn (Single Intersection):** Bắt đầu từ các thuật toán cơ bản dạng bảng (Tabular RL như Q-learning) từ những năm 1990, sau đó tiến hóa lên Deep RL (như DQN, D3QN) để xử lý dữ liệu đầu vào phức tạp như hình ảnh không gian hoặc chuỗi dữ liệu (dùng LSTM) nhằm khắc phục vấn đề khuất tầm nhìn (partial observability).
                - **Điều khiển mạng lưới nhiều nút giao (Multi-intersection / MARL):** Đây là trọng tâm hiện nay, được chia làm 3 hướng tiếp cận:
                    - **Học độc lập (Independent Learning):** Mỗi ngã tư là một tác tử tự quyết định. Thách thức lớn nhất là "tính không dừng" (non-stationarity) vì môi trường liên tục thay đổi do các ngã tư khác cũng đang cập nhật chiến lược.
                    - **Học hợp tác (Collaborative Learning):** Các ngã tư giao tiếp với nhau. Các nghiên cứu đã chuyển từ việc gửi tin nhắn đơn thuần (như thuật toán Max-Plus) sang việc dùng **Mạng nơ-ron đồ thị (GNN)** (như mô hình *CoLight*) và **Transformer** (như *X-Light*) để tự động học được mức độ tương quan không gian giữa các ngã tư.
                    - **Điều khiển phân cấp (Hierarchical Control):** Chia nhỏ vấn đề thành nhiều tầng. Ví dụ, một tầng quản lý (Manager) thiết lập mục tiêu luồng xanh cho cả tuyến đường, trong khi tầng thực thi (Worker) ở dưới tự chuyển đổi pha đèn cho từng ngã tư.
                
                **3. Đánh giá Thực nghiệm (Experiments)** Bài báo không chỉ lý thuyết mà còn chạy thử nghiệm lại các mô hình tiêu biểu (gồm *FGLight, InitLight, HiLight, PressLight, CoLight, X-Light* v.v.) trên các nền tảng mô phỏng lớn như SUMO và CityFlow. Kết quả chỉ ra nguyên tắc chọn mô hình:
                
                - Các mạng giao thông **không có cấu trúc chuẩn và biến động mạnh** (như hệ thống thực tế) cực kỳ phù hợp với các mô hình GNN/Transformer như *X-Light* hay *CoLight*.
                - Với mạng giao thông **chia theo lưới hình vuông chuẩn** (Grid), mô hình chia sẻ tham số như *FGLight* lại cho hiệu quả tốt nhất.
                
                **4. Những thách thức lớn và Hướng đi tương lai** Dù AI rất hứa hẹn trên mô phỏng, để đưa ra ứng dụng thực tế (deploy), bài báo nêu bật các rào cản:
                
                - **Khoảng cách giữa mô phỏng và thực tế (Sim-to-real gap) & Nhiễu cảm biến:** Dữ liệu camera, cảm biến ngoài đời thường bị nhiễu, trễ hoặc mất kết nối. Cần tích hợp các bộ lọc dự đoán để làm mịn dữ liệu đầu vào.
                - **Rủi ro an toàn:** RL học thông qua quá trình "thử - sai", nhưng bạn không thể "thử sai" bừa bãi trên đường thật vì sẽ gây tai nạn. Cần đưa các giới hạn vật lý và ràng buộc an toàn nghiêm ngặt vào thuật toán.
                - **Thiếu tính diễn giải (Interpretability):** Mạng nơ-ron giống như một "hộp đen". Các cơ quan quản lý giao thông khó mà tin tưởng hệ thống nếu không hiểu được lý do tại sao AI lại ra quyết định chuyển đèn.
                - **Khả năng mở rộng:** Tính toán cho hàng ngàn ngã tư đòi hỏi tài nguyên khổng lồ, điều kiện phần cứng hiện tại trên các thiết bị biên (edge devices) ở tủ đèn tín hiệu còn hạn chế
        - [X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner](https://arxiv.org/pdf/2404.12090): giới thiệu một giải pháp đột phá giúp hệ thống điều khiển đèn giao thông bằng Học tăng cường (RL) có thể học hỏi và áp dụng dễ dàng giữa các thành phố khác nhau sử dụng Transformer
            - Tóm tắt paper
                
                **1. Vấn đề của các nghiên cứu trước đây** Các mô hình điều khiển đèn giao thông bằng RL hiện tại thường chỉ hoạt động tốt trên một "kịch bản" cố định (một thành phố hoặc mạng lưới đường cụ thể). Nếu muốn đem hệ thống này đi áp dụng cho một thành phố khác, người ta phải tốn chi phí xây dựng lại môi trường mô phỏng và huấn luyện AI lại từ đầu.
                
                Một số nghiên cứu Meta-RL trước đó (như MetaLight, GESA) đã giải quyết được việc chuyển giao sang môi trường mới nhưng lại đi theo hướng "đơn tác tử", tức là bỏ qua sự hợp tác, phối hợp giữa các ngã tư với nhau. Các nỗ lực dùng "đa tác tử" (như mô hình MetaGAT) lại dựa vào Mạng nơ-ron đồ thị (GNN) để giao tiếp, vốn chỉ truyền tải "trạng thái" quan sát (*o*) mà bỏ qua thông tin quan trọng về các "hành động" (*a*) và "phần thưởng" (*r*) trước đó, khiến mô hình học tập thiếu ổn định và bị suy giảm hiệu suất khi sang môi trường mới.
                
                **2. Kiến trúc cốt lõi: Transformer on Transformer (TonT)** Để giải quyết đồng thời việc "hợp tác cục bộ" và "tổng quát hóa chéo thành phố", bài báo đề xuất kiến trúc **X-Light** với 2 mạng Transformer xếp chồng lên nhau:
                
                - **Lower Transformer (Tầng thấp - Hợp tác cục bộ):** Tầng này làm nhiệm vụ liên kết thông tin giữa ngã tư mục tiêu và các ngã tư lân cận. Khác với mạng GNN, Lower Transformer thu thập toàn bộ dữ liệu (MDP) bao gồm: Quan sát (*o*), Hành động (*a*) và Phần thưởng (*r*). Điều này giúp mô hình hiểu sâu sắc hơn về nguyên nhân và kết quả khi tương tác với các ngã tư bên cạnh.
                - **Upper Transformer (Tầng cao - Ra quyết định tổng quát):** Hoạt động như một bộ "siêu học tập" (Meta-RL learner). Tầng này xử lý chuỗi dữ liệu lịch sử theo thời gian để rút ra các quy luật động lực học chung của môi trường giao thông. Nhờ đó, AI học được các đặc tính không phụ thuộc vào một kịch bản cụ thể nào (scenario-agnostic), đảm bảo khả năng linh hoạt chuyển giao.
                
                **3. Các kỹ thuật tối ưu hóa quan trọng**
                
                - **Dynamic Predictor (Bộ dự đoán động):** Một mạng dự đoán được gắn vào Upper Transformer để ép AI phải dự báo trước tương lai của chuỗi trạng thái. Điều này giúp hệ thống mô hình hóa các quy luật chéo kịch bản tốt hơn.
                - **Residual Link (Kết nối thặng dư):** Trải qua nhiều lớp Transformer, dữ liệu có thể bị "trừu tượng hóa" quá mức. X-Light cộng trực tiếp thông tin quan sát ban đầu của ngã tư vào quyết định cuối cùng, giúp AI xử lý rất tốt các kịch bản mạng lưới giao thông đơn giản, nơi chỉ cần chú ý vào trạng thái cục bộ là đủ.
                - **Multi-scenario Co-training (Đồng huấn luyện đa kịch bản):** X-Light dùng module tích hợp chung (GPI) để gộp mọi dạng ngã tư (3 ngã, 4 ngã...) về một chuẩn chung, sau đó trộn ngẫu nhiên dữ liệu từ nhiều mạng lưới đường của các thành phố khác nhau vào để huấn luyện. Kỹ thuật này cải thiện đáng kể tính ổn định.
                
                **4. Kết quả đạt được**
                
                - **Thành công trong chuyển giao Zero-shot:** Khi được áp dụng thẳng vào một hệ thống giao thông ở thành phố mới (chưa từng được nhìn thấy trong quá trình huấn luyện), X-Light vẫn điều phối giao thông mượt mà. Nó vượt qua tất cả các mô hình tốt nhất hiện tại, giảm thời gian di chuyển trung bình thêm 7.91%, và lên tới 16.3% trong mạng lưới đường hình lưới.
                - **Tốc độ hội tụ tốt:** X-Light đạt tốc độ hội tụ khi huấn luyện nhanh gần gấp đôi so với các phương pháp dùng Mạng nơ-ron đồ thị (GNN) như MetaGAT
        - [**A Survey on Multi-agent Reinforcement Learning for Adaptive Transportation Solutions (Liang et al.)](https://link.springer.com/article/10.1007/s42979-025-04475-3):** Khảo sát tập trung vào việc áp dụng Multi-Agent Reinforcement Learning (MARL) trong các bài toán giao thông, bao gồm traffic signal control. Một định hướng quan trọng là thu thập dữ liệu offline dataset ngay từ đầu để hỗ trợ nghiên cứu Generalization và Offline RL trong tương lai.
    - Lý do lựa chọn làm theo hướng presslight
        - **Bám sát mục tiêu môn học và bản chất RL:** PressLight sử dụng mạng Deep Q-Network (DQN) kinh điển, kết nối trực tiếp với các khái niệm cốt lõi đã học như MDP, hàm giá trị (Value Function) và phương trình Bellman. Dù việc tự tay lập trình các mô hình phức tạp như Transformer từ đầu là kỹ năng không xa lạ, nhưng chọn DQN giúp đồ án tập trung hoàn toàn vào việc mổ xẻ cơ chế Học tăng cường (quá trình thử - sai, thiết kế hàm phần thưởng), thay vì bị cuốn vào việc tinh chỉnh kiến trúc Deep Learning đồ sộ.
        - **Tính khả thi về tài nguyên và thời gian:** X-Light yêu cầu khối lượng dữ liệu khổng lồ từ nhiều thành phố khác nhau và cơ chế đồng huấn luyện (co-training) song song để đạt được Zero-shot. Điều này đòi hỏi lượng VRAM GPU rất lớn và thời gian huấn luyện dài, hoàn toàn không khả thi với nguồn tài nguyên và thời lượng của một đồ án môn học. PressLight tinh gọn, train nhanh và chạy mượt mà trên các tài nguyên miễn phí như Google Colab.
        - **Tính thực dụng cao:** Tập trung giải quyết triệt để và tối ưu hóa cực tốt một bài toán cụ thể tại địa phương luôn mang lại giá trị thực tiễn lớn — yếu tố ghi điểm cực mạnh trong các hội đồng đánh giá hay các sân chơi công nghệ. Lý thuyết Max-Pressure mang lại nền tảng toán học vững chắc, giúp mô hình dù đơn giản nhưng vẫn đạt hiệu suất vượt trội.
- Công thức
    - Không gian Quan sát (State Space):
        - Tại mỗi bước thời gian $t$, tác tử tại nút giao $i$ sẽ thu thập thông tin cục bộ để tạo thành véc-tơ trạng thái $s_i$. Để đảm bảo tác tử không bị "mù" bối cảnh và tránh hiện tượng nhấp nháy đèn liên tục, trạng thái cần bao gồm mật độ xe và pha đèn hiện hành:
            
            $s_i = [\mathbf{q}_{in}, \mathbf{q}_{out}, \mathbf{p}_{current}]$
            
        - $\mathbf{q}_{in} = [q_{in\_1}, q_{in\_2}, \dots, q_{in\_k}]$*: Số lượng xe đang chờ ở các làn lối vào.*
        - $*\mathbf{q}_{out} = [q_{out\_1}, q_{out\_2}, \dots, q_{out\_k}]$: Số lượng xe ở các làn lối ra.*
        - $*\mathbf{p}_{current}*$: Véc-tơ one-hot đại diện cho pha đèn xanh hiện tại.
        - Đặc điểm: Đây là quan sát cục bộ (Local Observation), giúp đảm bảo tính chất Học độc lập (Independent Learning) mà không cần nới rộng không gian trạng thái bằng dữ liệu từ các nút giao lân cận.
    - Không gian Hành động (Action Space):
        - Tác tử chọn pha đèn xanh tiếp theo từ một tập hợp các pha hữu hạn không xung đột: $\mathcal{A}i = \{a_0, a_1, \dots, a{K-1}\}$
        - $K$: Tổng số pha của nút giao (thường từ 4 đến 8 pha).
        - Mạng Deep Q-Network (DQN) sẽ xuất ra giá trị Q (Q-value) cho từng pha. Tác tử sẽ chọn hành động mang lại giá trị Q cao nhất (Cơ chế `argmax`).
    - Hàm Phần thưởng (Reward Function) - Max Pressure:
        - Hàm phần thưởng tại bước $t$ được định nghĩa bằng giá trị âm của Áp lực giao thông ($P_i$): 
        $r_i = - P_i = - \left( \sum_{l \in \text{in}} q_l - \sum_{m \in \text{out}} q_m \right)$
        - **Lý do lựa chọn Max Pressure:**
            - **Nền tảng lý thuyết vững chắc:** Theo lý thuyết dòng chảy mạng lưới (Network Flow Theory), việc tối thiểu hóa áp lực cục bộ $P_i$ tương đương với việc tối đa hóa thông lượng (throughput) trên toàn mạng lưới.
            - **Loại bỏ Trọng số thủ công (Heuristic Tuning):** Không cần phải tinh chỉnh (tuning) các trọng số phức tạp như các hàm phần thưởng kết hợp đa mục tiêu
             $\sum (w_i \cdot \text{metric}_i)$.
            - **Độ phức tạp $O(1)$:** Việc tính toán cực kỳ nhẹ do chỉ dựa trên dữ liệu đếm xe hiện tại, không đòi hỏi các phép tính phức tạp như theo dõi thời gian chờ của từng xe.
    - Kiến trúc Mạng Deep Q-Network (DQN):
        - Mạng Nơ-ron nhân tạo đóng vai trò xấp xỉ hàm giá trị hành động:
        $Q(s_i, a_i; \theta) = \text{MLP}(s_i)$
        - Cấu trúc mạng (MLP): Input Layer (Kích thước $s_i$) $\rightarrow$ Fully Connected (128 units, ReLU) $\rightarrow$ Fully Connected (64 units, ReLU) $\rightarrow$ Output Layer (Kích thước $|A|$, Linear).
        - Siêu tham số (Hyperparameters):
            - Tối ưu hóa (Optimizer): Adam
            - Tốc độ học (Learning Rate): $10^{-3}$
            - Hệ số chiết khấu (Discount factor - $\gamma$): $0.95$
            - Chiến lược khám phá: $\epsilon$-greedy (giảm dần từ $1.0$ xuống $0.01$).
    - Cơ chế Chia sẻ tham số (Parameter Sharing) - Yếu tố then chốt:
        - Tất cả $N$ tác tử (ví dụ: lưới $3\times3$ có 9 tác tử) đều dùng chung một mạng DQN với bộ trọng số $\theta_{shared}$ duy nhất:
        $\forall i \in \{1, 2, \dots, N\}: \pi_i(s_i) = \arg\max_{a} Q(s_i, a; \theta_{shared})$
        - **Lợi ích đột phá:** Thay vì hệ thống phải học $N \times |\theta|$ tham số, giờ đây chỉ cần học $|\theta|$ tham số. Khả năng sử dụng dữ liệu (Sample Efficiency) tăng lên gấp $N$ lần vì tại mỗi bước mô phỏng, hệ thống thu thập được $N$ mẫu huấn luyện. Việc này giúp MARL hội tụ cực nhanh.
    - Huấn luyện (Experience Replay & Huber Loss):
        - **Cơ chế Replay Buffer:** Hệ thống lưu trữ các tương tác vào Replay Buffer dưới dạng các tuple $(s, a, r, s', \text{done})$. Trong quá trình huấn luyện, mô hình sẽ lấy mẫu ngẫu nhiên (sample) từ bộ nhớ này để phá vỡ sự tương quan theo thời gian giữa các mẫu dữ liệu.
        - **Công thức Hàm mất mát (Loss Function):** Mạng DQN được tối ưu hóa bằng cách giảm thiểu sai số giữa giá trị Q dự đoán và giá trị mục tiêu (Target). Để tránh hiện tượng bùng nổ gradient (gradient explosion) khi phần thưởng $r$ biến động quá mạnh trong các kịch bản kẹt xe nặng, hệ thống sử dụng Huber Loss thay vì MSE thông thường:
        $\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \mathcal{H} \left( y_j - Q(s_j, a_j; \theta) \right) \right]$
        - Trong đó, giá trị mục tiêu $y_j$ được tính bởi mạng Target (với bộ trọng số $\theta^-$):
        $y_j = r_j + \gamma \max_{a'} Q(s'_j, a'; \theta^-)$
        - Hàm Huber $\mathcal{H}(x)$ được định nghĩa từng phần:
        
        $\mathcal{H}(x)=\begin{cases} \frac{1}{2}x^2 & \text{nếu } |x| \le \delta \quad \text{(Hành xử như MSE ở vùng sai số nhỏ)} \\ \delta \left( |x| - \frac{1}{2}\delta \right) & \text{nếu } |x| > \delta \quad \text{(Hành xử như MAE ở vùng sai số lớn)} \end{cases}$
        
        - Cấu hình Siêu tham số (Tối ưu cho Sprint 14 ngày):
            - Kích thước bộ nhớ (buffer_size): $5,000$
            - Kích thước mẻ học (batch_size): $64$
            - Chu kỳ cập nhật mạng mục tiêu (target_update_freq - $C$): Mỗi $200$ steps
            - Ngưỡng Huber ($\delta_{huber}$): $1.0$
        - **Lý do chọn cấu hình này:**
        1. **Target Network (**$\theta$**):** Việc tách biệt mạng dự đoán ($\theta$) và mạng mục tiêu ($\theta^-$) cập nhật chậm (sau mỗi 200 steps) giúp neo giữ giá trị $y_j$ ổn định, tránh tình trạng mô hình tự rượt đuổi chính nó (chasing its own tail) gây phân kỳ.
        2. **Kích thước Buffer & Batch:** Cặp thông số (5000 / 64) là điểm ngọt (sweet spot) được tinh chỉnh riêng cho thời gian dự án ngắn. Nếu tăng lên mức tiêu chuẩn (50,000 / 256), mô hình sẽ học ổn định hơn nhưng đòi hỏi thời gian huấn luyện dài hơn rất nhiều, không phù hợp với quỹ thời gian 14 ngày.
    - Baseline - MaxPressure Actuated Control (Cơ sở đối sánh):
        - Đây là thuật toán Rule-based thuần túy dùng công thức vật lý để tự chuyển đèn mà không dùng học máy. Baseline này đóng vai trò là cận dưới (lower bound) để so sánh hiệu năng của AI:
        $a^* = \arg\max_{k \in K} \sum_{l \in \text{phase}k} \left( q{in}(l) - q_{out}(l) \right)$
        - **Cơ chế:** Tại mỗi nhịp ra quyết định, thuật toán tự động tính toán tổng áp lực của từng tổ hợp pha đèn. Pha nào có áp lực cao nhất sẽ được chọn để bật đèn xanh. Mạng DQN sau khi huấn luyện bắt buộc phải cho ra kết quả Average Travel Time và Throughput vượt trội hơn hệ thống Baseline này.
- **Roadmap**
    
    💡 **Mục tiêu:** Cung cấp bức tranh toàn cảnh về quy trình xây dựng, huấn luyện và đánh giá hệ thống Bản sao số (Digital Twin) đa tác tử điều khiển tín hiệu giao thông, đồng thời tạo ra một bộ khung (sandbox) chuẩn để có thể tái sử dụng cho các hệ thống như điều phối tự hành (AGV) trong nhà kho thông minh.
    
    - 🗺️ TỔNG QUAN PIPELINE
        - **INPUT:** Trình mô phỏng giao thông CityFlow (cấu hình mạng lưới và luồng phương tiện bất đối xứng).
        - **GIAI ĐOẠN 1 (Môi trường & Baseline):** Thiết lập môi trường giả lập → Xây dựng thuật toán cơ sở (Luật vật lý MaxPressure) làm mốc đối sánh.
        - **GIAI ĐOẠN 2 (Lõi AI & Huấn luyện):** Xây dựng mạng Deep Q-Network (DQN) → Tích hợp Chia sẻ tham số (Parameter Sharing) → Chạy vòng lặp huấn luyện.
        - **GIAI ĐOẠN 3 (Đánh giá & Ghi log):** Chạy thử nghiệm đối đầu (AI vs Baseline) → Trích xuất tập dữ liệu ngoại tuyến (Offline Logging) → Phân tích biểu đồ hội tụ.
        - **GIAI ĐOẠN 4 (Đóng gói API & Web):** Xây dựng Backend API đọc file log → Dựng Frontend Dashboard trực quan hóa → Hoàn thiện Báo cáo.
        - **OUTPUT:** Tập dữ liệu Offline (CSV/SQLite) + Bảng phân tích hiệu năng + Giao diện Web Demo + Báo cáo kỹ thuật.
    - GIAI ĐOẠN 1: MÔI TRƯỜNG & BASELINE (Thiết lập cơ sở đối sánh)
        
        Mục tiêu: Dựng xong sân chơi và có ngay một con số chuẩn (lower bound) để AI nhắm tới.
        
        - **Bước 1: Khởi tạo môi trường giả lập (Simulator)**
            - Cài đặt công cụ CityFlow.
            - Viết script tự động sinh tệp cấu hình mạng lưới giao thông (`roadnet.json`) dạng lưới 3x3 và tệp luồng phương tiện bất đối xứng (`flow.json`).
        - **Bước 2: Xây dựng Baseline (Luật vật lý)**
            - Lập trình thuật toán MaxPressure Actuated Control. Đây là hệ thống tự chuyển pha đèn dựa trên công thức áp lực thuần túy, không dùng học máy.
        - **Bước 3: Chạy thử nghiệm Baseline**
            - Cho Baseline chạy trên CityFlow và dùng thư viện Python cơ bản (như `matplotlib` hoặc `pandas`) để in ra chỉ số Thời gian chờ trung bình và Thông lượng. Ghi nhận con số này.
    - GIAI ĐOẠN 2: LÕI AI & HUẤN LUYỆN (Xây dựng bộ não MARL)
        - Mục tiêu: Code xong DQN, tích hợp Parameter Sharing và ép mô hình hội tụ.
        • **Bước 4: Định nghĩa môi trường Học tăng cường (Environment)**
            ◦ Lập trình class `TrafficEnv` bằng Python.
            ◦ Trích xuất chính xác trạng thái $s_i = [\mathbf{q}_{in}, \mathbf{q}_{out}, \mathbf{p}_{current}]$.
            ◦ Lập trình hàm phần thưởng $r_i = - P_i$.
        • **Bước 5: Thiết lập Mạng Nơ-ron (Deep Q-Network)**
            ◦ Sử dụng PyTorch để code kiến trúc mạng DQN.
            ◦ Khởi tạo bộ nhớ trải nghiệm (Replay Buffer) và thiết lập chiến lược khám phá $\epsilon$-greedy để tác tử thử nghiệm các pha đèn.
        • **Bước 6: Tích hợp Chia sẻ tham số (Parameter Sharing)**
            ◦ Gộp toàn bộ 9 tác tử tại 9 nút giao lại để dùng chung một mạng trọng số $\theta$.
        • **Bước 7: Huấn luyện & Tối ưu hóa**
            ◦ Cài đặt Target Network và Huber Loss để tránh bùng nổ gradient.
            ◦ Chạy vòng lặp huấn luyện (train loop) qua hàng trăm episodes. Lưu lại file trọng số mô hình tốt nhất (`.pth`).
    - GIAI ĐOẠN 3: ĐÁNH GIÁ & GHI LOG (Validate mô hình)
        
        Mục tiêu: Đảm bảo mô hình RL thực sự thông minh hơn Baseline thông qua các bài test đối đầu trực tiếp, chưa cần dùng đến web.
        
        - **Bước 8: Đánh giá đối sánh (Head-to-head)**
            - Tải file `.pth` đã train. Cho cả Baseline và mạng IDQN chạy song song trên cùng một luồng phương tiện.
        - **Bước 9: Thu thập Dataset ngoại tuyến (Logging)**
            - Ghi xuất toàn bộ lịch sử tương tác `(state, action, reward, next_state, done)` của cả AI và Baseline ra các tệp CSV hoặc SQLite.
        - **Bước 10: Phân tích chuyên sâu (Bằng Python)**
            - Sử dụng `seaborn` và `matplotlib` để đọc file CSV vừa sinh ra.
            - Vẽ biểu đồ đường cong học tập (Learning Curve) để chứng minh loss giảm, reward tăng. Vẽ box-plot so sánh độ trễ xe giữa AI và Baseline.
    - GIAI ĐOẠN 4: ĐÓNG GÓI API & GIAO DIỆN WEB (Trình diễn sản phẩm)
        
        Mục tiêu: Bọc (wrap) mọi thứ đã hoàn thiện ở trên thành một sản phẩm có thể tương tác và demo trực quan.
        
        - **Bước 11: Đóng gói Backend API**
            - Khởi tạo project FastAPI.
            - Viết các endpoint API (ví dụ: `/replay`) có nhiệm vụ đọc file log CSV đã sinh ra ở Giai đoạn 3 và trả về tọa độ xe, trạng thái đèn theo từng giây.
        - **Bước 12: Khởi tạo Frontend Dashboard**
            - Dựng khung project bằng Next.js. Gọi API từ FastAPI.
            - Vẽ lại lưới đường 2D (bằng Canvas hoặc SVG). Render các phương tiện (xe) di chuyển và đèn tín hiệu đổi màu nhịp nhàng theo dữ liệu từ file log.
        - **Bước 13: Đóng gói dự án & Báo cáo**
            - Nhúng các biểu đồ tĩnh (đã vẽ ở Bước 10) lên web để có cái nhìn tổng quan.
            - Hoàn thiện báo cáo kỹ thuật giải thích cách RL giải quyết bài toán giao thông. Đóng gói mã nguồn.
- Thu thập Data & Setup Simulator & Thu thập Data & Setup Simulator
    - Tại sao chọn CityFlow?
        
        Trong các dự án MARL mô phỏng giao thông, trình giả lập đóng vai trò là môi trường tương tác cốt lõi. Dự án này quyết định chọn **CityFlow** thay vì SUMO (Simulation of Urban MObility) dựa trên các tiêu chí cực kỳ phù hợp với quỹ thời gian ngắn và yêu cầu huấn luyện:
        
        - **CityFlow (Lựa chọn tối ưu):**
            - Tốc độ mô phỏng cực nhanh (nhanh hơn SUMO từ 10x đến 50x lần), hỗ trợ chạy đa luồng rất tốt cho quá trình huấn luyện RL cần lặp lại hàng vạn episodes.
            - Hỗ trợ Native Python API, dễ dàng tích hợp trực tiếp với PyTorch.
            - Cấu hình môi trường bằng JSON đơn giản, minh bạch.
            - Là trình giả lập được sử dụng chính thức trong nghiên cứu gốc của thuật toán PressLight.
        - **SUMO (Loại bỏ trong phase này):**
            - Dù chi tiết về mặt vật lý vi mô, nhưng SUMO quá nặng và tốc độ chạy chậm.
            - Cấu hình bằng XML cồng kềnh, API giao tiếp (TraCI) tốn chi phí hiệu năng (overhead) lớn, không phù hợp cho một sprint phát triển nhanh.
        
    - Cấu trúc tệp cấu hình giả lập
        
        Môi trường CityFlow được khởi tạo thông qua hai tệp JSON chính:
        
        - **`roadnet.json` (Topology):** Mô tả cấu trúc mạng lưới đường đi. Bao gồm danh sách các nút giao (intersections), các đoạn đường (roads), số làn xe mỗi chiều, chiều dài đường và tốc độ tối đa. *Ví dụ: Một lưới 3×3 sẽ tự động sinh ra 9 nút giao và 24 đoạn đường hai chiều.*
        - **`flow.json` (Lưu lượng):** Mô tả dòng xe chạy. Cấu hình chi tiết từng luồng phương tiện với các tham số: `startTime`, `endTime`, `route` (danh sách ID các đoạn đường xe đi qua), và khoảng thời gian phát sinh xe (interval).
    - Quy trình sinh dữ liệu (Data Generation Pipeline)
        - Sinh cấu hình tự động: Chạy script Python để tự động tạo roadnet.json (lưới NxN) và flow.json (lưu lượng xe tuân theo phân phối Poisson để tạo tính ngẫu nhiên nhưng có kiểm soát).
        - Chạy Baseline: Kích hoạt thuật toán Rule-based MaxPressure chạy mô phỏng qua 50 episodes.
        - Lưu toàn bộ quỹ đạo $(s, a, r, s')$ vào tệp buffer_baseline.csv.Chạy Mô hình DQN: Tiến hành huấn luyện RL online trong hơn 500 episodes. Liên tục ghi log mọi tương tác của agent vào tệp buffer_dqn.csv.
        - Hợp nhất Dataset: Gộp chung hai tệp buffer thành một tập dữ liệu chuẩn offline_dataset.csv (hoặc import vào SQLite) làm tài nguyên nghiên cứu Offline RL cho các phase sau.
    - Lược đồ Nhật ký dữ liệu (Data Log Schema)
        
        Tập dữ liệu tĩnh (Offline Dataset) được lưu trữ theo cấu trúc các trường (columns) sau nhằm tối đa hóa khả năng tái sử dụng:
        
        - `columns`: `[episode, step, agent_id, state_vec, action, reward, next_state_vec, done, atl, throughput]`
        - **Đặc điểm kỹ thuật:**
            - `state_vec` và `next_state_vec` được lưu dưới dạng JSON string (mảng số thực) để tiết kiệm không gian.
            - `agent_id`: Định danh ID của nút giao, dùng để phân biệt khi chia sẻ tham số.
            - `atl` (Average Travel Time) và `throughput`: Được lưu sẵn ở mỗi bước để có thể vẽ ngay biểu đồ (Line chart/Recharts) trên web cấp episode mà không cần phải chạy lại bộ giả lập.
    - Kịch bản Lưu lượng (Flow Scenarios)
        - Để đánh giá toàn diện năng lực của thuật toán DQN, hệ thống được thiết lập để kiểm thử trên 3 kịch bản lưu lượng khác nhau:
            - Low (Thấp): ~300 xe/giờ. Dùng làm Base training phase. (Kỳ vọng: Baseline và DQN có hiệu năng tương đương, không xảy ra ùn tắc).
            - Medium (Trung bình): ~600 xe/giờ. Dùng làm Evaluation phase.
            - High (Cao - Stress test): ~900 xe/giờ. Đây là môi trường kẹt xe nặng. (Kỳ vọng: DQN bắt đầu vượt trội hoàn toàn so với Baseline nhờ khả năng học được tầm nhìn dài hạn $\gamma$, tránh được hiệu ứng lan truyền điểm ùn tắc).
    - Danh sách Dependencies cốt lõi (Tech Stack)
        
        Hệ thống sử dụng các thư viện gọn nhẹ và hiện đại nhất để chia tách 3 lớp AI - Backend - Frontend:
        
        - **Lõi Mô phỏng & AI:** `cityflow`, `torch >= 2.0` (PyTorch), `numpy`, `pandas`.
        - **Lớp Backend API:** `fastapi`, `uvicorn`, `pydantic`, `sqlite3` (lưu trữ database nội bộ nhẹ).
        - **Lớp Giao diện (Frontend):** `next.js 14`, `recharts` (vẽ biểu đồ real-time).