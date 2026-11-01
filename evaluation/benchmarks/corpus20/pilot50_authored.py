"""Source-authored pilot questions, not generated model answers or reviewed gold."""
QUESTIONS=[]
def add(alias,question,answer,refs,attribute,value,condition=None,categories=('text',)):
    QUESTIONS.append(dict(alias=alias,question=question,reference_answer=answer,refs=refs,
        attribute=attribute,value=value,condition=condition,categories=list(categories)))

add('TWIN','Digital Twin Software kết hợp hai nền tảng nào và hỗ trợ những công việc gì trước khi mua thiết bị thật?',
    'Kết hợp realvirtual.io với Unity; hỗ trợ kiểm tra bố trí thiết bị trước khi mua, đào tạo người vận hành và giảm thời gian thiết lập/điều chỉnh.',
    [('TWIN',4491)],'platforms_and_uses',['realvirtual.io','Unity','kiểm tra bố trí trước khi mua','đào tạo','giảm thời gian thiết lập'])
add('TWIN','realvirtual.io Professional hỗ trợ những giao diện và kết nối nào để mô phỏng thiết bị?',
    'Hỗ trợ TwinCAT và OPC UA, kết nối PLC và software wiring; thư viện asset có băng tải và động cơ.',
    [('TWIN',4506)],'simulation_interfaces',['TwinCAT','OPC UA','PLC connections','software wiring'])
add('TWIN','realvirtual.io Professional là phần bổ sung cho nền tảng nào, và thư viện của nó có những ví dụ thiết bị gì?',
    'Là add-on cho Unity; các ví dụ asset được nêu là băng tải và động cơ.',
    [('TWIN',4506)],'platform_and_assets',['Unity','conveyors','motors'])
add('BOX','RC8 Controller Protective Box Light có cấp bảo vệ và khối lượng bao nhiêu? Khối lượng đó có gồm controller không?',
    'Cấp bảo vệ IP54 hoặc tương đương; khoảng 17,5 kg, không gồm robot controller.',
    [('BOX',4558)],'protection_and_weight',dict(protection='IP54 or equivalent',weight_kg=17.5,includes_controller=False),categories=('table','condition'))
add('BOX','Những trường hợp nào phải dùng RC8 Controller Protective Box nguyên bản thay cho bản Light?',
    'VM Series; VS068/087 có trục bổ sung; MC8 có tổng công suất động cơ từ 2.000 W trở lên.',
    [('BOX',4547),('BOX',4562),('BOX',4563)],'light_variant_exclusions',['VM Series','VS068/087 with additional axis','MC8 total motor capacity >= 2000 W'],categories=('condition','footnote'))
add('BOX','Giới hạn nhiệt độ và độ ẩm của Protective Box Light khi vận hành khác khi lưu kho/vận chuyển thế nào?',
    'Vận hành: 0–40 °C, không quá 90% RH, không ngưng tụ. Lưu kho/vận chuyển: −10–60 °C, không quá 75% RH, không ngưng tụ.',
    [('BOX',4558)],'environment_limits',dict(operation='0–40 °C; <=90% RH; non-condensing',storage_transport='-10–60 °C; <=75% RH; non-condensing'),categories=('table','comparison','condition'))
add('PLUG','Khi điều chỉnh khe hở bugi Iridium Power, dụng cụ không được chạm vào những bộ phận nào và phải kiểm tra lại bằng gì?',
    'Không được chạm điện cực trung tâm iridium hoặc phần sứ; kiểm tra lại bằng dụng cụ đo khe hở đã hiệu chuẩn.',
    [('PLUG',4995),('PLUG',4996)],'gapping_precautions',['không chạm điện cực trung tâm iridium','không chạm phần sứ','dụng cụ đo đã hiệu chuẩn'],categories=('procedure','condition'))
add('PLUG','Với bugi 14 mm flat seat mới, ren không bôi trơn, mô-men siết cho nắp máy gang và nhôm lần lượt là bao nhiêu?',
    'Nắp máy gang: 35–40 N·m; nắp máy nhôm: 20–30 N·m. Bảng áp dụng cho bugi mới với ren không bôi trơn.',
    [('PLUG',5000),('PLUG',5017)],'14mm_flat_seat_torque',dict(cast_iron_Nm=[35,40],aluminum_Nm=[20,30]),'Bugi mới; ren không bôi trơn',categories=('table','comparison','footnote','condition'))
add('PLUG','Nếu ren bugi được bôi trơn thì tài liệu yêu cầu điều chỉnh mô-men siết như thế nào và vì sao?',
    'Giảm mô-men siết khoảng một phần ba để tránh siết quá chặt.',
    [('PLUG',5017)],'lubricated_thread_torque_reduction','approximately 1/3','Ren được bôi trơn',categories=('footnote','condition'))
add('RACE','Trong sơ đồ và chú giải Iridium Racing, nhãn A và G chỉ hai điện cực có vật liệu và đường kính nào?',
    'A: điện cực mass bằng 100% platinum, 0,8 mm. G: điện cực trung tâm iridium siêu mảnh, đường kính 0,4 mm.',
    [('RACE',5113),('RACE',5132)],'electrodes_A_G',dict(A='100% platinum ground electrode, 0.8 mm',G='iridium center electrode, 0.4 mm'),categories=('figure_text','comparison'))
add('RACE','Lớp silicone oil trên phần cách điện của bugi Iridium Racing được dùng để ngăn vấn đề gì trước cuộc đua?',
    'Đặc tính đẩy nước giúp hạn chế ẩm và carbon trên bề mặt cách điện, tránh giảm khả năng cách điện và các vấn đề khởi động do muội carbon/bridging.',
    [('RACE',5123)],'silicone_oil_function','Giữ bề mặt cách điện tránh ẩm và carbon; bảo vệ khả năng cách điện',categories=('condition','text'))
add('INSUL','Trong quy tắc mã bugi Iridium Racing, ký hiệu I biểu thị gì và U, XU tương ứng cỡ ren nào?',
    'I là Iridium; U là ren 10 mm; XU là ren 12 mm.',
    [('INSUL',5246)],'plug_code_mapping',dict(I='Iridium',U='10 mm',XU='12 mm'),categories=('table','identifier'))
add('INSUL','So sánh IU01* và IXU01*: đường kính ren và chiều cao bugi của mỗi loại là bao nhiêu?',
    'IU01*: đường kính ren 10 mm, chiều cao 50 mm. IXU01*: đường kính ren 12 mm, chiều cao 51 mm.',
    [('INSUL',5251)],'IU01_IXU01_dimensions',dict(IU01=dict(thread_mm=10,height_mm=50),IXU01=dict(thread_mm=12,height_mm=51)),categories=('table','comparison'))
add('COMP','Trong bảng chẩn đoán máy nén, khi áp suất phía cao bất thường và ngay sau khi dừng máy nén giảm nhanh về khoảng 28 psi, nguyên nhân và xử lý được nêu là gì?',
    'Có không khí lọt vào chu trình lạnh; xả môi chất, hút chân không đầy đủ rồi nạp lại.',
    [('COMP',6008)],'high_pressure_drop_28psi',dict(cause='Air entered in refrigerating cycle',remedy='Discharge refrigerant, evacuate sufficiently, recharge'),categories=('table','condition','procedure'))
add('COMP','Khi thấy các bu-lông của máy nén bị dính dầu, bảng chẩn đoán nêu vị trí rò gas và biện pháp gì?',
    'Gas rò từ các phần bắt bu-lông; siết lại bu-lông.',
    [('COMP',6012)],'oil_stained_bolts',dict(cause='Gas leaking from bolted parts',remedy='Retighten bolts'),categories=('table','condition'))
add('CLUTCH','Khi ly hợp không hút nhưng hệ thống dây điện bình thường, nguyên nhân và biện pháp trong bảng Clutch là gì?',
    'Cuộn coil bị đứt mạch (coil open); thay thế.',
    [('CLUTCH',4599)],'no_attraction_normal_wiring',dict(cause='Coil open',remedy='Replace'),'Wiring normal',categories=('table','condition'))
add('CLUTCH','Khi đầu ra van tiết lưu không lạnh và không cảm nhận được chênh lệch nhiệt đầu vào/đầu ra, bảng nêu hai nguyên nhân và cách xử lý tương ứng nào?',
    'Ống cảm nhiệt rò gas: thay van tiết lưu. Môi chất trong chu trình không đủ: sửa điểm rò gas rồi nạp môi chất.',
    [('CLUTCH',4600)],'expansion_valve_outlet_not_cold',dict(heat_sensing_tube_leak='replace expansion valve',insufficient_refrigerant='repair leak and charge refrigerant'),categories=('table','condition','comparison'))
add('DUALAC','Trong ca A/C hai dàn làm mát kém, máy nén và van tiết lưu phía trước đã được kiểm tra hoặc xử lý thế nào trước khi xác định nguyên nhân phía sau?',
    'Máy nén đã được xác nhận hoạt động tốt; van tiết lưu phía trước vừa được thay gần đây.',
    [('DUALAC',5375)],'prior_diagnostic_state',dict(compressor='confirmed OK',front_expansion_valve='recently replaced'),categories=('condition','text'))
add('DUALAC','Ống cảm nhiệt của van tiết lưu phía sau bị lỏng gây làm mát kém qua cơ chế nào?',
    'Không đọc đúng nhiệt độ đầu ra giàn bay hơi, khiến van mở quá mức và nhiều môi chất hơn chảy về dàn lạnh phía sau.',
    [('DUALAC',5376)],'loose_rear_sensing_tube_effect','Đọc sai nhiệt độ đầu ra giàn bay hơi → van mở quá mức → tăng dòng môi chất về dàn sau',categories=('causal_relation','condition'))
add('DUALAC','Tài liệu hướng dẫn cố định lại ống cảm nhiệt phía sau ở đâu và cần tham khảo gì về cách nhiệt?',
    'Cố định chặt vào rãnh phù hợp trên ống áp suất thấp của dàn sau; tham khảo thông tin dịch vụ của nhà sản xuất để đáp ứng yêu cầu cách nhiệt.',
    [('DUALAC',5378)],'rear_sensing_tube_correction','Secure to appropriate groove on rear cooling unit low-pressure tube; consult manufacturer insulation requirements',categories=('procedure','condition'))
add('RING','Sau khi lắp máy nén và tháo/lắp ly hợp từ, tiếng ồn kèm làm mát kém có thể liên quan đến chi tiết lắp sai nào?',
    'Có thể vòng hãm giữ stator và rotor được lắp không đúng.',
    [('RING',4690),('RING',4691)],'post_installation_noise_possible_cause','Improperly installed snap ring securing stator and rotor','Sau lắp máy nén, có magnetic clutch R&R',categories=('condition','causal_relation'))
add('RING','Vòng hãm có mép vát phải quay về phía nào khi lắp vào máy nén, và phải bảo đảm điều gì?',
    'Mép vát quay ra xa máy nén; vòng phải ngồi hoàn toàn vào vị trí và giữ chắc stator, rotor.',
    [('RING',4692)],'snap_ring_orientation','Beveled edge facing away from compressor; fully seated and securely retaining stator/rotor',categories=('procedure','spatial_text'))
add('FLASH','Khi ly hợp máy nén không đóng và đèn A/C nhấp nháy trong ca được nêu, bộ phận nào bị ngắt và gây ra hai triệu chứng đó?',
    'Rơ-le ly hợp từ bị ngắt, làm ly hợp nhả và đèn A/C nhấp nháy.',
    [('FLASH',5469),('FLASH',5470)],'clutch_flashing_cause','Magnetic clutch relay disengaging',categories=('condition','causal_relation'))
add('FLASH','Nếu có đúng các triệu chứng của tài liệu và rơ-le mang Toyota P/N 90987-02022, phải thay bằng mã nào?',
    'Thay rơ-le Toyota P/N 90987-02022 bằng rơ-le mới Toyota P/N 90987-02028.',
    [('FLASH',5478)],'relay_part_replacement',dict(old='90987-02022',new='90987-02028'),'Có các triệu chứng đã mô tả và mã rơ-le cũ khớp',categories=('identifier','condition'))
add('FLASH','Chú thích mở rộng phạm vi cho Toyota Matrix và Toyota Sienna nêu những đời xe nào, và việc mở rộng đó có điều kiện gì?',
    'Cả Matrix và Sienna đều được nêu đời 2003–2004; tài liệu nói có thể gặp trên các model có cấu hình hệ thống tương tự, không khẳng định mọi xe đều có lỗi.',
    [('FLASH',5479)],'relay_case_extended_applicability',dict(Matrix='2003–2004',Sienna='2003–2004'),'May also be seen in models with similar system configurations',categories=('footnote','condition','identifier'))
add('FAN','Tài liệu Water Temperature Switch hướng dẫn kiểm tra thông mạch ở nhiệt độ phòng giữa hai điểm nào? Kết quả hở mạch có ý nghĩa gì?',
    'Đo giữa vỏ và chân cực của công tắc nhiệt độ nước; nếu hở mạch thì công tắc đã hỏng và phải thay.',
    [('FAN',4889)],'water_temperature_switch_test','Continuity between housing and switch terminal; open circuit means failed and replace','Ở nhiệt độ phòng',categories=('procedure','condition'))
add('FAN','Khi lắp công tắc nhiệt độ nước A/C, tài liệu cho phép dùng dụng cụ gì và cấm những thao tác/vật liệu nào?',
    'Dùng dụng cụ cầm tay với đầu khẩu sâu; không siết quá lực, không dùng dụng cụ khí nén, không làm rơi/va đập công tắc và không dùng băng làm kín.',
    [('FAN',4896)],'water_temperature_switch_installation','Hand tools with deep socket; no over-torque, air tools, dropping/striking or sealing tape',categories=('procedure','condition'))
add('COBOPT','Bộ Camera Built-in Set của COBOTTA liệt kê những thành phần nào?',
    'AF Camera (N10-W02), End-effector EtherNet Cable và Camera Stay.',
    [('COBOPT',7822)],'camera_set_components',['AF Camera (N10-W02)','End-effector EtherNet Cable','Camera Stay'],categories=('figure_text','identifier'))
add('COBOPT','Tùy chọn hút chân không của COBOTTA có cần máy nén khí bên ngoài không?',
    'Không; tài liệu nói công cụ này cho phép hút lấy vật mà không cần cung cấp máy nén khí ngoài.',
    [('COBOPT',7839),('COBOPT',7840)],'vacuum_tool_external_air_compressor',False,categories=('condition','text'))
add('EVP2','EVP2 Guidance và EVP2 Runtime có vai trò và yêu cầu thiết bị khác nhau thế nào?',
    'Guidance chạy trên PC để thiết lập xử lý ảnh. Khi Runtime chạy, chỉ cần robot controller và camera kết nối để vận hành.',
    [('EVP2',7879),('EVP2',7880),('EVP2',7881)],'guidance_runtime_roles',dict(Guidance='PC setup',Runtime='controller and connected camera for operation'),categories=('comparison','multi_evidence'))
add('EVP2','EVP2 liệt kê năm chức năng tìm kiếm ảnh cơ bản nào?',
    'Shape search, Contrast search, Blob search, Color search và Auxiliary search.',
    [('EVP2',7888)],'search_functions',['Shape search','Contrast search','Blob search','Color search','Auxiliary search'])
add('WINCAPS','Để WINCAPS Plus tự sinh đường chạy tránh vật cản và có chu kỳ ngắn nhất, người dùng cần xác định những điểm nào?',
    'Xác định điểm bắt đầu và điểm kết thúc của chuyển động robot.',
    [('WINCAPS',7757)],'automatic_path_inputs',['start point','end point'],categories=('procedure','condition'))
add('WINCAPS','Home Position Guidance giả định gì về đường chạy tự động, và dùng RC8 cần thêm thiết bị gì theo chú thích?',
    'Giả định đường chạy tự động an toàn, không có vật cản; nếu dùng RC8 thì cần một IPC chuyên dụng riêng.',
    [('WINCAPS',7759),('WINCAPS',7760),('WINCAPS',7761)],'home_position_guidance_conditions',dict(path='safe and obstacle-free',RC8='separate dedicated IPC'),'Home Position Guidance; RC8 cho yêu cầu IPC',categories=('footnote','condition','multi_evidence'))
add('DUALARM','Dual-arm Control hỗ trợ những dòng robot nào và controller nào? Một controller điều khiển được bao nhiêu tay robot?',
    'HSR Series và HS-A1 Series, controller RC8A; một controller điều khiển hai tay robot.',
    [('DUALARM',7951),('DUALARM',7954)],'dual_arm_compatibility',dict(robots=['HSR','HS-A1'],controller='RC8A',arms_per_controller=2),categories=('comparison','multi_evidence'))
add('DUALARM','Những chức năng nào giúp hai tay robot tránh va chạm trong không gian hẹp, và chức năng nào phục vụ vật dài hoặc nặng?',
    'Exclusive control và virtual fence giúp tránh va chạm; coordinated control dùng cho vật dài hoặc nặng.',
    [('DUALARM',7974)],'dual_arm_functions',dict(collision_avoidance=['exclusive control','virtual fence'],long_heavy_objects='coordinated control'),categories=('condition','comparison'))
add('LPH','LPH-040A1 có tổng chiều dài hai tay và tải trọng tối đa bao nhiêu?',
    'Tổng chiều dài 400 mm (200 + 200 mm); tải trọng tối đa 3 kg.',
    [('LPH',8042)],'LPH_arm_length_and_payload',dict(arm_length_mm=400,payload_kg=3),categories=('table',))
add('LPH','Chu kỳ 0,45 giây của LPH được đo với tải, khoảng cách hai điểm và độ cao nào?',
    'Tải 2 kg, hai điểm cách nhau 300 mm, ở độ cao 25 mm.',
    [('LPH',8042),('LPH',8044)],'LPH_cycle_test_conditions',dict(cycle_s=.45,payload_kg=2,distance_mm=300,height_mm=25),categories=('table','footnote','condition'))
add('LPH','Tính năng command-slave của LPH cho phép PLC điều khiển robot qua gì, hỗ trợ bao nhiêu loại lệnh và có cần lập trình phía robot không?',
    'Qua Function Block (FB), hỗ trợ 130 loại lệnh robot; không cần tạo chương trình phía robot. Tính năng command-slave được nêu là tùy chọn miễn phí.',
    [('LPH',8027)],'LPH_PLC_command_slave',dict(interface='Function Block',command_types=130,robot_side_program_required=False,option='free'),categories=('condition','text'))
add('JACKET','Áo bảo vệ robot cho chế biến thực phẩm tương thích robot/controller nào và loại cấu hình nào bị loại trừ?',
    'Robot VS068/VS087; controller RC8A và RC8. Không dùng cho Communication Interface Flange-A, UL specifications hoặc Cleanroom type.',
    [('JACKET',8069),('JACKET',8070)],'protective_jacket_compatibility',dict(robots=['VS068','VS087'],controllers=['RC8A','RC8'],excluded=['Communication Interface Flange-A','UL specifications','Cleanroom type']),categories=('footnote','condition'))
add('JACKET','Khi lắp áo bảo vệ, tải trọng tối đa của VS068 và VS087 thay đổi thế nào so với cấu hình tiêu chuẩn?',
    'Cả VS068 và VS087 đều giảm từ 7 kg ở cấu hình tiêu chuẩn xuống 6 kg khi lắp áo.',
    [('JACKET',8085)],'jacket_payload_change',dict(VS068=dict(standard_kg=7,jacket_kg=6),VS087=dict(standard_kg=7,jacket_kg=6)),categories=('table','comparison','condition'))
add('JACKET','Phạm vi tuân thủ vật liệu tiếp xúc thực phẩm có ngoại lệ nào và có áp dụng cho tiêu chuẩn ngoài Nhật Bản không?',
    'Các vật liệu của phần có thể tiếp xúc thực phẩm tuân thủ tiêu chuẩn Nhật được nêu, ngoại trừ chỉ may; tài liệu nói không tuân thủ các tiêu chuẩn có hiệu lực ở các nước ngoài Nhật.',
    [('JACKET',8083)],'food_contact_material_scope','Tuân thủ tiêu chuẩn Nhật được nêu, trừ chỉ may; không áp dụng tuân thủ tiêu chuẩn ngoài Nhật',categories=('footnote','condition'))
add('VS050','VS-050S2 có tầm với, tải trọng tối đa và chu kỳ được công bố lần lượt bao nhiêu?',
    'Tầm với 520 mm, tải trọng tối đa 4 kg, chu kỳ 0,35 giây.',
    [('VS050',8121)],'VS050_reach_payload_cycle',dict(reach_mm=520,payload_kg=4,cycle_s=.35),categories=('table',))
add('VS050','VS-050S2 có cấp sạch và cấp bảo vệ nào ở cổ tay và thân máy?',
    'ISO Class 5; cổ tay IP67, thân máy (Unit) IP65.',
    [('VS050',8128)],'VS050_cleanliness_and_protection',dict(cleanliness='ISO Class 5',wrist='IP67',unit='IP65'),categories=('table','comparison'))
add('VS050','Khi chọn electric gripper connection flange specification-A, giới hạn góc J5 của VS-050S2 thay đổi thành bao nhiêu so với tiêu chuẩn?',
    'Tiêu chuẩn J5 là ±115°; khi chọn tùy chọn này, J5 là +110° đến −102°.',
    [('VS050',8161),('VS050',8165)],'VS050_J5_option_range',dict(standard='+115°/-115°',flange_A='+110°/-102°'),'Khi chọn electric gripper connection flange specification-A',categories=('table','footnote','condition'))
add('BRAKE','Má phanh DENSO First Time Fit được mô tả thế nào về vật liệu gốm và hàm lượng đồng, và bộ kit có kèm phần cứng không?',
    '100% ceramic, 100% copper-free; có kèm hardware kit.',
    [('BRAKE',8225)],'brake_pad_material_and_kit',dict(ceramic='100%',copper='copper-free',hardware_included=True))
add('BRAKE','Trong hình cấu tạo bộ má phanh, multilayer shim có chức năng gì và NVH là viết tắt của gì?',
    'Multilayer shim giúp giảm NVH; NVH là Noise, Vibration, and Harshness.',
    [('BRAKE',8193),('BRAKE',8212)],'brake_pad_shim_function','Mitigate NVH: Noise, Vibration, and Harshness',categories=('figure_text','footnote'))
add('WIPER','Trong bảng Conventional & Designer Wiper Blades, chiều dài 14 inch và 26 inch tương ứng bao nhiêu mm?',
    '14 inch tương ứng 355 mm; 26 inch tương ứng 650 mm.',
    [('WIPER',8258)],'wiper_length_conversion',dict(inch_14_mm=355,inch_26_mm=650),categories=('table','comparison'))
add('WIPER','Trong bảng Rear Wiper Blades, cỡ 12 inch/300 mm liệt kê những mã sản phẩm nào?',
    '160-5512, 160-5612, 160-5712 và 160-5912.',
    [('WIPER',8264),('WIPER',8265)],'rear_wiper_12inch_codes',['160-5512','160-5612','160-5712','160-5912'],categories=('table','identifier'))
add('LPH+VS050','So sánh LPH-040A1 và VS-050S2 về tầm với, tải trọng tối đa và chu kỳ. Hai giá trị chu kỳ có được đo ở cùng tải không?',
    'LPH: 400 mm, tải tối đa 3 kg, chu kỳ 0,45 s đo với tải 2 kg. VS-050S2: 520 mm, tải tối đa 4 kg, chu kỳ 0,35 s đo với tải 1 kg. Cả hai nêu khoảng cách 300 mm, độ cao 25 mm nhưng tải thử khác nhau, nên không coi đây là so sánh chu kỳ cùng tải.',
    [('LPH',8042),('LPH',8044),('VS050',8161),('VS050',8162)],'cross_robot_specs_and_test_load',dict(LPH=dict(reach_mm=400,max_payload_kg=3,cycle_s=.45,test_load_kg=2),VS050=dict(reach_mm=520,max_payload_kg=4,cycle_s=.35,test_load_kg=1)),'Hai điều kiện thử chu kỳ có tải khác nhau',categories=('multi_document','table','comparison','footnote','condition'))
add('BOX+DUALARM','Trong tài liệu Protective Box Light và Dual-arm Control, RC8A đóng vai trò gì? Tên robot tương thích với Dual-arm Control là gì?',
    'RC8A là controller: được liệt kê là controller được hỗ trợ bởi Protective Box Light và controller tương thích với Dual-arm Control. Robot của Dual-arm Control là HSR Series và HS-A1 Series; không được gán RC8A thành tên robot.',
    [('BOX',4545),('BOX',4546),('DUALARM',7951)],'controller_robot_role_separation',dict(RC8A='controller',dual_arm_robots=['HSR Series','HS-A1 Series']),categories=('multi_document','comparison','attribute_leakage'))
