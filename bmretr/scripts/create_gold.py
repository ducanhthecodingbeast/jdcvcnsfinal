#!/usr/bin/env python3
"""
Create perfect (gold standard) extraction for first 10 CV and JD from phase1 markdown.
This is the human-curated "perfect" retrieval for benchmarking.
"""

import pandas as pd
import re
from pathlib import Path

def parse_md_table(path):
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if '|' in line and not re.match(r'^\s*\|[\s\-:]+\|', line):
            header_idx = i
            break

    headers = [h.strip() for h in lines[header_idx].split('|') if h.strip()]

    rows = []
    for line in lines[header_idx+2:]:
        if not line.strip() or re.match(r'^\s*\|[\s\-:]+\|', line):
            continue
        cells = [c.strip() for c in line.split('|')]
        cells = cells[1:-1] if len(cells) > 2 else cells
        if len(cells) >= len(headers) - 1:
            rows.append(cells[:len(headers)])

    df = pd.DataFrame(rows, columns=headers)
    return df

def create_gold_cv(cv10: pd.DataFrame) -> pd.DataFrame:
    """
    Perfect extraction for first 10 CVs.
    Based on careful reading of the raw Target, Skills, Degree, etc.
    """
    records = []

    # Row 0: UserID 964496 - Truck driver, 10 years exp
    records.append({
        'UserID': '964496',
        'summary': 'Với kinh nghiệm 10 năm lái xe tải; mong muốn vị trí nhân viên lái xe tải ổn định lâu dài; tuân thủ luật giao thông và đảm bảo an toàn hàng hóa',
        'education_level': 'trung cấp nghề',
        'institution': 'Trường trung cấp nghề',
        'major': '',
        'skills': 'Bằng lái xe hạng C, D, E; kinh nghiệm 5 năm lái xe tải, 5 năm lái xe khách; am hiểu xe tải, xử lý vấn đề xe; bảo quản xe sạch sẽ, bảo dưỡng thường xuyên; lái xe an toàn, tuân thủ luật giao thông; ghi nhớ nhanh, nhớ đường tốt; giao tiếp, thiết lập quan hệ khách hàng đồng nghiệp; giải quyết vấn đề; sắp xếp tổ chức quản lý công việc; chịu áp lực cao',
        'location': 'Hà Nội',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên lái xe bằng B2'
    })

    # Row 1: UserID 986206 - Driver D, long term commitment
    records.append({
        'UserID': '986206',
        'summary': 'Gắn bó lâu dài với công việc và doanh nghiệp; sát sao công việc, tích cực học hỏi; phát triển bản thân, đạt thành tựu; hòa hợp đồng nghiệp; bổ sung kỹ năng; nâng cao kỹ năng lái xe, tuân thủ luật giao thông; có kế hoạch thăng tiến, thăng lương; đảm bảo chất lượng, hạn chế sai sót; tích cực tăng ca, chuyên cần',
        'education_level': 'trung cấp nghề',
        'institution': 'Trường trung cấp nghề',
        'major': '',
        'skills': 'Có chí tiến thủ; khả năng đi xa, đi dài ngày; sức khỏe đảm bảo cho vận tải; chăm chỉ, nhiệt tình, không ngại khó khăn; tinh thần trách nhiệm cao; chấp hành quy định công ty; thông thạo luật giao thông, thạo đường xá; giải quyết vấn đề; có thể tăng ca ngoài giờ; kiểm tra đánh giá tuyến vận chuyển, rút kinh nghiệm',
        'location': 'Bình Dương',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên lái xe bằng D'
    })

    # Row 2: UserID 769632 - Admin/office, Chinese, but Degree field is polluted with job history
    records.append({
        'UserID': '769632',
        'summary': 'Tìm kiếm vị trí nhân viên hành chính môi trường doanh nghiệp; hoàn thiện tin học văn phòng, lập báo cáo biên bản sắp xếp hoạt động; phát huy năng lực hỗ trợ làm việc hiệu quả; học hỏi kinh nghiệm từ đồng nghiệp trưởng phòng; phấn đấu trưởng phòng hành chính nhân sự trong 4 năm tới',
        'education_level': '',
        'institution': '',
        'major': '',
        'skills': 'Am hiểu quy trình hành chính nhân sự và quy tắc quản lý giấy tờ hồ sơ doanh nghiệp; quản lý thời gian, sắp xếp lập kế hoạch công việc hiệu quả; giao tiếp ứng xử tốt trong công việc hành chính; kỹ năng làm việc nhóm, sẵn sàng hỗ trợ người khác',
        'location': 'Bắc Giang',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên văn phòng, khu công nghiệp sử dụng tiếng Trung'
    })

    # Row 3: UserID 981314 - Vehicle repair engineer / logistics support
    records.append({
        'UserID': '981314',
        'summary': 'Trở thành kỹ sư tốt trong sửa chữa xe vận tải; người hậu cần xuất sắc hỗ trợ chuyên chở hàng hóa; đảm bảo thực hiện công việc tốt do công ty giao; làm việc tinh thần trách nhiệm cao, chịu trách nhiệm sai sót; thăng chức quản lý đội ngũ bảo dưỡng trong 3-5 năm tới; xây dựng mối quan hệ tốt với đồng nghiệp cấp cao trên tinh thần đồng hành hợp tác phát triển',
        'education_level': 'trung cấp nghề',
        'institution': 'Trung tâm dạy nghề',
        'major': '',
        'skills': 'Có khả năng chu toàn công việc, siêng năng, tỉ mỉ; kỹ năng bảo dưỡng sửa chữa xe tốt; am hiểu kỹ thuật khó mang tính chuyên môn cao; có khả năng quản lý công việc, từng nằm trong đội ngũ chuyên viên kỹ thuật xe tải; có chứng chỉ đảm bảo kỹ thuật sửa chữa bảo trì phương tiện ô tô xe máy; thân thiện hòa đồng cởi mở; khả năng công nghệ thông tin tin học văn phòng trung bình khá; tốt nghiệp nghiệp vụ kỹ thuật ôtô',
        'location': 'Hà Nội',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên lái xe hạng C'
    })

    # Row 4: UserID 962892 - Purchasing staff, textile background
    records.append({
        'UserID': '962892',
        'summary': 'Có công việc nhiều cơ hội thăng tiến, môi trường chuyên nghiệp thân thiện; công việc phù hợp bản thân; có cơ hội thăng tiến tốt; phát huy khả năng vốn có; chế độ công ty đảm bảo',
        'education_level': 'cao đẳng',
        'institution': 'Cao Đẳng Kinh Tế Kỹ Thuật Vinatex TP.Hồ Chí Minh',
        'major': 'Công nghệ may',
        'skills': 'Trung thực, cẩn thận, có óc sáng tạo; kỹ năng giao tiếp linh hoạt, khả năng thuyết trình trình bày; dễ dàng hòa nhập môi trường làm việc mới; kỹ năng tổ chức và quản lý công việc hiệu quả',
        'location': 'Hồ Chí Minh',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên thu mua'
    })

    # Row 5: UserID 967782 - HR Admin, accounting background
    records.append({
        'UserID': '967782',
        'summary': 'Làm việc môi trường có yếu tố cạnh tranh tích cực; công việc phù hợp và ổn định; mong muốn chỗ làm có cơ hội thăng tiến tốt; rèn luyện kỹ năng bản thân tốt hơn; công ty có chế độ đãi ngộ tốt',
        'education_level': 'đại học',
        'institution': 'Đại học Thương Mại',
        'major': 'Kế Toán',
        'skills': 'Trung thực, cẩn thận, có óc sáng tạo; kỹ năng giao tiếp linh hoạt, khả năng thuyết trình trình bày; có khả năng thích nghi nhanh với môi trường làm việc; quản lý thời gian hiệu quả; có kỹ năng tin học văn phòng',
        'location': 'Bắc Giang',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên Hành Chính Nhân Sự'
    })

    # Row 6: UserID 986307 - Professional driver C, memory for routes
    records.append({
        'UserID': '986307',
        'summary': 'Mục tiêu ngắn hạn: làm việc công ty lâu dài, học hỏi kiến thức từ đồng nghiệp để phát triển; trở thành lái xe chuyên nghiệp có hiểu biết rộng, có khả năng giúp đỡ đồng nghiệp khi cần; mục tiêu dài hạn: thăng tiến vị trí cao hơn trong công ty',
        'education_level': 'trung cấp nghề',
        'institution': 'Trường trung cấp nghề',
        'major': '',
        'skills': 'Có thể ghi nhớ tên cùng vị trí đường ngay cả khi mới đi qua 1 lần; tinh thần trách nhiệm cao, biết ưu tiên đầu việc quan trọng hoàn thành kịp tiến độ; có khả năng thích nghi nhanh với môi trường làm việc; có thể làm việc theo ca, làm việc độc lập hoặc nhóm; có kỹ năng lái xe an toàn, tuân thủ luật giao thông',
        'location': 'Bình Dương',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên lái xe bằng C'
    })

    # Row 7: UserID 990789 - Driver C with 3 years exp, wants professional environment
    records.append({
        'UserID': '990789',
        'summary': 'Với 3 năm kinh nghiệm trong ngành vận tải lái xe; mong đóng góp ý tưởng giúp mở rộng đa dạng loại hình vận tải cho công ty, phục vụ nhiều nhu cầu khách hàng; muốn làm việc môi trường chuyên nghiệp như công ty ABC; mong muốn công việc ổn định gắn bó lâu dài',
        'education_level': 'trung cấp nghề',
        'institution': 'Trung tâm dạy nghề',
        'major': '',
        'skills': 'Có bằng lái xe B2; nắm rõ quy định quy trình vận tải hành khách và hàng hóa, hiểu luật giao thông; đảm bảo lái xe an toàn trên mọi hành trình; xử lý tình huống nhanh linh hoạt; có sức khỏe tốt, chịu được áp lực công việc; có kinh nghiệm lái xe khách và xe tải; có kỹ năng giao tiếp tốt với khách hàng',
        'location': 'Bình Dương',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên lái xe hạng C'
    })

    # Row 8: UserID 981988 - Service / interpreter, wants stability + promotion + good salary
    records.append({
        'UserID': '981988',
        'summary': 'Trau dồi thêm kinh nghiệm và kỹ năng; mong muốn chỗ làm ổn định lâu dài; có cơ hội thăng tiến tốt; có mức lương tốt; có cơ hội cống hiến bản thân tốt; làm việc trong môi trường chuyên nghiệp',
        'education_level': 'trung học',
        'institution': 'ĐH Sài Gòn',
        'major': '',
        'skills': 'Ngoại ngữ, làm việc nhóm, thái độ làm việc; biết lắng nghe tiếp thu ý kiến đồng nghiệp và cấp trên; chăm chỉ chịu khó tìm hiểu vấn đề trong công việc và cuộc sống; tinh thần trách nhiệm trong công việc; có khả năng giao tiếp tốt',
        'location': 'Hồ Chí Minh, Hồ Chí Minh, Hà Nội',
        'salary_expectation': '5,000,000 - 10,000,000',
        'desired_job': 'Phục vụ, Phiên dịch'
    })

    # Row 9: UserID 966890 - Sales/construction background, 2 years exp
    records.append({
        'UserID': '966890',
        'summary': 'Được làm việc môi trường có yếu tố cạnh tranh tích cực; công việc ổn định để gắn bó lâu dài; phấn đấu trở thành nhân viên có vị trí cao hơn; rèn luyện kỹ năng bản thân tốt hơn; chế độ công ty đảm bảo',
        'education_level': '',
        'institution': '',
        'major': '',
        'skills': 'Có niềm say mê trong công việc; giao tiếp ứng xử với mọi người tốt; có khả năng thích nghi nhanh với môi trường làm việc; kỹ năng lên kế hoạch tổ chức sắp xếp công việc; khả năng chịu áp lực; có 2 năm kinh nghiệm xây dựng: khai thác tìm kiếm thông tin tiếp cận khách hàng; cập nhật nguồn thông tin dự án khách hàng tiềm năng; đánh giá phân tích lựa chọn dự án khả thi phù hợp chiến lược năng lực công ty; tiếp cận dự án khách hàng; thiết lập duy trì phát triển quan hệ khách hàng; chăm sóc duy trì quan hệ khách hàng hiện có; phát triển quan hệ khách hàng mới tiềm năng; đề xuất tham mưu; nắm chắc thông tin dự án khách hàng, đề ra kế hoạch triển khai; xây dựng triển khai kế hoạch đấu thầu; nghiên cứu hồ sơ phân tích yêu cầu; phối hợp chặt chẽ các phòng ban; thương thảo đàm phán ký kết hợp đồng',
        'location': 'Hồ Chí Minh',
        'salary_expectation': 'Thỏa thuận',
        'desired_job': 'Nhân viên Kinh Doanh'
    })

    return pd.DataFrame(records)

def create_gold_jd(jd10: pd.DataFrame) -> pd.DataFrame:
    """
    Perfect extraction for first 10 JDs.
    """
    records = []

    # JobID 0: Sale Admin Website
    records.append({
        'JobID': '0',
        'job_description': 'Sale Admin Website',
        'responsibilities': 'Thường xuyên cập nhật thay đổi chính sách bán hàng chương trình khuyến mại; làm việc với đối tác chiến lược mang lại chương trình collab hiệu quả; kiểm tra đơn hàng từ website TMĐT, xác thực đơn hàng chịu trách nhiệm điều phối; phối hợp phòng ban đảm bảo quy trình xử lý đơn hàng giao hàng; phối hợp Kho vận Sales nhận phản hồi khách hàng tư vấn giải đáp thắc mắc; phối hợp IT tối ưu website bán hàng; thực hiện công việc khác khi có yêu cầu từ cấp trên',
        'requirements': 'Tốt nghiệp đại học/cao đẳng; có kinh nghiệm 1-2 năm vị trí tương tự; ưu tiên kinh nghiệm SEO Web; ưu tiên kinh nghiệm công ty TMĐT Bán lẻ; kỹ năng tin học văn phòng tốt đặc biệt thành thạo Excel; kỹ năng giao tiếp tốt có tinh thần phối hợp tư vấn hỗ trợ; nhanh nhẹn trung thực cẩn thận',
        'required_education_level': 'đại học/cao đẳng',
        'required_major': '',
        'required_skills': 'tin học văn phòng tốt, thành thạo Excel; giao tiếp tốt; tinh thần phối hợp tư vấn hỗ trợ; nhanh nhẹn trung thực cẩn thận; kinh nghiệm 1-2 năm vị trí tương tự; ưu tiên SEO Web; ưu tiên TMĐT Bán lẻ',
        'salary_offer': '5,000,000 - 10,000,000',
        'job_location': 'Hà Nội'
    })

    # JobID 1: Thực Tập Sinh Lập Trình No-Code Low-Code
    records.append({
        'JobID': '1',
        'job_description': 'Thực Tập Sinh Lập Trình (No-Code, Low-Code Platform)',
        'responsibilities': 'Phối hợp bộ phận lập trình No-code Low-code phát triển ứng dụng sử dụng nền tảng; thiết lập xác định cấu hình hệ thống cho yêu cầu riêng từng khách hàng; tạo trang web bằng Webflow, Shopify, Burble, Adalo; xây dựng trang website thương mại điện tử EC, phát triển hệ thống web, phát triển ứng dụng điện thoại thông minh; tìm hiểu đăng tải thông tin về bốn nền tảng Nocode và Lowcode',
        'requirements': 'Sinh viên năm 3, năm 4 hoặc sinh viên cao học năm 1, năm 2 chuyên ngành công nghệ thông tin; thời gian làm việc thứ 2 đến thứ 6, thực tập tối thiểu 4 ngày/tuần; quan tâm thị trường No-code Low-code; thích làm việc hiệu quả, chủ động học hỏi trao dồi kiến thức; có laptop cá nhân',
        'required_education_level': 'đang học đại học/cao học năm 3-4 chuyên ngành công nghệ thông tin',
        'required_major': 'công nghệ thông tin',
        'required_skills': 'No-code, Low-code (Webflow, Shopify, Burble, Adalo); phát triển ứng dụng web và mobile; chủ động học hỏi; có laptop cá nhân; có thể thực tập tối thiểu 4 ngày/tuần',
        'salary_offer': '1,000,000 - 5,000,000',
        'job_location': 'Hồ Chí Minh'
    })

    # JobID 2: HR Business Partner
    records.append({
        'JobID': '2',
        'job_description': 'HR Business Partner',
        'responsibilities': 'Xây dựng thực hiện chiến lược nhân sự phù hợp chiến lược kinh doanh tổng thể từng giai đoạn phát triển; giám sát quản lý hệ thống đánh giá thưởng phạt nhân viên; hỗ trợ nhu cầu kinh doanh qua tham gia phát triển giữ chân lao động; xây dựng quản lý kế hoạch lương thưởng chế độ đãi ngộ phúc lợi; quản lý xây dựng văn hóa doanh nghiệp; báo cáo Ban Lãnh Đạo hỗ trợ ra quyết định qua số liệu nhân sự; đảm bảo tuân thủ pháp luật quản trị nguồn nhân lực; thiết kế xây dựng hệ thống KPI đánh giá hiệu quả công việc, hướng dẫn đánh giá định kỳ, xây dựng chính sách khen thưởng theo tháng quý năm',
        'requirements': 'Tốt nghiệp đại học chuyên ngành kinh tế, quản trị, hành chánh, luật; am hiểu sâu Luật Lao động, Luật doanh nghiệp, Luật BHXH, Luật Công đoàn; ít nhất 3 năm kinh nghiệm quản trị nhân sự hành chánh, ít nhất 1 năm vị trí tương đương; kỹ năng lãnh đạo nhân viên, lập kế hoạch, giao tiếp tốt; kỹ năng tổ chức giám sát công việc; kỹ năng phân tích tổng hợp làm báo cáo; chịu áp lực cao; trung thực, chính trực, nhiệt tình, sáng tạo',
        'required_education_level': 'đại học',
        'required_major': 'kinh tế, quản trị, hành chánh, luật',
        'required_skills': 'Luật Lao động, Luật doanh nghiệp, Luật BHXH, Luật Công đoàn; quản trị nhân sự hành chánh 3+ năm; lãnh đạo, lập kế hoạch, giao tiếp; tổ chức giám sát; phân tích tổng hợp báo cáo; chịu áp lực; trung thực chính trực nhiệt tình sáng tạo',
        'salary_offer': 'Thỏa thuận',
        'job_location': 'Hồ Chí Minh'
    })

    # JobID 3: General Manager - hotel
    records.append({
        'JobID': '3',
        'job_description': 'General Manager',
        'responsibilities': 'Lập triển khai kế hoạch kinh doanh: phối hợp bộ phận đặt chỉ tiêu định hướng lập kế hoạch; xây dựng mối quan hệ khách hàng trọng điểm; tham mưu giúp việc Ban Giám đốc mọi hoạt động quản lý điều hành; chịu trách nhiệm mọi mặt điều hành bao gồm sự hài lòng dịch vụ, kết quả tài chính, nguồn doanh thu lợi nhuận; duy trì đảm bảo hoạt động các bộ phận vận hành tốt; kiểm tra chất lượng phòng vệ sinh sảnh dịch vụ; giám sát điều chỉnh thái độ chất lượng phục vụ; kiểm soát bảo trì bảo dưỡng nâng cấp thiết bị tài sản; tối ưu hóa lợi nhuận, phát triển phương pháp tăng lợi nhuận tìm kiếm cơ hội mới; xây dựng quy trình hoạt động tiêu chuẩn, mô tả công việc quy trình chuẩn cho từng vị trí, triển khai áp dụng; thay đổi cải tiến quy trình phù hợp định hướng mới; chỉ đạo giải quyết sự cố vấn đề phát sinh; quản lý trực tiếp gián tiếp toàn bộ hệ thống nhân sự; đảm bảo đội ngũ đáp ứng yêu cầu; tham gia phỏng vấn tuyển dụng đàm phán chế độ đãi ngộ; đánh giá đề xuất khen thưởng kỷ luật; đại diện khách sạn làm việc đối tác cơ quan chức năng truyền thông',
        'requirements': 'Tốt nghiệp đại học trở lên quản trị du lịch, quản trị kinh doanh hoặc chứng chỉ tương đương; ít nhất 3-5 năm kinh nghiệm vị trí tương đương khách sạn 3,4,5 sao; quan hệ đối tác tốt sâu rộng để phát triển; sử dụng 4 kỹ năng tiếng Anh thành thạo; kỹ năng lãnh đạo truyền cảm hứng, quản trị; kỹ năng phân tích tổng hợp lập kế hoạch; tầm nhìn chiến lược; thành thạo tin học văn phòng các phần mềm quản lý',
        'required_education_level': 'đại học trở lên',
        'required_major': 'quản trị du lịch, quản trị kinh doanh hoặc tương đương',
        'required_skills': '3-5 năm kinh nghiệm General Manager hoặc tương đương khách sạn 3-5 sao; tiếng Anh 4 kỹ năng thành thạo; lãnh đạo truyền cảm hứng quản trị; phân tích tổng hợp lập kế hoạch; tầm nhìn chiến lược; tin học văn phòng phần mềm quản lý; quan hệ đối tác tốt',
        'salary_offer': 'Thỏa thuận',
        'job_location': 'Đà Nẵng'
    })

    # JobID 4: Lễ Tân Gym
    records.append({
        'JobID': '4',
        'job_description': 'Lễ Tân Gym Quận 12',
        'responsibilities': 'Trực quầy Lễ tân; tư vấn bán hàng gói membership cho khách hàng; gọi điện chúc sinh nhật hỏi thăm hội viên không đi tập, thông báo chương trình khuyến mãi; tiếp nhận xử lí phản hồi khách hàng về dịch vụ; trực trả lời fanpage, inbox, comment; check in check out hội viên; vệ sinh khu vực quầy lễ tân; các công việc khác theo sự phân công của quản lý',
        'requirements': 'Giới tính ưu tiên Nữ 22-35 tuổi; ưu tiên kinh nghiệm Lễ tân, Chăm sóc khách hàng, Telesale; không có kinh nghiệm sẽ được đào tạo; tính cách vui vẻ, năng lượng, nhiệt tình, lịch sự; trình độ tốt nghiệp THPT trở lên; có khả năng giao tiếp tốt; có kỹ năng sử dụng máy tính, phần mềm văn phòng cơ bản; có thể làm việc theo ca, làm việc cuối tuần ngày lễ',
        'required_education_level': 'tốt nghiệp THPT trở lên',
        'required_major': '',
        'required_skills': 'giao tiếp tốt; sử dụng máy tính phần mềm văn phòng cơ bản; làm việc theo ca cuối tuần ngày lễ; vui vẻ năng lượng nhiệt tình lịch sự; ưu tiên kinh nghiệm Lễ tân Chăm sóc khách hàng Telesale',
        'salary_offer': '5,000,000 - 10,000,000',
        'job_location': 'Hồ Chí Minh'
    })

    # JobID 5: Graphic Design
    records.append({
        'JobID': '5',
        'job_description': 'Nhân Viên Thiết Kế Đồ Họa Graphic Design',
        'responsibilities': 'Thu thập đánh giá yêu cầu người dùng với sản phẩm nội dung phụ trách; minh họa ý tưởng thiết kế bằng bảng phân cảnh quy trình xử lý sơ đồ trang web; thiết kế phần tử giao diện người dùng đồ họa như menu, tab, widget, v.v.; phát triển giao diện người dùng đồ họa thông qua các công cụ thiết kế; xây dựng các trang đích, biểu ngữ, biểu tượng, đồ họa thông tin và các yếu tố đồ họa khác; chuẩn bị bản nháp sơ bộ và trình bày ý tưởng; phát triển đồ họa cho các chiến dịch tiếp thị sản phẩm; thiết kế logo, biểu tượng, bảng hiệu và các yếu tố nhận diện thương hiệu khác; cộng tác với các nhóm tiếp thị và sản phẩm để đảm bảo thiết kế phù hợp với thông điệp và chiến lược thương hiệu',
        'requirements': 'Kiến thức về công cụ khung dây (Figma và InDesign); cập nhật kiến thức phần mềm thiết kế Adobe Illustrator và Photoshop; tinh thần đồng đội; kỹ năng giao tiếp mạnh mẽ để hợp tác với các bên liên quan; kỹ năng quản lý thời gian để đáp ứng thời hạn; có khả năng làm việc độc lập và theo nhóm; có khả năng làm việc trong môi trường năng động, thay đổi nhanh chóng; có khả năng học hỏi và phát triển kỹ năng mới; có khả năng làm việc dưới áp lực',
        'required_education_level': '',
        'required_major': '',
        'required_skills': 'Figma, InDesign; Adobe Illustrator, Photoshop; thiết kế giao diện người dùng, đồ họa, banner, logo, nhận diện thương hiệu; tinh thần đồng đội; giao tiếp mạnh mẽ; quản lý thời gian; làm việc độc lập và nhóm; môi trường năng động thay đổi nhanh; học hỏi phát triển kỹ năng mới; chịu áp lực',
        'salary_offer': 'Thỏa thuận',
        'job_location': 'Hà Nội'
    })

    # JobID 6: Tư Vấn Online
    records.append({
        'JobID': '6',
        'job_description': 'Nhân Viên Tư Vấn Online',
        'responsibilities': 'Tìm hiểu nghiên cứu sản phẩm công ty để giới thiệu tư vấn giải đáp thắc mắc khách hàng; đàm phán thương lượng với khách hàng về giá cả hợp đồng, chốt đơn hỗ trợ khách hàng ký hợp đồng; chăm sóc khách hàng sau khi bán hàng; thực hiện các công việc khác theo yêu cầu của cấp trên',
        'requirements': 'Chấp nhận ứng viên mới tốt nghiệp hoặc đã tốt nghiệp cao đẳng/đại học liên quan khối ngành kinh tế, muốn tìm kiếm cơ hội trau dồi kinh nghiệm tư vấn bán hàng; có khả năng giao tiếp rất tốt; có tính tự giác chủ động để hoàn thành công việc được giao; có khả năng làm việc nhóm; có khả năng làm việc dưới áp lực; có khả năng học hỏi và phát triển kỹ năng mới',
        'required_education_level': 'cao đẳng/đại học',
        'required_major': 'kinh tế hoặc liên quan',
        'required_skills': 'giao tiếp rất tốt; tư vấn bán hàng; tự giác chủ động; làm việc nhóm; chịu áp lực; học hỏi phát triển kỹ năng mới; ưu tiên mới tốt nghiệp hoặc đã tốt nghiệp cao đẳng/đại học kinh tế',
        'salary_offer': '5,000,000 - 10,000,000',
        'job_location': 'Hồ Chí Minh'
    })

    # JobID 7: Marketing Mỹ Đình
    records.append({
        'JobID': '7',
        'job_description': 'Nhân Viên Marketing Tại Mỹ Đình - Hà Nội',
        'responsibilities': 'Tìm kiếm xử lý hình ảnh video sản phẩm; lên chiến dịch quảng cáo Facebook, Tik Tok, Youtube; phân tích tìm kiếm sản phẩm phù hợp với khách hàng; theo dõi đánh giá các chiến dịch hàng giờ hàng ngày để tối ưu chi phí tỉ lệ chuyển đổi; thực hiện các công việc khác theo yêu cầu của cấp trên',
        'requirements': 'Nam/Nữ tuổi từ 18-24 tuổi, ưu tiên có kinh nghiệm chạy quảng cáo Facebook Ads, Tik Tok Ads; hỗ trợ đào tạo cho sinh viên mới ra trường hoặc chuẩn bị ra trường, có đam mê marketing truyền thông quảng cáo; có khả năng làm việc nhóm; có khả năng làm việc dưới áp lực; có khả năng học hỏi và phát triển kỹ năng mới; có laptop cá nhân',
        'required_education_level': '',
        'required_major': '',
        'required_skills': 'Facebook Ads, Tik Tok Ads, Youtube Ads; xử lý hình ảnh video; phân tích đánh giá chiến dịch; tối ưu chi phí chuyển đổi; làm việc nhóm; chịu áp lực; học hỏi kỹ năng mới; có laptop; đam mê marketing truyền thông quảng cáo; ưu tiên 18-24 tuổi có kinh nghiệm ads',
        'salary_offer': '10,000,000 - 15,000,000',
        'job_location': 'Hà Nội'
    })

    # JobID 8: Lập Trình Viên iOS
    records.append({
        'JobID': '8',
        'job_description': 'Lập Trình Viên IOS (Từ 5 Năm Kinh Nghiệm Trở Lên, Tiếng Anh Giao Tiếp) - Đối Tác Us',
        'responsibilities': 'Phát triển ứng dụng iOS sử dụng Objective-C hoặc Swift, React Native, Cocoa Touch; làm việc với iOS frameworks như Core Data, Core Animation; offline storage, threading, performance tuning; làm việc với RESTful APIs để kết nối ứng dụng iOS với backend; làm việc với các dịch vụ bên thứ ba như Google Maps, Firebase; thiết kế kiến trúc ứng dụng, giao tiếp với các thành viên trong nhóm; tham gia review code, đảm bảo chất lượng code; tham gia họp với đối tác US, giao tiếp tiếng Anh',
        'requirements': 'Ít nhất 5+ năm kinh nghiệm với Objective-C hoặc Swift, React Native, Cocoa Touch; kinh nghiệm với iOS frameworks Core Data, Core Animation; kinh nghiệm offline storage, threading, performance tuning; quen thuộc RESTful APIs; kinh nghiệm Google Maps, Firebase là lợi thế; khả năng giao tiếp tiếng Anh tốt; khả năng làm việc nhóm; khả năng học hỏi công nghệ mới; có khả năng làm việc độc lập; có khả năng làm việc dưới áp lực',
        'required_education_level': '',
        'required_major': '',
        'required_skills': 'Objective-C hoặc Swift 5+ năm; React Native; Cocoa Touch; Core Data, Core Animation; offline storage, threading, performance tuning; RESTful APIs; Google Maps, Firebase (lợi thế); tiếng Anh giao tiếp tốt; làm việc nhóm; học hỏi công nghệ mới; làm việc độc lập; chịu áp lực',
        'salary_offer': 'Thỏa thuận',
        'job_location': 'Hồ Chí Minh'
    })

    # JobID 9: Bán Hàng Thời Trang Nữ Yên Bái
    records.append({
        'JobID': '9',
        'job_description': 'Nhân Viên Bán Hàng Thời Trang Nữ Tại Yên Bái',
        'responsibilities': 'Nhân viên tư vấn bán hàng thời trang nữ công sở, hàng thiết kế của công ty; trưng bày các mặt hàng, đón khách, tư vấn khách; nhập thông tin khách hàng, sản phẩm vào phần mềm để in hóa đơn cho khách; trao đổi chi tiết khi phỏng vấn',
        'requirements': 'Ưu tiên các bạn nữ yêu thích kinh doanh; nhanh nhẹn, thật thà, giao tiếp tốt; có kinh nghiệm bán hàng quần áo công sở, bán hàng online; sử dụng được máy tính để nhập thông tin hóa đơn bán hàng trên phần mềm; có thể làm việc cuối tuần ngày lễ',
        'required_education_level': '',
        'required_major': '',
        'required_skills': 'bán hàng thời trang nữ công sở; tư vấn khách hàng; trưng bày hàng hóa; sử dụng máy tính phần mềm bán hàng in hóa đơn; nhanh nhẹn thật thà giao tiếp tốt; kinh nghiệm bán hàng quần áo công sở hoặc online (ưu tiên); làm việc cuối tuần ngày lễ; nữ yêu thích kinh doanh',
        'salary_offer': '5,000,000 - 10,000,000',
        'job_location': 'Yên Bái'
    })

    return pd.DataFrame(records)

def main():
    cv_path = '/home/aiface/jdcvcnsfinal/phase1/USER_DATA_FINAL.md'
    jd_path = '/home/aiface/jdcvcnsfinal/phase1/JOB_DATA_FINAL.md'

    cv_df = parse_md_table(cv_path)
    jd_df = parse_md_table(jd_path)

    cv10 = cv_df.head(10)
    jd10 = jd_df.head(10)

    gold_cv = create_gold_cv(cv10)
    gold_jd = create_gold_jd(jd10)

    outdir = Path(__file__).parent.parent / 'gold'
    outdir.mkdir(parents=True, exist_ok=True)

    gold_cv.to_csv(outdir / 'cv.csv', index=False)
    gold_jd.to_csv(outdir / 'jd.csv', index=False)

    print("Saved gold standard to:")
    print(f"  {outdir / 'cv.csv'}")
    print(f"  {outdir / 'jd.csv'}")
    print(f"\nCV gold columns: {list(gold_cv.columns)}")
    print(f"JD gold columns: {list(gold_jd.columns)}")

if __name__ == "__main__":
    main()