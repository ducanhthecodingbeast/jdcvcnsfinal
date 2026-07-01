#!/usr/bin/env python3
"""
Create mocked gold standard by filling fields that were left empty in the original perfect extraction.
Goal: eliminate "unfair zeros" so we can fairly compare QA vs EMB on those fields too.

We use the raw text (from inputs/*.csv) + reasonable human inference to fill:
- CV: education_level, institution, major
- JD: required_education_level, required_major

For fields where raw text truly gives no info, we put a sensible default (e.g. "trung cấp nghề" for vocational drivers, or "" only if impossible).
"""

import pandas as pd
import re
from pathlib import Path

bmretr = Path(__file__).parent.parent
gold_cv = pd.read_csv(bmretr / 'gold' / 'cv.csv')
gold_jd = pd.read_csv(bmretr / 'gold' / 'jd.csv')

cv_raw = pd.read_csv(bmretr / 'raw' / 'cv_first10.csv')
jd_raw = pd.read_csv(bmretr / 'raw' / 'jd_first10.csv')

def get_cv_row(uid):
    return cv_raw[cv_raw['UserID'].astype(str) == str(uid)].iloc[0]

def get_jd_row(jid):
    return jd_raw[jd_raw['JobID'].astype(str) == str(jid)].iloc[0]

# ============================================
# CV MOCKS
# ============================================
# Strategy:
# - If Degree mentions "trung cấp nghề" or "trung tâm dạy nghề" → education_level = "trung cấp nghề"
# - Institution from "Đơn vị đào tạo: XXX"
# - Major: try to extract after "Chuyên ngành:", otherwise infer or put a reasonable vocational major for the job type.
#   For drivers with no major info, we put "nghề lái xe / vận tải" as a mocked value so the field is non-empty.

cv_mocks = {}

# 964496 - Driver, Degree: "Đơn vị đào tạo: Trường trung cấp nghề"
cv_mocks['964496'] = {
    'education_level': 'trung cấp nghề',
    'institution': 'Trường trung cấp nghề',
    'major': 'nghề lái xe / vận tải'
}

# 986206 - same
cv_mocks['986206'] = {
    'education_level': 'trung cấp nghề',
    'institution': 'Trường trung cấp nghề',
    'major': 'nghề lái xe / vận tải'
}

# 769632 - polluted Degree (job history text), no clean education info
# We mock conservatively: no clear diploma level visible in the polluted field.
cv_mocks['769632'] = {
    'education_level': 'không rõ (Degree bị nhiễu)',
    'institution': 'không rõ',
    'major': 'không rõ'
}

# 981314 - "Đơn vị đào tạo: Trung tâm dạy nghề"
cv_mocks['981314'] = {
    'education_level': 'trung cấp nghề',
    'institution': 'Trung tâm dạy nghề',
    'major': 'nghề kỹ thuật ô tô / bảo dưỡng xe'
}

# 962892 - already had major in gold, but make sure
cv_mocks['962892'] = {
    'education_level': 'cao đẳng',
    'institution': 'Cao Đẳng Kinh Tế Kỹ Thuật Vinatex TP.Hồ Chí Minh',
    'major': 'Công nghệ may'
}

# 967782
cv_mocks['967782'] = {
    'education_level': 'đại học',
    'institution': 'Đại học Thương Mại',
    'major': 'Kế Toán'
}

# 986307
cv_mocks['986307'] = {
    'education_level': 'trung cấp nghề',
    'institution': 'Trường trung cấp nghề',
    'major': 'nghề lái xe / vận tải'
}

# 990789
cv_mocks['990789'] = {
    'education_level': 'trung cấp nghề',
    'institution': 'Trung tâm dạy nghề',
    'major': 'nghề lái xe / vận tải'
}

# 981988 - "Trung học Đơn vị đào tạo: ĐH Sài Gòn"
cv_mocks['981988'] = {
    'education_level': 'trung học',
    'institution': 'ĐH Sài Gòn',
    'major': 'không rõ chuyên ngành (trung học)'
}

# 966890 - polluted Degree (job history)
cv_mocks['966890'] = {
    'education_level': 'không rõ (Degree bị nhiễu)',
    'institution': 'không rõ',
    'major': 'kinh nghiệm xây dựng 2 năm (tự học / thực tế)'
}

# Apply CV mocks
for uid, mocks in cv_mocks.items():
    mask = gold_cv['UserID'].astype(str) == uid
    for f, v in mocks.items():
        gold_cv.loc[mask, f] = v

# ============================================
# JD MOCKS
# ============================================

jd_mocks = {}

# JobID 0 - Sale Admin: "Tốt nghiệp đại học/cao đẳng"
jd_mocks['0'] = {
    'required_education_level': 'đại học/cao đẳng',
    'required_major': 'không ưu tiên chuyên ngành cụ thể'
}

# JobID 4 - Lễ Tân Gym: "Tốt nghiệp trung cấp trở lên các ngành quản trị kinh doanh, dịch vụ khách hàng, nhà hàng khách sạn..."
jd_mocks['4'] = {
    'required_education_level': 'trung cấp trở lên',
    'required_major': 'quản trị kinh doanh, dịch vụ khách hàng, nhà hàng khách sạn'
}

# JobID 5 - Graphic Design: requirements talk about tools (Figma, Adobe), no degree mentioned in the provided text.
# We mock conservatively: no strict degree requirement stated.
jd_mocks['5'] = {
    'required_education_level': 'không yêu cầu bằng cấp cụ thể (ưu tiên kỹ năng)',
    'required_major': 'thiết kế đồ họa / multimedia (ưu tiên)'
}

# JobID 7 - Marketing: "18-24 tuổi, hỗ trợ đào tạo sinh viên mới ra trường..."
jd_mocks['7'] = {
    'required_education_level': 'không yêu cầu (ưu tiên đam mê + kỹ năng, hỗ trợ đào tạo)',
    'required_major': 'marketing, truyền thông, quảng cáo (ưu tiên)'
}

# JobID 8 - iOS dev: English JD, focuses on 5+ years experience + skills. No degree mentioned.
jd_mocks['8'] = {
    'required_education_level': 'không đề cập bằng cấp (ưu tiên kinh nghiệm 5+ năm)',
    'required_major': 'computer science / software engineering (ưu tiên, không bắt buộc)'
}

# JobID 9 - Bán hàng thời trang: no degree requirement explicit.
jd_mocks['9'] = {
    'required_education_level': 'không yêu cầu bằng cấp',
    'required_major': 'không yêu cầu'
}

# Apply JD mocks
for jid, mocks in jd_mocks.items():
    mask = gold_jd['JobID'].astype(str) == jid
    for f, v in mocks.items():
        gold_jd.loc[mask, f] = v

# Save
gold_cv.to_csv(bmretr / 'gold_mocked' / 'cv.csv', index=False)
gold_jd.to_csv(bmretr / 'gold_mocked' / 'jd.csv', index=False)

print("Mocked gold saved to:")
print("  ", bmretr / 'gold_mocked' / 'cv.csv')
print("  ", bmretr / 'gold_mocked' / 'jd.csv')

print("\n=== Mocked Gold Preview (first 3 rows, key fields) ===")
print("\nCV mocked (education related):")
print(gold_cv[['UserID', 'education_level', 'institution', 'major']].head(3).to_string(index=False))

print("\nJD mocked (edu/major):")
print(gold_jd[['JobID', 'required_education_level', 'required_major']].to_string(index=False))
PYEOF
python bmretr/create_mocked_gold.py
