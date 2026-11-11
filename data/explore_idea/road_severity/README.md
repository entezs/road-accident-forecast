# road_severity

ข้อมูลของไอเดีย "ถ้าเกิดอุบัติเหตุบนถนนเส้นนี้ ที่หลัก กม. นี้ จะรุนแรงกี่ %"

ไฟล์ในโฟลเดอร์นี้สร้างจาก `notebooks/explore_idea/road_severity.ipynb` ถ้าไฟล์หาย ให้รัน notebook จากบนลงล่าง (ประมาณ 1 นาที)

| โฟลเดอร์ | ไฟล์ | คืออะไร |
|---|---|---|
| `clean/` | `road_accidents_clean.csv` | หนึ่งแถวคืออุบัติเหตุ 1 ครั้ง (148,990 แถว) มีเส้นถนน หลัก กม. ช่วงถนน 5 กม. ประเภทรถ เวลา อากาศ และคำตอบ `severe` |
| `features/` | `road_features.csv` | X ทั้ง 20 คอลัมน์ (0/1 และประวัติช่วงถนน) กับ `year`, `split` และ `severe` |
| `results/` | `test_scores.csv` | confusion matrix และคะแนนบนชุด test (2025-2026) ของ Random Forest เทียบกับ baseline |
| `results/` | `factor_effects.csv` | แต่ละปัจจัยเปลี่ยน % รุนแรงที่โมเดลทายไปกี่จุด |

**ข้อมูลดิบ** ไม่ได้คัดลอกมาไว้ที่นี่ notebook อ่านจาก `data/datagov/` โดยตรง (ปี 2020–2026)
