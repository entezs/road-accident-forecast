# severity_map

ข้อมูลของไอเดีย "ถ้าเกิดอุบัติเหตุตรงนี้ จะร้ายแรงไหม"
ทุกไฟล์ในโฟลเดอร์นี้สร้างจาก `notebooks/explore_idea/severity_map.ipynb`

| โฟลเดอร์ | ไฟล์ | คืออะไร |
|---|---|---|
| (raw) | `data/datagov/accident2020.csv` ถึง `accident2026.csv` | ข้อมูลดิบ ไม่ได้คัดลอกมาไว้ที่นี่ เพราะไฟล์รวมกันใหญ่ประมาณ 85 MB |
| `clean/` | `accidents_clean.csv` | ข้อมูลที่คลีนแล้ว หนึ่งแถวคือหนึ่งเหตุ |
| `features/` | `accidents_features.csv` | ตารางที่พร้อมใช้ฝึกโมเดล มีคอลัมน์ `split` บอกว่าเป็น train, validation หรือ test |
| `model/` | `severity_model.joblib` | โมเดลที่ฝึกด้วยข้อมูลทั้งหมด ใช้ในเว็บ `app/severity_map_app.py` |
