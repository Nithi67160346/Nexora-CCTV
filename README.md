# NEXORA CCTV

ชุดสำหรับนำเว็บ NEXORA ไปติดตั้งบน server: ตัวเล่นหลักใช้ YOLO Pose ร่วมกับ Fall RandomForest 54 features และโมเดลภาพ ResNet18 + LSTM สำหรับการทำร้ายร่างกาย รวม Location, โซนเตียง, Wandering และใบหน้า

รุ่น `v0.1.0-rc.1` · เว็บ `qa-fall-lstm-main-20261007-fall54fix1` · สถานะทดลอง

## ติดตั้ง

1. ติดตั้ง Docker Engine / Docker Desktop ที่ใช้ Linux containers และ Docker Compose v2
2. ดาวน์โหลด **NEXORA-Server-v0.1.0-rc.1.zip** และไฟล์ `.sha256` จาก [Release](https://github.com/Nithi67160346/Nexora-CCTV/releases/tag/v0.1.0-rc.1) แล้วตรวจ checksum ก่อนแตกไฟล์ ZIP นี้มีโค้ดและโมเดลพร้อมตำแหน่งที่ต้องใช้ แต่ยังไม่มี Docker image ที่ build แล้ว
3. เปิด terminal ในโฟลเดอร์ที่แตกไฟล์ แล้วทำตาม [คู่มือติดตั้ง server](deploy/README_TH.md)

ถ้าใช้ `git clone` โค้ดจะไม่มีไฟล์โมเดล ให้ดาวน์โหลดและแตก Release เช่นกัน แล้วคัดลอกเฉพาะโมเดลด้วย Python 3.10+:

```sh
git clone https://github.com/Nithi67160346/Nexora-CCTV.git
cd Nexora-CCTV
python3 scripts/install_models.py --from-dir /path/to/extracted-release
```

สคริปต์ตรวจขนาดและ SHA256 ของโมเดลทั้งหมดก่อนคัดลอก ไม่โหลด pickle และไม่ดาวน์โหลดเอง ใช้เฉพาะชุดโมเดลที่เชื่อถือได้ เพราะ runtime ต้องโหลดไฟล์โมเดลจากทีม

## โครงสร้าง

- `web/` — เว็บ API และตัวเล่นหลัก
- `integration/` — shared pose และการเชื่อม feature; `integration/tests/` เก็บการตรวจพฤติกรรม
- `app/`, `Location/`, `Wandering/`, `Fall/` — feature และ reference ของโมเดลตรวจล้ม
- `docker/`, `Dockerfile`, `compose*.yaml` — dependency และ container CPU/GPU
- `deploy/` — คู่มือ, manifest โมเดล, ตัวอย่าง HTTPS ที่มีรหัสผ่าน
- `scripts/install_models.py` — ติดตั้งหรือตรวจโมเดลจาก Release

ข้อมูลคลิป ใบหน้าที่ลงทะเบียน รีวิว และโปรไฟล์กล้องไม่อยู่ใน repo หรือ Release ข้อมูล runtime อยู่ใน Docker volume `nexora-server_nexora_data` เมื่อใช้ชื่อ project ตามคู่มือ

## ข้อจำกัดของโมเดล

Healthy หมายถึงระบบและ detector พร้อม ไม่ได้ยืนยันความแม่นยำ Fall ยังอาจแจ้งเตือนการนอนตั้งใจหรือ pose ที่ผิด และ LSTM เคยให้คะแนนสูงกับคลิปที่ไม่ได้ทำร้ายร่างกาย คะแนนที่แสดงไม่ใช่ accuracy ที่วัดจากชุดทดสอบ โมเดล LSTM ตรวจทั้งภาพ จึงไม่มีการระบุผู้กระทำจากโมเดลนี้ ต้องตรวจเหตุการณ์ก่อนใช้ตัดสินใจ

ใช้ scikit-learn **1.9.1** สำหรับ RF 54 features ตาม lock เดิม ไฟล์ AI1/scaler เก่าที่แนบใน bundle มีไว้รักษาชุดต้นฉบับและไม่ได้ deserialize โมเดล LSTM และ Fall ไม่ได้ถูกเทรนใหม่ใน Release นี้

ประวัติของ repo นี้เริ่มจาก snapshot สำหรับ server ไม่มีประวัติ Git ของ repo พัฒนาและไม่มีเอกสาร presentation / tech review ส่วน runtime AI ใช้ชุดล่าสุดเดิม ดู [การตรวจชุด deploy](deploy/VALIDATION_TH.md)
