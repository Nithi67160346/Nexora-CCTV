# NEXORA CCTV

ชุดสำหรับนำเว็บ NEXORA ไปติดตั้งบน server: ตัวเล่นหลักใช้ YOLO Pose ร่วมกับ Fall RandomForest 54 features และโมเดลภาพ ResNet18 + LSTM สำหรับการทำร้ายร่างกาย รวม Location, โซนเตียง, Wandering และใบหน้า

รุ่น `v0.2.0-rc.2` · เว็บ `qa-upload-delete-20261010` · สถานะทดลอง

รุ่นนี้เพิ่มเว็บแคมผ่าน browser พร้อมค้นหากล้อง, ประมวลผลคลิปตามเฟรมที่แสดง, ดูหลักฐานแจ้งเตือนแยกจากตัวเล่นหลัก, อัปโหลดหลายคลิปและลบคลิปจากรายการ พร้อมโมเดลการทำร้ายร่างกาย ResNet18 + LSTM v3

## ชุดพร้อมรันสำหรับ QA บน Windows

ดาวน์โหลด `NEXORA-QA-Ready-20261011.zip` และไฟล์ image `.part001`, `.part002`, `.part003` ให้ครบจาก [Release ล่าสุด](https://github.com/Nithi67160346/Nexora-CCTV/releases/tag/v0.2.0-rc.2) วางทั้งสี่ไฟล์ในโฟลเดอร์เดียวกันแล้วแตก ZIP ตรงนั้น เปิด Docker Desktop รอ Engine running แล้วดับเบิลคลิก `start_docker.cmd` ในโฟลเดอร์ที่แตก ตัวรันตรวจ checksum รวม image และนำเข้าเองครั้งแรก ไม่ต้อง build หรือเปิด PowerShell

ใช้ Windows Intel/AMD x64 กับ Docker Desktop แบบ Linux containers / WSL2 และเผื่อพื้นที่อย่างน้อย 35 GB หาก NVIDIA และ Docker CUDA พร้อมจะเลือก GPU อัตโนมัติ ไม่พร้อมใช้ CPU หยุดด้วย `stop_docker.cmd` ข้อมูลและรีวิวยังคงอยู่ ชุดนี้ไม่ได้ทดสอบ Windows ARM หรือ Mac

## ติดตั้ง

1. ติดตั้ง Docker Engine / Docker Desktop ที่ใช้ Linux containers และ Docker Compose v2
2. ใช้โค้ด `main` และดาวน์โหลด QA ZIP จาก Release ล่าสุดเพื่อรับโมเดล (server ไม่จำเป็นต้องดาวน์โหลด image parts ถ้าจะ build เอง)
3. ติดตั้งโมเดลตามคำสั่งด้านล่าง แล้วทำตาม [คู่มือติดตั้ง server](deploy/README_TH.md)

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

ดู [การตรวจชุด deploy](deploy/VALIDATION_TH.md) และ [Release ก่อนหน้า](https://github.com/Nithi67160346/Nexora-CCTV/releases/tag/v0.1.0-rc.1) หากต้องย้อนเวอร์ชัน โค้ดก่อนอัปเดตเก็บไว้ใน tag `backup/main-before-20261011` ด้วย
