# การตรวจชุดสำหรับ server — 9 ตุลาคม 2026

- Python regression ของตัวเล่นหลักและ feature: 181 cases, 180 ผ่าน / 1 skipped, ไม่มี failure หรือ error
- สคริปต์ติดตั้งโมเดล: 5 tests ผ่าน ครอบคลุม checksum ผิด, ไฟล์ขาด, path escape และการป้องกันเขียนทับ
- UI: 7 Node suites ผ่าน (display, face, feature toggle, multiview, product, review, violence)
- Docker Compose CPU และ GPU: `config --quiet` ผ่าน ตรวจ binding localhost, mount โมเดลแบบ read-only และ volume ของข้อมูล
- Python source 82 ไฟล์ parse ผ่านก่อนเพิ่มเอกสารผลนี้

ชุด regression ใช้ mocks สำหรับโมเดลและอุปกรณ์บางส่วน ไม่ได้วัดความแม่นยำจริง ยกเว้น test โหลด RF จริงออกจากรอบนี้เนื่องจาก Python host ไม่มี scikit-learn 1.9.1 และ Docker Engine ไม่เปิด test ใบหน้า pretrained ถูก skipped ในรอบนี้เพราะรันก่อนติดตั้งโมเดลในชุด deploy

ตรวจ SHA256 และขนาดโมเดลทั้ง 14 ไฟล์ตาม manifest รวมถึงรายการไฟล์และ checksum ของ ZIP ก่อนเผยแพร่ Release

ไม่ได้รัน Docker build, training, คลิปยาว หรือ deploy บน server จริง ผู้ดูแลต้อง build และตรวจ Healthy / CUDA / คลิปทดสอบที่ติดป้ายเหตุการณ์บนเครื่องปลายทาง รุ่นนี้ยัง experimental และมีข้อจำกัด false positive ที่ระบุใน README
