# การตรวจชุดสำหรับ QA / server — 11 ตุลาคม 2026

- ตัวรัน Docker: 62 tests ผ่าน รวม CPU/GPU อัตโนมัติ, ready image, รวม image parts, checksum ผิด, ไฟล์ขาด และไม่ build/import เมื่อใช้ CheckOnly / NoBuild
- เว็บชุดใหม่: 62 tests ผ่าน ครอบคลุม webcam, วงจรเริ่ม/หยุด, frame playback, upload/delete, LSTM v3 และตัวเล่นหลัก
- ติดตั้งโมเดล: 5 tests ผ่าน ตรวจ checksum และป้องกัน path escape / เขียนทับ
- UI: 8 Node suites ผ่าน ได้แก่ browser camera, clip library, event replay, frame player, main webcam, product, review และ feature toggle
- Compose CPU/GPU config ผ่าน; Python ทุกไฟล์ในชุด Git parse ผ่าน; ตรวจ manifest โมเดล 14 ไฟล์และไฟล์ทุกตัวใน QA ZIP
- reuse image ที่ผู้ใช้ build แล้ว ไม่ build/export ซ้ำ แยก TAR เดิมเป็น 3 ส่วนต่ำกว่า 2 GiB และตรวจ SHA256 ทั้ง TAR ก่อนจัด Release
- ก่อนจัด Release ได้ทดสอบ ready image เดียวกันบนเครื่องนี้: CPU และ RTX 3070 / CUDA Healthy; smoke ใช้เฟรมจำลองสั้น ตรวจ runtime และโมเดลสำเร็จ

Regression หลายรายการใช้โมเดล/อุปกรณ์จำลอง ไม่ได้วัดความแม่นยำของโมเดล ไม่ได้รัน training หรือคลิปยาว และยังไม่ได้ deploy บน server ของผู้ใช้หรือทดสอบฮาร์ดแวร์อื่น ชุดพร้อมรันรองรับ Windows Intel/AMD x64 กับ Docker Desktop Linux Engine เท่านั้นในรอบนี้

## ผลตรวจรุ่นก่อนหน้า — 9 ตุลาคม 2026

- Python regression ของตัวเล่นหลักและ feature: 181 cases, 180 ผ่าน / 1 skipped, ไม่มี failure หรือ error
- สคริปต์ติดตั้งโมเดล: 5 tests ผ่าน ครอบคลุม checksum ผิด, ไฟล์ขาด, path escape และการป้องกันเขียนทับ
- UI: 7 Node suites ผ่าน (display, face, feature toggle, multiview, product, review, violence)
- Docker Compose CPU และ GPU: `config --quiet` ผ่าน ตรวจ binding localhost, mount โมเดลแบบ read-only และ volume ของข้อมูล
- Python source 82 ไฟล์ parse ผ่านก่อนเพิ่มเอกสารผลนี้

ชุด regression ใช้ mocks สำหรับโมเดลและอุปกรณ์บางส่วน ไม่ได้วัดความแม่นยำจริง ยกเว้น test โหลด RF จริงออกจากรอบนี้เนื่องจาก Python host ไม่มี scikit-learn 1.9.1 และ Docker Engine ไม่เปิด test ใบหน้า pretrained ถูก skipped ในรอบนี้เพราะรันก่อนติดตั้งโมเดลในชุด deploy

ตรวจ SHA256 และขนาดโมเดลทั้ง 14 ไฟล์ตาม manifest รวมถึงรายการไฟล์และ checksum ของ ZIP ก่อนเผยแพร่ Release

ไม่ได้รัน Docker build, training, คลิปยาว หรือ deploy บน server จริง ผู้ดูแลต้อง build และตรวจ Healthy / CUDA / คลิปทดสอบที่ติดป้ายเหตุการณ์บนเครื่องปลายทาง รุ่นนี้ยัง experimental และมีข้อจำกัด false positive ที่ระบุใน README
