# เปิด NEXORA สำหรับ QA

ดาวน์โหลด QA ZIP และ image parts ทั้งสามจาก Release v0.2.0-rc.1 วางไฟล์ทั้งหมดในโฟลเดอร์เดียวกัน แล้วแตก ZIP ตรงนั้น

ใช้ Windows Intel/AMD x64 ติดตั้ง Docker Desktop แบบ Linux containers / WSL2 เปิด Docker รอ Engine running แล้วดับเบิลคลิก start_docker.cmd ในโฟลเดอร์ที่แตก

ครั้งแรกตัวรันตรวจ checksum รวม image และนำเข้า ไม่ build หรือดาวน์โหลด dependency เมื่อ NVIDIA และ Docker CUDA พร้อมจะใช้ GPU อัตโนมัติ ไม่พร้อมใช้ CPU ครั้งถัดไปใช้ image เดิม

เผื่อพื้นที่อย่างน้อย 35 GB หยุดด้วย stop_docker.cmd ข้อมูลและรีวิวยังคงอยู่ หากเกิด error ให้ส่ง log ใน local_only/qa ให้ทีม

เปิด webcam ด้วยปุ่มค้นหา webcam และอนุญาตกล้องใน browser ใช้ได้บน localhost หรือ HTTPS อัปโหลดคลิปหลายไฟล์แล้วเลือกจากรายการได้ ลบด้วยปุ่มของแต่ละแถวและยืนยันก่อนลบ คลิปที่กำลังใช้งานลบไม่ได้

ไฟล์ source จาก Git ไม่มีโมเดลหรือ image ผู้ดูแล server อ่าน deploy/README_TH.md ส่วน QA ใช้ไฟล์พร้อมรันจาก Release
