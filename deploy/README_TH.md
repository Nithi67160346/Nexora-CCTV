# ติดตั้ง NEXORA บน server

เริ่มด้วย CPU ได้เมื่อยังไม่ทราบชนิด server ใช้ Linux/amd64 containers รุ่นนี้ยังไม่ใช่ระบบที่ผ่านการรับรองความแม่นยำ

## ตรวจเครื่องและโมเดล

```sh
docker version
docker compose version
python3 scripts/install_models.py --check
docker compose -p nexora-server config --quiet
```

บน Windows ใช้ `python` แทน `python3` และ Docker Desktop Linux Engine ต้อง running คำสั่งตรวจโมเดลไม่ต้องมี Torch หรือ scikit-learn

ไฟล์บังคับคือ `models/yolo26n-pose.pt`, `best_lstm_model.pth` และ PKL ทั้งห้าใน `Fall/models_yolo/` ซึ่งมีครบใน Release ตัวเลือก pose อื่นและโมเดลใบหน้าก็แนบไว้ และตรวจ checksum ได้ด้วย `--check` โฟลเดอร์วิดีโอ `videos/` จะถูกสร้างเมื่อเริ่ม Compose; สามารถอัปโหลดคลิปผ่านเว็บโดยไม่วางคลิปใน Git

## CPU

ผู้ดูแลเป็นผู้รัน build; ครั้งแรกต้องดาวน์โหลด dependency และอาจใช้เวลานาน แยกขั้น build ออกจากขั้นเริ่มระบบเพื่อเห็น error ต้นเหตุชัดเจน

```sh
docker compose -p nexora-server build nexora
docker compose -p nexora-server up -d --no-build --wait --wait-timeout 240 nexora
docker compose -p nexora-server ps
docker compose -p nexora-server logs --tail 100 nexora
```

เว็บอยู่ที่ `http://127.0.0.1:8002` บนเครื่อง server การเปิด localhost ของคอมพิวเตอร์ตัวเองจะไม่ใช่เว็บบน server สำหรับทดสอบจากเครื่องตัวเองใช้ SSH tunnel:

```sh
ssh -L 8002:127.0.0.1:8002 USER@SERVER_IP
```

เปิด `http://127.0.0.1:8002` บนเครื่องที่เปิด tunnel ถ้าพอร์ตนี้ถูกเว็บเดิมใช้ ให้เลือกพอร์ตฝั่งเครื่องตัวเองอื่น เช่น `18002:127.0.0.1:8002` แล้วเปิด localhost:18002

## NVIDIA GPU (เลือกเฉพาะเมื่อเครื่องรองรับ)

ตรวจ `nvidia-smi` บน server; ต้องมี NVIDIA driver และตั้งค่า Docker ให้เข้าถึง GPU ได้ตาม [Docker GPU prerequisites](https://docs.docker.com/compose/how-tos/gpu-support/)

```sh
docker compose -p nexora-server -f compose.yaml -f compose.gpu.yaml config --quiet
docker compose -p nexora-server -f compose.yaml -f compose.gpu.yaml build nexora
docker compose -p nexora-server -f compose.yaml -f compose.gpu.yaml up -d --no-build --wait --wait-timeout 240 nexora
docker compose -p nexora-server -f compose.yaml -f compose.gpu.yaml exec nexora python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

ต้องได้ `True` และชื่อ GPU จริงก่อนใช้ตัวเลือก GPU ในเว็บ ถ้าเลือก GPU แต่ runtime ใช้ไม่ได้ container จะรายงาน error ไม่มีการสลับเป็น CPU เงียบ ๆ

## ใช้ image ที่ build ไว้แล้ว

Release นี้ยังไม่มี image TAR หากมี image ที่ทีม build จาก source รุ่นนี้แล้ว ให้กำหนดชื่อ image ใน `.env` โดยคัดลอกจาก `.env.example` (`NEXORA_IMAGE` สำหรับ CPU / `NEXORA_GPU_IMAGE` สำหรับ GPU) และใช้ `docker load -i /path/image.tar` หรือ `docker pull IMAGE` ก่อน `up --no-build` โมเดลยังต้องติดตั้งแยกผ่าน bind mount เหมือนเดิม image ไม่รวม volume รีวิวหรือข้อมูลผู้ใช้

## เปิดผ่านโดเมน

ตัวแอปยังไม่มีระบบบัญชีผู้ใช้ จึง bind พอร์ตไว้กับ localhost ให้ใช้ HTTPS reverse proxy ที่ตรวจสิทธิ์ทุก path ก่อนเปิดให้เข้าจากเครือข่าย ตัวอย่าง `Caddyfile.example` ใช้ Caddy บน host รุ่น 2.8+:

1. ตั้ง DNS ของโดเมนไปที่ server และติดตั้ง Caddy
2. สร้าง password hash ด้วย `caddy hash-password` และใส่ hash พร้อมชื่อผู้ใช้ใน config เปลี่ยนโดเมนตัวอย่างเป็นโดเมนจริง
3. คัดลอก config ไปที่ตำแหน่งของ Caddy บน server แล้วตรวจ `caddy validate --config /etc/caddy/Caddyfile` ก่อน reload
4. ให้ reverse proxy เข้าถึง localhost:8002 และเปิดพอร์ต 80/443 ของ host ตามการตั้งค่า HTTPS

อย่า commit config ที่มีข้อมูลเข้าระบบกล้องหรือรหัสผ่านจริง ตัวอย่างไม่มี credential ที่ใช้งานได้ ดู [Caddy basic_auth](https://caddyserver.com/docs/caddyfile/directives/basic_auth) สำหรับรูปแบบ hash

เมื่อใช้กล้องของ browser ผ่านโดเมนต้องใช้ HTTPS กล้อง USB ของ server ไม่ได้ถูกส่งเข้า container โดยอัตโนมัติ ใช้อัปโหลดคลิป, browser camera หรือ RTSP ที่ server เข้าถึงได้

## อัปเดตและย้อนรุ่น

ส่งออกรีวิวและโปรไฟล์จากเว็บก่อนเปลี่ยนรุ่น เก็บสำรอง Docker volume ตามนโยบาย server และคัดลอก image tag/digest ของรุ่นที่ใช้อยู่

```sh
docker compose -p nexora-server stop nexora
git fetch --tags
git checkout v0.1.0-rc.1
# ติดตั้งโมเดลและเลือก CPU/GPU ตามขั้นตอนด้านบน จากนั้น build และ up
```

ใช้ tag ของรุ่นเดิมเมื่อต้องย้อนกลับ และตรวจว่าโมเดลตรง manifest ของรุ่นนั้น ห้ามใช้ `down -v` หากต้องการเก็บรีวิว/โปรไฟล์ การย้อน source ไม่ได้ย้อนฐานข้อมูล ต้องมี backup volume แยก หากใช้ GPU ให้ใช้ไฟล์ Compose overlay เดิมทุกครั้ง

ตรวจ AI readiness จาก `docker compose ... ps` และหน้า runtime ของเว็บ แล้วทดสอบคลิปสั้นที่ทีมติดป้ายเหตุการณ์ไว้ก่อนรันงานจริง รอบเตรียม repo นี้ไม่ได้ build image หรือ deploy บน server จริง
