"""UI alerts follow temporal lifecycle, never a raw score threshold."""
def fall_display(status):
    if status.get('backend') == 'yolo_pose_rf_v2':
        score=status.get('fall_score')
        suffix=f" {score*100:.1f}%" if score is not None else ''
        phase=status.get('phase')
        if phase=='FALL_DETECTED':return 'V2 อาจล้ม'+suffix+' • ตรวจสอบ', 'alert'
        if phase=='ABNORMAL_MOVEMENT':return 'V2 ท่าผิดปกติ'+suffix, 'warning'
        if phase=='WARMING_UP':return f"V2 เก็บภาพ {status.get('sampled_frames',0)}/{status.get('required_frames',6)}", 'observing'
        if phase=='NORMAL':return 'V2 ท่าปกติ', 'normal'
    if status.get('phase') == 'FALL_DETECTED':
        return 'อาจล้ม • กรุณาตรวจสอบ', 'alert'
    if status.get('phase') == 'POSTURE_REVIEW':
        return 'ท่าทางเสี่ยง • รอตรวจสอบ', 'warning'
    if status.get('phase') == 'POSTURE_CHECK':
        return 'กำลังตรวจท่าทาง...', 'observing'
    if status.get('phase') == 'UNAVAILABLE':
        return 'ยังไม่มีภาพคนที่ใช้ประเมินได้', 'observing'
    if status.get('phase') == 'CONFIRMING':
        return 'กำลังตรวจท่าล้มต่อเนื่อง...', 'observing'
    if status.get('phase') == 'WARMING_UP':
        return 'กำลังเก็บท่าอ้างอิง...', 'observing'
    return 'ยังไม่มีเหตุแจ้งเตือนการล้ม', 'normal'
