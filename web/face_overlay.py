"""Render registered names, including Thai, without changing detector input."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np


@lru_cache(maxsize=1)
def label_font():
    from PIL import ImageFont
    for path in ('C:/Windows/Fonts/tahoma.ttf', '/usr/share/fonts/truetype/tlwg/Loma.ttf',
                 '/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf','C:/Windows/Fonts/arial.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, 17)
    return ImageFont.load_default()


def draw_person_label(frame, label, x, y, color):
    if label.isascii():
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        x=max(0,min(x,frame.shape[1]-size[0]-8))
        cv2.rectangle(frame, (x, max(0, y - 22)), (x + size[0] + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        return
    from PIL import Image, ImageDraw
    font = label_font()
    left, top, right, bottom = font.getbbox(label)
    x=max(0,min(x,frame.shape[1]-(right-left)-8))
    y0 = max(0, y - (bottom - top) - 10)
    x2=min(frame.shape[1],x+right-left+8);y2=min(frame.shape[0],y0+bottom-top+8)
    if x2<=x or y2<=y0:return
    # Convert only the label area, not the entire camera frame per person.
    image=Image.fromarray(cv2.cvtColor(frame[y0:y2,x:x2],cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,x2-x,y2-y0),fill=tuple(reversed(color)))
    draw.text((4-left,4-top),label,font=font,fill=(255,255,255))
    frame[y0:y2,x:x2]=np.asarray(image)[:,:,::-1]
