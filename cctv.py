import numpy as np
import cv2
import Cars
import time

# Mendefinisikan input dan output
cnt_up = 0
cnt_down = 0

# Membuka cctv
# Definisikan URL stream
url_cctv_yogya = 'https://pantau.margamandala.co.id:3443/merak/exit/exit.m3u8'

# Membuka video dari stream online
cap = cv2.VideoCapture(url_cctv_yogya)


# Menampilkan properti video
for i in range(19):
    print(i, cap.get(i))

w = cap.get(3)
h = cap.get(4)
frameArea = h * w
areaTH = frameArea / 500
print('Area Threshold:', areaTH)

# Garis masuk dan keluar
line_up = int(1.85 * (h / 3))
line_down = int(3 * (h / 4))
up_limit = int(2.65 * (h / 5))
down_limit = int(3.5 * (h / 4))

print("Red line y:", line_down)
print("Blue line y:", line_up)

line_down_color = (255, 0, 0)
line_up_color = (0, 0, 255)

# Membuat garis sebagai array titik
pts_L1 = np.array([[0, line_down], [w, line_down]], np.int32).reshape((-1, 1, 2))
pts_L2 = np.array([[0, line_up], [w, line_up]], np.int32).reshape((-1, 1, 2))
pts_L3 = np.array([[0, up_limit], [w, up_limit]], np.int32).reshape((-1, 1, 2))
pts_L4 = np.array([[0, down_limit], [w, down_limit]], np.int32).reshape((-1, 1, 2))

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Elemen struktural untuk morfologi
kernelOp = np.ones((3, 3), np.uint8)
kernelOp2 = np.ones((5, 5), np.uint8)
kernelCl = np.ones((11, 11), np.uint8)

# Variabel
font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau tidak bisa dibaca.")
        break

    for i in cars:
        i.age_one()

    # Pra-pemrosesan
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    try:
        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print('UP:', cnt_up)
        print('DOWN:', cnt_down)
        break

    # Kontur
    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit, down_limit):
                for i in cars:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_UP(line_down, line_up):
                            cnt_up += 1
                            print("ID:", i.getId(), 'naik pada', time.strftime("%c"))
                        elif i.going_DOWN(line_down, line_up):
                            cnt_down += 1
                            print("ID:", i.getId(), 'turun pada', time.strftime("%c"))
                        break

                if new:
                    p = Cars.MyCar(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

            # Gambar objek dan bounding box
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menghapus objek yang selesai
    for i in cars:
        if i.getState() == '1':
            if i.getDir() == 'down' and i.getY() > down_limit:
                i.setDone()
            elif i.getDir() == 'up' and i.getY() < up_limit:
                i.setDone()
        if i.timedOut():
            index = cars.index(i)
            cars.pop(index)
            del i

    # Tampilkan info dan garis
    str_up = 'Keluar: ' + str(cnt_up)
    str_down = 'Masuk: ' + str(cnt_down)
    frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
    frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
    frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

    cv2.putText(frame, str_up, (15, 40), font, 0.5, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, str_up, (15, 40), font, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (15, 60), font, 0.5, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, str_down, (15, 60), font, 0.5, (0, 0, 200), 1, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Frame', frame)

    # Tekan ESC untuk keluar
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
