"""
Bước 4-5: Game né đá điều khiển bằng giọng nói ("left" / "right").
Chạy: python game.py   (cần kws_rnn.keras + norm.npz từ train_rnn.py)
Phím mũi tên trái/phải vẫn dùng được để test; R = chơi lại sau khi thua.
"""
import math
import queue
import random
import threading
import time

import numpy as np
import pygame
import sounddevice as sd
import tensorflow as tf

from train_rnn import LABELS, SR, mfcc

CONF_THRESHOLD = 0.70   # độ tự tin tối thiểu để chấp nhận lệnh
COOLDOWN = 0.6          # giây — tránh 1 tiếng hô bị nhận 2 lần
MIN_RMS = 0.005         # ngưỡng âm lượng; tăng lên nếu phòng ồn

cmd_q = queue.Queue()
buf = np.zeros(SR, dtype=np.float32)    # bộ đệm vòng 1 giây gần nhất
lock = threading.Lock()


def audio_callback(indata, frames, t, status):
    global buf
    with lock:
        buf = np.roll(buf, -frames)
        buf[-frames:] = indata[:, 0]


# Trạng thái micro/model để hiển thị trên màn hình (giúp tìm lỗi)
mic = {"status": "loading model...", "rms": 0.0, "pred": "-", "prob": 0.0}


def listener():
    """Luồng nền: cứ 0.2s lấy 1 giây audio gần nhất -> RNN -> đẩy lệnh vào queue."""
    try:
        model = tf.keras.models.load_model("kws_rnn.keras")
        norm = np.load("norm.npz")
        mean, std = norm["mean"], norm["std"]
        last = 0.0
        mic["status"] = "opening mic..."
        with sd.InputStream(samplerate=SR, channels=1, blocksize=1600,
                            callback=audio_callback):
            mic["status"] = "listening"
            while True:
                time.sleep(0.2)
                with lock:
                    y = buf.copy()
                rms = float(np.sqrt(np.mean(y ** 2)))
                mic["rms"] = rms
                if rms < MIN_RMS:                       # quá im -> bỏ qua
                    continue
                x = (mfcc(y) - mean) / std
                p = model(x[None], training=False).numpy()[0]
                k = int(p.argmax())
                mic["pred"], mic["prob"] = LABELS[k], float(p[k])
                if k < 2 and p[k] > CONF_THRESHOLD and time.time() - last > COOLDOWN:
                    cmd_q.put(LABELS[k])
                    last = time.time()
                    with lock:                          # xoá bộ đệm để 1 tiếng hô không bị nhận 2 lần
                        buf[:] = 0
    except Exception as ex:                             # hiện lỗi lên màn hình thay vì im lặng
        mic["status"] = f"ERROR: {ex}"[:70]
        print("Listener error:", repr(ex))


def draw_person(screen, rect, t, running=True):
    """Vẽ một người nhỏ bên trong rect (40x70). t = thời gian (ms) để tạo dáng chạy."""
    cx, top = rect.centerx, rect.top
    skin, hair = (255, 214, 170), (60, 40, 30)
    shirt, pants, shoe = (80, 170, 255), (40, 60, 110), (30, 30, 30)
    swing = int(7 * math.sin(t * 0.015)) if running else 0   # chân/tay đung đưa

    # Chân (vẽ trước để thân che phần đầu chân)
    pygame.draw.line(screen, pants, (cx - 5, top + 46), (cx - 6 + swing, top + 64), 7)
    pygame.draw.line(screen, pants, (cx + 5, top + 46), (cx + 6 - swing, top + 64), 7)
    pygame.draw.ellipse(screen, shoe, (cx - 12 + swing, top + 62, 12, 7))
    pygame.draw.ellipse(screen, shoe, (cx + 1 - swing, top + 62, 12, 7))

    # Tay
    pygame.draw.line(screen, skin, (cx - 11, top + 28), (cx - 18 - swing, top + 44), 5)
    pygame.draw.line(screen, skin, (cx + 11, top + 28), (cx + 18 + swing, top + 44), 5)

    # Thân (áo)
    pygame.draw.rect(screen, shirt, (cx - 12, top + 24, 24, 26), border_radius=7)

    # Đầu: mặt, tóc, mắt
    pygame.draw.circle(screen, skin, (cx, top + 13), 11)
    pygame.draw.arc(screen, hair, (cx - 11, top + 2, 22, 20), 0.2, math.pi - 0.2, 6)
    pygame.draw.circle(screen, (20, 20, 20), (cx - 4, top + 13), 2)
    pygame.draw.circle(screen, (20, 20, 20), (cx + 4, top + 13), 2)
    if running:   # miệng cười
        pygame.draw.arc(screen, (150, 60, 60), (cx - 4, top + 14, 8, 6), math.pi, 2 * math.pi, 2)
    else:         # thua: miệng chữ O
        pygame.draw.circle(screen, (150, 60, 60), (cx, top + 19), 2)


def draw_brick(screen, rect):
    """Vẽ một khối gạch đỏ: 3 hàng gạch so le, ngăn bằng vữa xám."""
    brick, dark, mortar = (178, 74, 52), (140, 52, 36), (205, 195, 180)
    pygame.draw.rect(screen, mortar, rect, border_radius=4)          # nền vữa
    rows, gap = 3, 3
    row_h = (rect.height - gap * (rows + 1)) / rows
    for r in range(rows):
        y = rect.top + gap + r * (row_h + gap)
        offset = 0 if r % 2 == 0 else -rect.width // 4               # hàng lẻ lệch nửa viên
        x = rect.left + gap + offset
        while x < rect.right - gap:
            w = rect.width // 2 - gap
            left, right = max(x, rect.left + gap), min(x + w, rect.right - gap)
            if right - left > 2:
                b = pygame.Rect(left, y, right - left, row_h)
                pygame.draw.rect(screen, brick, b, border_radius=2)
                pygame.draw.line(screen, dark, b.bottomleft, b.bottomright, 2)  # bóng dưới
            x += w + gap


def name_screen(screen, clock, W, H):
    """Màn hình nhập tên nhân vật. Trả về tên, hoặc None nếu người chơi đóng cửa sổ."""
    title_font = pygame.font.SysFont("arial", 40, bold=True)
    font = pygame.font.SysFont("arial", 30)
    small = pygame.font.SysFont("arial", 20)
    name = ""
    pygame.key.start_text_input()
    while True:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return None
            if e.type == pygame.TEXTINPUT and len(name) < 16:
                name += e.text                                  # gõ được cả tiếng Việt
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and name.strip():
                    return name.strip()

        screen.fill((25, 28, 40))
        t = title_font.render("VOICE DODGE", True, (255, 220, 120))
        screen.blit(t, t.get_rect(center=(W // 2, H // 3 - 40)))
        p = font.render("Enter your character name:", True, (255, 255, 255))
        screen.blit(p, p.get_rect(center=(W // 2, H // 3 + 30)))

        box = pygame.Rect(W // 2 - 160, H // 3 + 65, 320, 50)
        pygame.draw.rect(screen, (45, 50, 70), box, border_radius=8)
        pygame.draw.rect(screen, (80, 170, 255), box, 2, border_radius=8)
        cursor = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else ""
        n = font.render(name + cursor, True, (255, 255, 255))
        screen.blit(n, n.get_rect(center=box.center))

        draw_person(screen, pygame.Rect(W // 2 - 20, H // 3 + 150, 40, 70),
                    pygame.time.get_ticks())
        h = small.render("Press Enter to start", True, (160, 165, 185))
        screen.blit(h, h.get_rect(center=(W // 2, H // 3 + 260)))
        pygame.display.flip()


# Độ khó: tốc độ rơi ban đầu (px/frame), khoảng cách thả gạch (ms), mức tăng tốc mỗi viên né được
LEVELS = [
    ("Easy",   "Slow",     (110, 200, 120), dict(speed=2.0, spawn_ms=1800, accel=0.08)),
    ("Medium", "Faster",    (255, 200, 90),  dict(speed=3.5, spawn_ms=1300, accel=0.15)),
    ("Hard",   "Super fast!",  (255, 90, 90),   dict(speed=5.5, spawn_ms=900,  accel=0.25)),
]


def level_screen(screen, clock, W, H, player_name):
    """Chọn độ khó bằng ↑/↓ + Enter, phím 1/2/3, hoặc click chuột. Trả về index, None nếu đóng cửa sổ."""
    title_font = pygame.font.SysFont("arial", 34, bold=True)
    font = pygame.font.SysFont("arial", 30, bold=True)
    small = pygame.font.SysFont("arial", 20)
    sel = 0
    boxes = [pygame.Rect(W // 2 - 150, 230 + i * 95, 300, 75) for i in range(len(LEVELS))]
    while True:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return None
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(LEVELS)
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(LEVELS)
                elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    return e.key - pygame.K_1
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return sel
            if e.type == pygame.MOUSEMOTION:
                for i, b in enumerate(boxes):
                    if b.collidepoint(e.pos):
                        sel = i
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, b in enumerate(boxes):
                    if b.collidepoint(e.pos):
                        return i

        screen.fill((25, 28, 40))
        t = title_font.render(f"Hello, {player_name}!", True, (255, 220, 120))
        screen.blit(t, t.get_rect(center=(W // 2, 120)))
        p = small.render("Choose difficulty:", True, (255, 255, 255))
        screen.blit(p, p.get_rect(center=(W // 2, 180)))
        for i, (name, desc, color, _) in enumerate(LEVELS):
            b = boxes[i]
            pygame.draw.rect(screen, (55, 60, 85) if i == sel else (40, 44, 62), b, border_radius=12)
            pygame.draw.rect(screen, color if i == sel else (70, 75, 100), b, 3, border_radius=12)
            n = font.render(f"{i + 1}. {name}", True, color)
            screen.blit(n, n.get_rect(midleft=(b.left + 22, b.centery - 12)))
            d = small.render(desc, True, (190, 195, 210))
            screen.blit(d, d.get_rect(midleft=(b.left + 26, b.centery + 18)))
        h = small.render("Up/Down + Enter  ·  keys 1/2/3  ·  or click", True, (160, 165, 185))
        screen.blit(h, h.get_rect(center=(W // 2, H - 120)))
        pygame.display.flip()


def main():
    threading.Thread(target=listener, daemon=True).start()

    pygame.init()
    W, H, LANES = 600, 700, 5
    LW = W // LANES
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Voice Dodge — say LEFT / RIGHT")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 28)
    big = pygame.font.SysFont("arial", 44, bold=True)
    name_font = pygame.font.SysFont("arial", 16, bold=True)
    small = pygame.font.SysFont("arial", 14)

    player_name = name_screen(screen, clock, W, H)
    if player_name is None:
        pygame.quit()
        return
    pygame.key.stop_text_input()
    lv = level_screen(screen, clock, W, H, player_name)
    if lv is None:
        pygame.quit()
        return
    level_name, _, level_color, cfg = LEVELS[lv]
    while not cmd_q.empty():          # bỏ các lệnh giọng nói lọt vào lúc đang nhập tên
        cmd_q.get()

    def reset():
        return dict(lane=LANES // 2, rocks=[], score=0, speed=cfg["speed"],
                    spawn=0, over=False, last="-")

    s = reset()
    while True:
        dt = clock.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_LEFT:
                    cmd_q.put("left")
                elif e.key == pygame.K_RIGHT:
                    cmd_q.put("right")
                elif e.key == pygame.K_r and s["over"]:
                    s = reset()

        # Lệnh từ giọng nói (hoặc bàn phím)
        while not cmd_q.empty():
            c = cmd_q.get()
            s["last"] = c
            if not s["over"]:
                step = -1 if c == "left" else 1
                s["lane"] = min(LANES - 1, max(0, s["lane"] + step))

        # Hitbox gọn theo dáng người (rộng 40px), nằm giữa làn
        player = pygame.Rect(s["lane"] * LW + LW // 2 - 20, H - 90, 40, 70)

        if not s["over"]:
            s["spawn"] += dt
            if s["spawn"] > cfg["spawn_ms"]:             # khoảng cách thả gạch theo độ khó
                s["spawn"] = 0
                lane = random.randrange(LANES)
                s["rocks"].append(pygame.Rect(lane * LW + 15, -60, LW - 30, 50))
            for r in s["rocks"]:
                r.y += s["speed"]
            if any(r.colliderect(player) for r in s["rocks"]):
                s["over"] = True
            passed = [r for r in s["rocks"] if r.top > H]
            s["score"] += len(passed)
            s["speed"] += cfg["accel"] * len(passed)     # càng chơi càng nhanh
            s["rocks"] = [r for r in s["rocks"] if r.top <= H]

        # Vẽ
        screen.fill((25, 28, 40))
        for i in range(1, LANES):
            pygame.draw.line(screen, (50, 55, 75), (i * LW, 0), (i * LW, H))
        for r in s["rocks"]:
            draw_brick(screen, r)
        draw_person(screen, player, pygame.time.get_ticks(), running=not s["over"])
        tag = name_font.render(player_name, True, (255, 255, 255))   # tên trên đầu nhân vật
        screen.blit(tag, tag.get_rect(midbottom=(player.centerx, player.top - 4)))
        screen.blit(font.render(f"Score: {s['score']}", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Heard: {s['last']}", True, (255, 220, 120)), (10, 45))
        # Thanh debug micro ở đáy màn hình
        bar_w = int(min(1.0, mic["rms"] / 0.05) * 120)
        pygame.draw.rect(screen, (60, 65, 85), (10, H - 18, 120, 10), border_radius=3)
        pygame.draw.rect(screen, (110, 200, 120) if mic["rms"] >= MIN_RMS else (120, 120, 120),
                         (10, H - 18, bar_w, 10), border_radius=3)
        st_col = (255, 90, 90) if mic["status"].startswith("ERROR") else (170, 175, 195)
        dbg = small.render(f"mic: {mic['status']}  |  model: {mic['pred']} {mic['prob']:.2f}",
                           True, st_col)
        screen.blit(dbg, (140, H - 22))
        lv_txt = font.render(level_name, True, level_color)
        screen.blit(lv_txt, lv_txt.get_rect(topright=(W - 10, 10)))
        if s["over"]:
            l1 = big.render(f"GAME OVER, {player_name}", True, (255, 90, 90))
            l2 = font.render("Press R", True, (255, 255, 255))
            screen.blit(l1, l1.get_rect(center=(W // 2, H // 2 - 25)))
            screen.blit(l2, l2.get_rect(center=(W // 2, H // 2 + 25)))
        pygame.display.flip()


if __name__ == "__main__":
    main()
