"""
Voice-controlled rock-dodging game ("left" / "right").
Run: python game.py   (requires kws_rnn.keras + norm.npz from train_rnn.py)
Left/right arrow keys also work for testing; press R to restart after losing.
"""
import queue
import random
import threading
import time

import numpy as np
import pygame
import sounddevice as sd
import tensorflow as tf

from train_rnn import LABELS, SR, mfcc

CONF_THRESHOLD = 0.85   # minimum confidence required to accept a command
COOLDOWN = 0.6          # seconds, prevents one spoken command from being detected twice
MIN_RMS = 0.01          # volume threshold; increase it in a noisy room

cmd_q = queue.Queue()
buf = np.zeros(SR, dtype=np.float32)    
lock = threading.Lock()


def audio_callback(indata, frames, t, status):
    global buf
    with lock:
        buf = np.roll(buf, -frames)
        buf[-frames:] = indata[:, 0]


def listener():
    """Background thread: every 0.2s, take the most recent 1 second of audio -> RNN -> push the command to the queue."""
    model = tf.keras.models.load_model("kws_rnn.keras")
    norm = np.load("norm.npz")
    mean, std = norm["mean"], norm["std"]
    last = 0.0
    with sd.InputStream(samplerate=SR, channels=1, blocksize=1600,
                        callback=audio_callback):
        while True:
            time.sleep(0.2)
            with lock:
                y = buf.copy()
            if np.sqrt(np.mean(y ** 2)) < MIN_RMS:    
                continue
            x = (mfcc(y) - mean) / std
            p = model(x[None], training=False).numpy()[0]
            k = int(p.argmax())
            if k < 2 and p[k] > CONF_THRESHOLD and time.time() - last > COOLDOWN:
                cmd_q.put(LABELS[k])
                last = time.time()


def main():
    threading.Thread(target=listener, daemon=True).start()

    pygame.init()
    W, H, LANES = 600, 700, 5
    LW = W // LANES
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Voice Dodge — say LEFT / RIGHT")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    def reset():
        return dict(lane=LANES // 2, rocks=[], score=0, speed=3.0,
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

        # Commands from voice (or keyboard)
        while not cmd_q.empty():
            c = cmd_q.get()
            s["last"] = c
            if not s["over"]:
                # Only say left and right
                step = -1 if c == "left" else 1
                s["lane"] = min(LANES - 1, max(0, s["lane"] + step))

        player = pygame.Rect(s["lane"] * LW + 20, H - 90, LW - 40, 70)

        if not s["over"]:
            s["spawn"] += dt
            if s["spawn"] > 1400:                        # spawn a new rock every 1.4 second
                s["spawn"] = 0
                lane = random.randrange(LANES)
                s["rocks"].append(pygame.Rect(lane * LW + 15, -60, LW - 30, 50))
            for r in s["rocks"]:
                r.y += s["speed"]
            if any(r.colliderect(player) for r in s["rocks"]):
                s["over"] = True
            passed = [r for r in s["rocks"] if r.top > H]
            s["score"] += len(passed)
            s["speed"] += 0.15 * len(passed)             # faster if play longer
            s["rocks"] = [r for r in s["rocks"] if r.top <= H]

        # Front end
        screen.fill((25, 28, 40))
        for i in range(1, LANES):
            pygame.draw.line(screen, (50, 55, 75), (i * LW, 0), (i * LW, H))
        for r in s["rocks"]:
            pygame.draw.rect(screen, (140, 130, 120), r, border_radius=12)
        pygame.draw.rect(screen, (80, 170, 255), player, border_radius=10)
        screen.blit(font.render(f"Score: {s['score']}", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Heard: {s['last']}", True, (255, 220, 120)), (10, 45))
        if s["over"]:
            msg = font.render("GAME OVER - press R", True, (255, 90, 90))
            screen.blit(msg, msg.get_rect(center=(W // 2, H // 2)))
        pygame.display.flip()


if __name__ == "__main__":
    main()
