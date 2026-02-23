"""
Auto-launcher for Plarium Play -> Mech Arena.
"""

import ctypes
import os
import subprocess
import time

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw


def _click_at(x: int, y: int):
    """Click at absolute screen coords, supporting negative values (multi-monitor)."""
    ctypes.windll.user32.SetCursorPos(x, y)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.02)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up

PLARIUM_EXE   = r"C:\Users\Kids\AppData\Local\PlariumPlay\PlariumPlay.exe"
WINDOW_TITLE  = "Plarium Play"
GAME_NAME     = "Mech Arena"
TEMPLATE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "play_button_template.png"))

MATCH_THRESHOLD = 0.75


class PlariumLauncher:

    def __init__(self):
        self._template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
        if self._template is None:
            raise FileNotFoundError(f"Play button template not found: {TEMPLATE_PATH}")

    # ------------------------------------------------------------------
    def launch_and_play(self) -> bool:
        # 1. Start Plarium Play if not running
        existing = [w for w in gw.getWindowsWithTitle(WINDOW_TITLE)
                    if w.title.strip() == WINDOW_TITLE]
        if existing:
            print("[Launcher] Plarium Play already running.")
        else:
            print("[Launcher] Starting Plarium Play ...")
            subprocess.Popen([PLARIUM_EXE])

        # 2. Wait for the Plarium Play window (exact title match)
        print("[Launcher] Waiting for Plarium Play window ...")
        win = None
        for _ in range(40):
            wins = gw.getWindowsWithTitle(WINDOW_TITLE)
            # Exact match only — avoid partial hits on terminal windows
            exact = [w for w in wins
                     if w.title.strip() == WINDOW_TITLE and w.width > 100]
            if exact:
                win = exact[0]
                print(f"[Launcher] Window ready: {win.width}x{win.height}"
                      f" @ ({win.left},{win.top})")
                break
            time.sleep(1)
        if win is None:
            raise RuntimeError("Plarium Play window never appeared.")

        # 3. Wait for the UI to load
        print("[Launcher] Waiting 5s for UI ...")
        time.sleep(5)

        # 4. Find and click the Play button
        print("[Launcher] Looking for Play button ...")
        if not self._click_play_button(win):
            raise RuntimeError(
                "Could not find Play button. Check assets/launcher_debug.png")

        # 5. Wait for Mech Arena window to appear
        print("[Launcher] Waiting for Mech Arena window", end="", flush=True)
        game_win = None
        for _ in range(30):
            time.sleep(1)
            print(".", end="", flush=True)
            candidates = gw.getWindowsWithTitle("Mech Arena")
            exact = [w for w in candidates
                     if w.title.strip() == "Mech Arena" and w.width > 100]
            if exact:
                game_win = exact[0]
                break
        print()

        if game_win is None:
            print("[Launcher] Mech Arena window not found after 30s.")
            return True

        print(f"[Launcher] Mech Arena window: {game_win.width}x{game_win.height}"
              f" @ ({game_win.left},{game_win.top})")

        # 6. Move Mech Arena to the same monitor as Plarium Play
        if win.left < 0 and game_win.left >= 0:
            # Plarium Play is on the left monitor, game opened on the right
            target_x = win.left
            target_y = 0
            print(f"[Launcher] Moving Mech Arena to left monitor ({target_x}, {target_y})")
            game_win.moveTo(target_x, target_y)
            time.sleep(0.5)
            print(f"[Launcher] Mech Arena now: {game_win.width}x{game_win.height}"
                  f" @ ({game_win.left},{game_win.top})")

        return True

    # ------------------------------------------------------------------
    def _click_play_button(self, win) -> bool:
        import mss
        region = {"left": win.left, "top": win.top,
                  "width": win.width, "height": win.height}
        img = cv2.cvtColor(np.array(mss.mss().grab(region)), cv2.COLOR_BGRA2BGR)
        cv2.imwrite("assets/launcher_debug.png", img)

        # DPI scaling: mss captures physical pixels, pyautogui uses logical
        img_h, img_w = img.shape[:2]
        scale_x = img_w / win.width
        scale_y = img_h / win.height
        print(f"[Launcher] DPI scale: {scale_x:.2f}x{scale_y:.2f}  "
              f"(captured {img_w}x{img_h}, window {win.width}x{win.height})")

        res = cv2.matchTemplate(img, self._template, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        print(f"[Launcher] Template match: {val:.3f}")
        if val < MATCH_THRESHOLD:
            return False

        th, tw = self._template.shape[:2]
        # Template match coords are in physical pixels — scale to logical
        cx = (loc[0] + tw // 2) / scale_x
        cy = (loc[1] + th // 2) / scale_y
        sx, sy = int(win.left + cx), int(win.top + cy)
        print(f"[Launcher] Clicking Play at ({sx}, {sy})")
        _click_at(sx, sy)
        return True
