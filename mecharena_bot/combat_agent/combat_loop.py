"""
Async inference + action-execution loop with menu navigation.

Architecture
------------
  InferenceThread  -- continuously runs NitroGen.predict() and pushes
                     16-step action chunks into a queue.

  CombatLoop.run() -- pops actions from the queue one step at a time
                     at ACTION_FPS, passes each step to the ActionMapper.

  Menu navigation  -- handles LOBBY (clicks BATTLE), POPUP (presses Escape),
                     MODE_SELECT (clicks 5v5 Deathmatch), and LOADING (waits).

SAFETY: The bot NEVER clicks Purchase/Buy buttons. Popups are dismissed
with Escape only. BATTLE button is found via template matching.
"""

import ctypes
import os
import time
import threading
import queue
import signal

import cv2
import numpy as np
import pyautogui


def _click_at(x: int, y: int):
    """Click at absolute screen coords, supporting negative values (multi-monitor)."""
    ctypes.windll.user32.SetCursorPos(x, y)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.02)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up

import config
from capture.screen_capture import ScreenCapture
from controller.action_mapper import ActionMapper
from state_detection.state_detector import StateDetector, State


_STEP_INTERVAL = 1.0 / config.ACTION_FPS   # seconds between action steps
_QUEUE_MAX     = 3                          # max buffered chunks before we drop

# Safety
pyautogui.PAUSE = 0.05

# Mode select template
_MODE_TEMPLATE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "mode_5v5_template.png"))


class _InferenceThread(threading.Thread):
    """Runs NitroGen in a background thread, feeding the action queue."""

    def __init__(self, wrapper, capture: ScreenCapture, action_q: queue.Queue):
        super().__init__(daemon=True)
        self.wrapper    = wrapper
        self.capture    = capture
        self.action_q   = action_q
        self._stop_evt  = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        print("[Inference] Thread started.")
        while not self._stop_evt.is_set():
            frame = self.capture.get_frame_resized(config.CAPTURE_W, config.CAPTURE_H)
            try:
                result = self.wrapper.predict(frame)
            except Exception as e:
                print(f"[Inference] Error: {e}")
                time.sleep(0.5)
                continue

            if self.action_q.full():
                try:
                    self.action_q.get_nowait()
                except queue.Empty:
                    pass

            self.action_q.put(result)

        print("[Inference] Thread stopped.")


class CombatLoop:
    """
    High-level controller.  Call run() to start the bot.
    Press Ctrl-C to stop cleanly.
    """

    def __init__(self, wrapper):
        self.wrapper    = wrapper
        self.capture    = ScreenCapture()
        self.mapper     = ActionMapper()
        self.detector   = StateDetector()
        self.action_q   = queue.Queue(maxsize=_QUEUE_MAX)
        self._running   = False

        # Current action chunk and step index within it
        self._chunk    : dict  = None
        self._step_idx : int   = 0

        # Cooldowns to avoid spamming clicks
        self._last_battle_click = 0.0
        self._last_escape_press = 0.0
        self._last_mode_click   = 0.0

        # Mode select template
        self._mode_template = None
        if os.path.exists(_MODE_TEMPLATE_PATH):
            self._mode_template = cv2.imread(_MODE_TEMPLATE_PATH, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------
    def run(self):
        print("[CombatLoop] Starting -- press Ctrl-C to stop.")
        self._running = True

        def _on_signal(sig, frame):
            print("\n[CombatLoop] Stopping ...")
            self._running = False
        signal.signal(signal.SIGINT, _on_signal)

        # Start inference thread
        inf_thread = _InferenceThread(self.wrapper, self.capture, self.action_q)
        inf_thread.start()

        try:
            while self._running:
                frame = self.capture.get_frame()
                state = self.detector.detect(frame)

                if state == State.LOADING:
                    print("[CombatLoop] Loading screen -- waiting ...")
                    self.mapper.release_all()
                    self._reset()
                    time.sleep(1.0)
                    continue

                if state == State.POPUP:
                    self._handle_popup()
                    time.sleep(0.5)
                    continue

                if state == State.LOBBY:
                    self.mapper.release_all()
                    self._handle_lobby(frame)
                    time.sleep(1.0)
                    continue

                if state == State.UNKNOWN:
                    # Could be mode select, mech selection, or transition.
                    # Do NOT press Escape — that backs out of menus.
                    now = time.time()
                    if now - self._last_battle_click < 15.0:
                        # Recently clicked BATTLE -- try mode select
                        self._handle_mode_select(frame)
                        time.sleep(1.0)
                        continue

                    # Otherwise just wait -- could be mech selection,
                    # matchmaking, or other transition. Game handles it.
                    time.sleep(0.5)
                    continue

                # --- IN_MATCH: execute next action step ------------------
                self._tick()
                time.sleep(_STEP_INTERVAL)

        finally:
            inf_thread.stop()
            inf_thread.join(timeout=5)
            self.mapper.release_all()
            print("[CombatLoop] Stopped.")

    # ------------------------------------------------------------------
    def _handle_popup(self):
        """Dismiss popup by pressing Escape. NEVER click Purchase/Buy."""
        now = time.time()
        if now - self._last_escape_press < 2.0:
            return
        print("[CombatLoop] POPUP detected -- pressing Escape to dismiss.")
        pyautogui.press("escape")
        self._last_escape_press = now
        self._reset()

    def _handle_lobby(self, frame):
        """Find and click the BATTLE button to start a match."""
        now = time.time()
        if now - self._last_battle_click < 5.0:
            return

        pos = self.detector.get_battle_button_pos(frame)
        if pos is not None:
            x, y = pos
            sx, sy = self._frame_to_screen(x, y)
            print(f"[CombatLoop] LOBBY -- clicking BATTLE at screen ({sx}, {sy})")
            _click_at(sx, sy)
            self._last_battle_click = now
        else:
            if now - self._last_escape_press > 3.0:
                print("[CombatLoop] LOBBY but no BATTLE found -- pressing Escape")
                pyautogui.press("escape")
                self._last_escape_press = now
        self._reset()

    def _handle_mode_select(self, frame):
        """Click the 5v5 Deathmatch mode button."""
        now = time.time()
        if now - self._last_mode_click < 5.0:
            return

        # Try template matching if we have a template
        if self._mode_template is not None:
            res = cv2.matchTemplate(
                frame, self._mode_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > 0.60:
                th, tw = self._mode_template.shape[:2]
                # Click center of the card (not just the text)
                x = loc[0] + tw // 2
                y = loc[1] + th + 80  # click below text, in card body
                # Convert from frame coords to screen coords
                sx, sy = self._frame_to_screen(x, y)
                print(f"[CombatLoop] MODE SELECT -- clicking 5v5 Deathmatch "
                      f"at screen ({sx}, {sy}) [match={val:.2f}]")
                _click_at(sx, sy)
                self._last_mode_click = now
                return

        # No template -- save snapshot for debugging
        snap = "assets/mode_select_screen.png"
        cv2.imwrite(snap, frame)
        print(f"[CombatLoop] MODE SELECT -- no template, saved {snap}")
        self._last_mode_click = now

    def _frame_to_screen(self, x, y):
        """Convert frame-relative coords to absolute screen coords."""
        region = self.capture._region
        return (region["left"] + x, region["top"] + y)

    # ------------------------------------------------------------------
    def _tick(self):
        """Execute one action step from the current chunk."""
        if self._chunk is None or self._step_idx >= self._chunk["j_left"].shape[0]:
            try:
                self._chunk     = self.action_q.get_nowait()
                self._step_idx  = 0
            except queue.Empty:
                return

        j_left  = self._chunk["j_left"][self._step_idx]
        j_right = self._chunk["j_right"][self._step_idx]
        buttons = self._chunk["buttons"][self._step_idx]

        self.mapper.execute(j_left, j_right, buttons)
        self._step_idx += 1

    def _reset(self):
        """Discard buffered actions (e.g. after a state change)."""
        self.wrapper.reset()
        while not self.action_q.empty():
            try:
                self.action_q.get_nowait()
            except queue.Empty:
                break
        self._chunk     = None
        self._step_idx  = 0
