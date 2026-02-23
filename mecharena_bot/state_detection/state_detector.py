"""
Lightweight state detector using colour heuristics -- no extra model needed.

States
------
  IN_MATCH   -- actively playing; inference + input should run
  LOBBY      -- main menu with BATTLE button visible; click BATTLE
  POPUP      -- offer/dialog overlay; press Escape to dismiss
  LOADING    -- loading screen; pause inference
  UNKNOWN    -- can't tell; treat as lobby (safe default)

Detection strategy
------------------
Mech Arena's in-match HUD has a red/orange health bar in the bottom-left.
The lobby has a distinctive salmon/pink BATTLE button in the bottom-right.
Popups darken the background and show a dialog in the center.
Loading screens are predominantly dark.
"""

import cv2
import numpy as np
import os

import config


# HSV ranges (H 0-179, S 0-255, V 0-255 in OpenCV)
# In-match: green health bar at bottom-center
_HEALTH_GREEN_LO = np.array([35, 80, 80])
_HEALTH_GREEN_HI = np.array([85, 255, 255])
# In-match: cyan ability buttons at bottom-left/right
_ABILITY_CYAN_LO = np.array([85, 60, 100])
_ABILITY_CYAN_HI = np.array([110, 255, 255])

_LOADING_DARK_THRESH = 30   # mean brightness below this -> loading screen

# Template for BATTLE button (loaded once)
_BATTLE_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "battle_button_template.png")
_BATTLE_TEMPLATE = None
_BATTLE_MATCH_THRESH = 0.65


def _load_battle_template():
    global _BATTLE_TEMPLATE
    if _BATTLE_TEMPLATE is None:
        t = cv2.imread(os.path.normpath(_BATTLE_TEMPLATE_PATH), cv2.IMREAD_COLOR)
        if t is not None:
            _BATTLE_TEMPLATE = t
    return _BATTLE_TEMPLATE


class State:
    IN_MATCH    = "IN_MATCH"
    LOBBY       = "LOBBY"
    MODE_SELECT = "MODE_SELECT"
    POPUP       = "POPUP"
    LOADING     = "LOADING"
    UNKNOWN     = "UNKNOWN"


class StateDetector:
    def __init__(self):
        self._last = State.UNKNOWN
        self._battle_template = _load_battle_template()

    def detect(self, frame_bgr: np.ndarray) -> str:
        """
        Args
        ----
        frame_bgr : np.ndarray  full-resolution BGR frame (1920x1080)

        Returns
        -------
        One of the State constants.
        """
        h, w = frame_bgr.shape[:2]

        # -- Loading screen: almost entirely dark -------------------------
        brightness = float(np.mean(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)))
        if brightness < _LOADING_DARK_THRESH:
            self._last = State.LOADING
            return State.LOADING

        # -- Popup: bright dialog in center with darkened edges -----------
        # Check BEFORE lobby, since BATTLE button can be visible behind popup
        center = frame_bgr[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
        center_bright = float(np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)))
        edge_top = frame_bgr[:h // 8, :]
        edge_bright = float(np.mean(cv2.cvtColor(edge_top, cv2.COLOR_BGR2GRAY)))

        if center_bright > 80 and center_bright > edge_bright * 1.8 and brightness > 40:
            self._last = State.POPUP
            return State.POPUP

        # -- Lobby: check for BATTLE button (before health bar) -----------
        # This prevents the orange Shop/NEW badges from triggering IN_MATCH
        has_battle = False
        if self._battle_template is not None:
            res = cv2.matchTemplate(
                frame_bgr, self._battle_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > _BATTLE_MATCH_THRESH:
                has_battle = True

        if has_battle:
            self._last = State.LOBBY
            return State.LOBBY

        # -- In-match: green health bar (bottom-center) + cyan ability
        #    buttons (bottom-left/right corners) --------------------------
        # Green health bar at bottom center
        roi_bc = frame_bgr[h * 7 // 8:, w // 3: w * 2 // 3]
        hsv_bc = cv2.cvtColor(roi_bc, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_bc, _HEALTH_GREEN_LO, _HEALTH_GREEN_HI)
        green_px = int(green_mask.sum() // 255)

        # Cyan ability buttons at bottom-left
        roi_bl = frame_bgr[h * 3 // 4:, :w // 6]
        hsv_bl = cv2.cvtColor(roi_bl, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv_bl, _ABILITY_CYAN_LO, _ABILITY_CYAN_HI)
        cyan_px = int(cyan_mask.sum() // 255)

        if green_px > 5000 or cyan_px > 5000:
            self._last = State.IN_MATCH
            return State.IN_MATCH

        # -- Fallback -----------------------------------------------------
        self._last = State.UNKNOWN
        return State.UNKNOWN

    @property
    def last(self) -> str:
        return self._last

    def get_battle_button_pos(self, frame_bgr: np.ndarray):
        """Return (x, y) screen position of BATTLE button center, or None."""
        if self._battle_template is None:
            return None
        res = cv2.matchTemplate(
            frame_bgr, self._battle_template, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        if val < _BATTLE_MATCH_THRESH:
            return None
        th, tw = self._battle_template.shape[:2]
        return (loc[0] + tw // 2, loc[1] + th // 2)
