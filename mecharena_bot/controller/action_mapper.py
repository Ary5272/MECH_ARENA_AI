"""
Maps NitroGen gamepad outputs → keyboard keys + relative mouse movement.

NitroGen was trained on gamepad data.  We play Mech Arena with keyboard,
so this module translates:

  j_left  (x, y)   →  WASD
  j_right (x, y)   →  relative mouse move (aim)
  buttons [21]     →  keyboard keys / mouse clicks

Gamepad button ordering  (shared.BUTTON_ACTION_TOKENS, index 0-20)
─────────────────────────────────────────────────────────────────────
 0  BACK           10  NORTH
 1  DPAD_DOWN      11  RIGHT_BOTTOM
 2  DPAD_LEFT      12  RIGHT_LEFT
 3  DPAD_RIGHT     13  RIGHT_RIGHT
 4  DPAD_UP        14  RIGHT_SHOULDER
 5  EAST           15  RIGHT_THUMB
 6  GUIDE          16  RIGHT_TRIGGER   ← shoot
 7  LEFT_SHOULDER  17  RIGHT_UP
 8  LEFT_THUMB     18  SOUTH
 9  LEFT_TRIGGER   19  START
                   20  WEST

Mech Arena PC (keyboard) default controls
─────────────────────────────────────────
  W/A/S/D          move
  Mouse            aim
  Left Click       fire / shoot
  Right Click      aim-down-sights
  Space            jump / dodge
  Q                ability 1
  E                ability 2
  R                reload
  Shift            sprint
  F                interact / melee
  Tab              scoreboard
  Escape           menu
"""

import time
import pyautogui

import config

pyautogui.FAILSAFE = False      # don't throw if mouse hits corner
pyautogui.PAUSE = 0.0           # remove built-in delay

# ---------------------------------------------------------------------------
# Button index → action  ("lmouse" / "rmouse" are handled as mouse clicks)
# ---------------------------------------------------------------------------
_BUTTON_NAMES = [
    'BACK', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'DPAD_UP',
    'EAST', 'GUIDE',
    'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_TRIGGER',
    'NORTH',
    'RIGHT_BOTTOM', 'RIGHT_LEFT', 'RIGHT_RIGHT',
    'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_TRIGGER', 'RIGHT_UP',
    'SOUTH', 'START', 'WEST',
]

# Map button name → pyautogui key string OR "lmouse" / "rmouse"
_BUTTON_MAP = {
    'SOUTH':          'space',      # A  → jump / dodge
    'EAST':           'r',          # B  → reload
    'NORTH':          'f',          # Y  → interact / melee
    'WEST':           'e',          # X  → ability 2
    'LEFT_SHOULDER':  'q',          # LB → ability 1
    'RIGHT_SHOULDER': 'shift',      # RB → sprint
    'LEFT_TRIGGER':   'rmouse',     # LT → aim-down-sights
    'RIGHT_TRIGGER':  'lmouse',     # RT → fire
    'DPAD_UP':        '1',          # D↑ → weapon slot 1
    'DPAD_DOWN':      '2',          # D↓ → weapon slot 2
    'DPAD_LEFT':      '3',          # D← → gadget slot 1
    'DPAD_RIGHT':     '4',          # D→ → gadget slot 2
    'START':          'escape',     # Start → menu
    'BACK':           'tab',        # Back → scoreboard
    'LEFT_THUMB':     'ctrl',       # L3 → crouch
    'RIGHT_THUMB':    'v',          # R3 → melee alt
    # GUIDE, RIGHT_BOTTOM, RIGHT_LEFT, RIGHT_RIGHT, RIGHT_UP → unmapped
}


class ActionMapper:
    """Translates one NitroGen action step into keyboard / mouse events."""

    def __init__(self):
        self._held_keys: set = set()    # currently held keyboard keys
        self._mouse_held: dict = {}     # 'lmouse'/'rmouse' → bool

    # ------------------------------------------------------------------
    def execute(self, j_left, j_right, buttons):
        """
        Args
        ----
        j_left   : np.ndarray shape (2,)   x, y in [-1, 1]
        j_right  : np.ndarray shape (2,)   x, y in [-1, 1]
        buttons  : np.ndarray shape (21,)  0/1 float
        """
        self._handle_movement(j_left)
        self._handle_mouse(j_right)
        self._handle_buttons(buttons)

    # ------------------------------------------------------------------
    def release_all(self):
        """Release every held key/button.  Call on shutdown or between matches."""
        for key in list(self._held_keys):
            pyautogui.keyUp(key)
        self._held_keys.clear()
        for btn, held in self._mouse_held.items():
            if held:
                btn_name = 'left' if btn == 'lmouse' else 'right'
                pyautogui.mouseUp(button=btn_name)
        self._mouse_held.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_movement(self, j_left):
        """j_left → WASD with deadzone."""
        dead = config.JOYSTICK_DEAD
        x, y = float(j_left[0]), float(j_left[1])

        # Build desired WASD state from joystick
        desired = set()
        if x < -dead:
            desired.add('a')
        elif x > dead:
            desired.add('d')
        if y < -dead:        # negative Y = forward on most gamepad conventions
            desired.add('w')
        elif y > dead:
            desired.add('s')

        self._sync_keys({'a', 'd', 'w', 's'}, desired)

    def _handle_mouse(self, j_right):
        """j_right → relative mouse movement for aiming."""
        x, y = float(j_right[0]), float(j_right[1])
        scale = config.MOUSE_SCALE
        dx = int(x * scale)
        dy = int(y * scale)
        if dx != 0 or dy != 0:
            pyautogui.moveRel(dx, dy, duration=0)

    def _handle_buttons(self, buttons):
        """buttons vector → key/mouse press/release."""
        thresh = config.BUTTON_THRESH

        for idx, name in enumerate(_BUTTON_NAMES):
            action = _BUTTON_MAP.get(name)
            if action is None:
                continue

            pressed = float(buttons[idx]) > thresh

            if action in ('lmouse', 'rmouse'):
                self._sync_mouse(action, pressed)
            else:
                self._sync_key(action, pressed)

    def _sync_keys(self, all_keys: set, desired: set):
        """Press/release a group of keys to match desired state."""
        for key in all_keys:
            self._sync_key(key, key in desired)

    def _sync_key(self, key: str, pressed: bool):
        if pressed and key not in self._held_keys:
            pyautogui.keyDown(key)
            self._held_keys.add(key)
        elif not pressed and key in self._held_keys:
            pyautogui.keyUp(key)
            self._held_keys.discard(key)

    def _sync_mouse(self, btn: str, pressed: bool):
        btn_name = 'left' if btn == 'lmouse' else 'right'
        currently = self._mouse_held.get(btn, False)
        if pressed and not currently:
            pyautogui.mouseDown(button=btn_name)
            self._mouse_held[btn] = True
        elif not pressed and currently:
            pyautogui.mouseUp(button=btn_name)
            self._mouse_held[btn] = False
