"""
Screen capture with optional Plarium Play window targeting.

If the game window is found, captures only that region.
Falls back to the primary monitor if the window cannot be located.
"""

import cv2
import numpy as np
import mss

import config


def _find_window_region(title: str):
    """
    Return an mss monitor dict for the window with the given title,
    or None if not found.
    Uses exact title matching to avoid false positives (e.g. terminal
    windows whose title contains 'Mech Arena').
    """
    try:
        import pygetwindow as gw
        # getWindowsWithTitle does substring matching, so we filter
        # for an exact match (stripped) to avoid false positives.
        candidates = gw.getWindowsWithTitle(title)
        windows = [w for w in candidates
                   if w.title.strip() == title and w.width > 0 and w.height > 0]
        if not windows:
            return None
        w = windows[0]
        return {"left": w.left, "top": w.top,
                "width": w.width, "height": w.height}
    except Exception:
        return None


class ScreenCapture:
    def __init__(self, window_title: str = None):
        self.sct = mss.mss()
        self._title = window_title or config.GAME_WINDOW_TITLE
        self._region = None          # set by _refresh_region()
        self._refresh_region()

    # ------------------------------------------------------------------
    def _refresh_region(self):
        region = _find_window_region(self._title)
        if region:
            self._region = region
            print(f"[Capture] Locked to '{self._title}' "
                  f"@ {region['width']}×{region['height']} "
                  f"+{region['left']}+{region['top']}")
        else:
            self._region = self.sct.monitors[1]
            print(f"[Capture] Window '{self._title}' not found, "
                  f"capturing primary monitor.")

    # ------------------------------------------------------------------
    def get_frame(self) -> np.ndarray:
        """Return a full-resolution BGR frame of the capture region."""
        img = self.sct.grab(self._region)
        frame = np.array(img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def get_frame_resized(self, w: int = None, h: int = None) -> np.ndarray:
        """Return a BGR frame resized to (w, h) for model input.
        If the captured region is already the target size, returns as-is."""
        w = w or config.CAPTURE_W
        h = h or config.CAPTURE_H
        frame = self.get_frame()
        fh, fw = frame.shape[:2]
        if fw == w and fh == h:
            return frame
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def reacquire_window(self):
        """Call this if Plarium Play was restarted between matches."""
        self._refresh_region()


if __name__ == "__main__":
    cap = ScreenCapture()
    while True:
        frame = cap.get_frame()
        cv2.imshow("AI Vision", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
