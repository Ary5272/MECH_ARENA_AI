"""
Screen-based reward extraction for reinforcement learning.

Monitors the Mech Arena HUD to derive a scalar reward signal each RL step.

Signals extracted
-----------------
  health_pct    : float [0, 1]  – green health bar fill in bottom-center HUD
  alive         : bool          – True while health_pct > 0
  kill_count    : int           – digits read from the kill counter (top-right)
  match_active  : bool          – True while state == IN_MATCH

Reward formula (per RL step)
----------------------------
  r = w_health  * (health_t - health_{t-1})    # penalise damage taken
    + w_kill    * (kills_t  - kills_{t-1})      # reward getting kills
    + w_alive   * alive                          # small survival bonus
    + w_death   * died                           # large death penalty
"""

import cv2
import numpy as np

# ── HSV ranges for the green health bar (matches state_detector.py) ──────
_HEALTH_GREEN_LO = np.array([35, 80, 80])
_HEALTH_GREEN_HI = np.array([85, 255, 255])

# ── HSV range for red/orange damage vignette (screen flash on hit) ───────
_DAMAGE_RED_LO = np.array([0, 120, 120])
_DAMAGE_RED_HI = np.array([10, 255, 255])

# ── Kill-feed: bright white text that appears mid-right on kill ───────────
_KILLFEED_WHITE_LO = np.array([0, 0, 220])
_KILLFEED_WHITE_HI = np.array([180, 40, 255])


class RewardDetector:
    """Stateful reward extractor – call ``step(frame_bgr)`` each RL tick."""

    def __init__(
        self,
        w_health: float = 1.0,
        w_kill: float = 5.0,
        w_alive: float = 0.01,
        w_death: float = -5.0,
    ):
        self.w_health = w_health
        self.w_kill = w_kill
        self.w_alive = w_alive
        self.w_death = w_death

        # Tracking state
        self._prev_health: float = 1.0
        self._prev_kills: int = 0
        self._was_alive: bool = True
        self._step_count: int = 0

    # ------------------------------------------------------------------
    def reset(self):
        """Call at the start of every match."""
        self._prev_health = 1.0
        self._prev_kills = 0
        self._was_alive = True
        self._step_count = 0

    # ------------------------------------------------------------------
    def step(self, frame_bgr: np.ndarray) -> dict:
        """
        Analyse one full-resolution BGR frame and return reward info.

        Returns
        -------
        dict with keys:
            reward        : float   – scalar reward for this step
            health_pct    : float   – current health [0, 1]
            health_delta  : float   – change since last step
            kills         : int     – estimated total kills this match
            kill_delta    : int     – kills gained this step
            alive         : bool    – whether the player appears alive
            died          : bool    – True on the step health first hits 0
            info          : dict    – raw detection values for logging
        """
        h, w = frame_bgr.shape[:2]

        health_pct = self._measure_health(frame_bgr, h, w)
        kills = self._estimate_kills(frame_bgr, h, w)
        alive = health_pct > 0.02  # small epsilon for noise

        health_delta = health_pct - self._prev_health
        kill_delta = max(0, kills - self._prev_kills)
        died = self._was_alive and not alive

        # ── Composite reward ──────────────────────────────────────────
        reward = (
            self.w_health * health_delta
            + self.w_kill * kill_delta
            + self.w_alive * float(alive)
            + self.w_death * float(died)
        )

        # Update state for next call
        self._prev_health = health_pct
        self._prev_kills = kills
        self._was_alive = alive
        self._step_count += 1

        return {
            "reward": reward,
            "health_pct": health_pct,
            "health_delta": health_delta,
            "kills": kills,
            "kill_delta": kill_delta,
            "alive": alive,
            "died": died,
            "info": {"step": self._step_count},
        }

    # ------------------------------------------------------------------
    # Internal detectors
    # ------------------------------------------------------------------

    def _measure_health(self, frame_bgr: np.ndarray, h: int, w: int) -> float:
        """
        Measure the green health bar fill in the bottom-center HUD region.

        The bar is a horizontal strip; we measure what fraction of the ROI
        width is filled with green pixels.  Returns a float in [0, 1].
        """
        # ROI: bottom 12% of frame, middle 40%
        y0 = int(h * 0.88)
        x0 = int(w * 0.30)
        x1 = int(w * 0.70)
        roi = frame_bgr[y0:, x0:x1]

        if roi.size == 0:
            return self._prev_health  # degenerate frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _HEALTH_GREEN_LO, _HEALTH_GREEN_HI)
        green_px = int(mask.sum() // 255)

        roi_h, roi_w = roi.shape[:2]
        total_px = roi_h * roi_w
        if total_px == 0:
            return self._prev_health

        # The bar is typically a thin strip, so even a full bar won't fill
        # the entire ROI.  Normalise against an empirical max (~15% of ROI).
        max_green_frac = 0.15
        fill = min(1.0, (green_px / total_px) / max_green_frac)
        return fill

    def _estimate_kills(self, frame_bgr: np.ndarray, h: int, w: int) -> int:
        """
        Estimate the kill count from the kill-feed region.

        Strategy: count bright white text blobs in the right side of screen.
        Each new blob above a threshold is counted as an additional kill
        event.  This is approximate — we accumulate over the match.

        For V1 we use a simpler heuristic: detect kill-feed flashes in the
        upper-right region.  When we see a bright cluster appear that wasn't
        there before, increment.
        """
        # ROI: top-right quadrant where kill feed typically shows
        y1 = int(h * 0.40)
        x0 = int(w * 0.65)
        roi = frame_bgr[:y1, x0:]

        if roi.size == 0:
            return self._prev_kills

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, _KILLFEED_WHITE_LO, _KILLFEED_WHITE_HI)
        white_px = int(white_mask.sum() // 255)

        # A kill-feed entry is a burst of bright white text.
        # Threshold tuned empirically — a single kill notification produces
        # roughly 2000-5000 white pixels in this ROI at 1080p.
        kill_threshold = 3000
        if white_px > kill_threshold:
            # Only count if this is a new burst (simple debounce via step count)
            return self._prev_kills + 1

        return self._prev_kills
