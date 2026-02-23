"""
RL-augmented combat loop.

Replaces the plain CombatLoop with an RL training loop that:

1. Runs NitroGen to get a base action (16-step chunk)
2. Runs the RL policy to get a residual correction
3. Combines them and executes via ActionMapper
4. Measures reward from screen-based HUD detection
5. Stores transitions in a rollout buffer
6. Runs PPO updates between matches (or when the buffer is full)

Menu navigation (LOBBY, POPUP, MODE_SELECT, LOADING) is unchanged from
the original CombatLoop.

NOTE: mss is NOT thread-safe -- each thread needs its own ScreenCapture
instance (and therefore its own mss.mss() handle).
"""

import os
import time
import signal
import threading
import queue

import cv2
import numpy as np
import torch
import pyautogui
import ctypes

import config
from capture.screen_capture import ScreenCapture
from controller.action_mapper import ActionMapper
from state_detection.state_detector import StateDetector, State
from models.nitrogen_wrapper import NitroGenWrapper

from rl.policy_network import ActorCritic, ACTION_DIM, N_CONTINUOUS
from rl.rollout_buffer import RolloutBuffer
from rl.ppo_trainer import PPOTrainer
from rl.reward_detector import RewardDetector


def _click_at(x: int, y: int):
    """Click at absolute screen coords, supporting negative values."""
    ctypes.windll.user32.SetCursorPos(x, y)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.02)
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


_STEP_INTERVAL = 1.0 / config.ACTION_FPS
_QUEUE_MAX = 3

pyautogui.PAUSE = 0.05

_MODE_TEMPLATE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "mode_5v5_template.png"))

# Debug screenshot directory
_DEBUG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "assets", "debug_rl"))


class _InferenceThread(threading.Thread):
    """
    Background NitroGen inference.

    IMPORTANT: creates its OWN ScreenCapture because mss is not thread-safe.
    """

    def __init__(self, wrapper: NitroGenWrapper, action_q: queue.Queue):
        super().__init__(daemon=True)
        self.wrapper = wrapper
        self.action_q = action_q
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        # Create a fresh ScreenCapture in THIS thread so mss handles are local
        capture = ScreenCapture()
        print("[Inference] Thread started (own ScreenCapture).")
        while not self._stop_evt.is_set():
            try:
                frame = capture.get_frame_resized(config.CAPTURE_W,
                                                  config.CAPTURE_H)
            except Exception as e:
                print(f"[Inference] Capture error: {e}")
                time.sleep(0.5)
                continue

            try:
                result = self.wrapper.predict(frame)
            except Exception as e:
                print(f"[Inference] Predict error: {e}")
                time.sleep(0.5)
                continue

            if self.action_q.full():
                try:
                    self.action_q.get_nowait()
                except queue.Empty:
                    pass
            self.action_q.put(result)
        print("[Inference] Thread stopped.")


class RLCombatLoop:
    """
    RL training loop built on top of the existing bot pipeline.

    The RL policy outputs *residual* adjustments to NitroGen's actions.
    One RL decision is made per 16-step NitroGen chunk (~1 second).
    """

    def __init__(self, wrapper: NitroGenWrapper, rl_checkpoint: str = None):
        # ── Existing components ───────────────────────────────────────
        self.wrapper = wrapper
        self.capture = ScreenCapture()   # main-thread capture
        self.mapper = ActionMapper()
        self.detector = StateDetector()
        self.action_q = queue.Queue(maxsize=_QUEUE_MAX)
        self._running = False

        # Chunk state
        self._chunk = None
        self._step_idx = 0

        # Menu cooldowns
        self._last_battle_click = 0.0
        self._last_escape_press = 0.0
        self._last_mode_click = 0.0

        self._mode_template = None
        if os.path.exists(_MODE_TEMPLATE_PATH):
            self._mode_template = cv2.imread(_MODE_TEMPLATE_PATH,
                                             cv2.IMREAD_COLOR)

        # ── RL components ─────────────────────────────────────────────
        self.device = config.DEVICE
        self.policy = ActorCritic(obs_size=config.CAPTURE_W).to(self.device)
        self.reward_det = RewardDetector(
            w_health=config.RL_W_HEALTH,
            w_kill=config.RL_W_KILL,
            w_alive=config.RL_W_ALIVE,
            w_death=config.RL_W_DEATH,
        )
        self.buffer = RolloutBuffer(
            buffer_size=config.RL_ROLLOUT_STEPS,
            obs_shape=(3, config.CAPTURE_W, config.CAPTURE_H),
            action_dim=ACTION_DIM,
            device=self.device,
        )
        self.trainer = PPOTrainer(
            policy=self.policy,
            device=self.device,
            lr=config.RL_LR,
            gamma=config.RL_GAMMA,
            gae_lambda=config.RL_GAE_LAMBDA,
            clip_eps=config.RL_CLIP_EPS,
            n_epochs=config.RL_N_EPOCHS,
            batch_size=config.RL_BATCH_SIZE,
            entropy_coef=config.RL_ENTROPY_COEF,
        )

        # Load existing RL checkpoint if provided
        self._rl_ckpt_path = rl_checkpoint or os.path.join(
            config.BASE_DIR, "models", "rl_policy.pt")
        if os.path.exists(self._rl_ckpt_path):
            self.trainer.load(self._rl_ckpt_path)

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_matches = 0
        self._in_match = False

        # Debug screenshot counter
        self._snap_counter = 0
        os.makedirs(_DEBUG_DIR, exist_ok=True)

        # State tracking for screenshots
        self._prev_state = None

    # ==================================================================
    # Debug screenshots
    # ==================================================================
    def _save_debug(self, frame_bgr: np.ndarray, label: str):
        """Save a timestamped debug screenshot."""
        ts = time.strftime("%H%M%S")
        name = f"{self._snap_counter:04d}_{ts}_{label}.png"
        path = os.path.join(_DEBUG_DIR, name)
        cv2.imwrite(path, frame_bgr)
        print(f"[DEBUG] Screenshot -> {path}")
        self._snap_counter += 1

    # ==================================================================
    # Main loop
    # ==================================================================
    def run(self):
        print("[RLCombat] Starting RL training loop -- press Ctrl-C to stop.")
        self._running = True

        def _on_signal(sig, frame):
            print("\n[RLCombat] Stopping ...")
            self._running = False
        signal.signal(signal.SIGINT, _on_signal)

        # Inference thread gets its OWN ScreenCapture (mss thread-safety)
        inf_thread = _InferenceThread(self.wrapper, self.action_q)
        inf_thread.start()

        try:
            while self._running:
                frame_full = self.capture.get_frame()
                state = self.detector.detect(frame_full)

                # Screenshot on every state change
                if state != self._prev_state:
                    self._save_debug(frame_full, f"state_{state}")
                    print(f"[RLCombat] State: {self._prev_state} -> {state}")
                    self._prev_state = state

                # ── Non-match states (menu navigation) ────────────────
                if state == State.LOADING:
                    self._on_leave_match()
                    self.mapper.release_all()
                    self._reset_actions()
                    time.sleep(1.0)
                    continue

                if state == State.POPUP:
                    self._save_debug(frame_full, "POPUP_handling")
                    self._handle_popup()
                    time.sleep(0.5)
                    continue

                if state == State.LOBBY:
                    self._on_leave_match()
                    self.mapper.release_all()
                    self._save_debug(frame_full, "LOBBY_handling")
                    self._handle_lobby(frame_full)
                    time.sleep(1.0)
                    continue

                if state == State.UNKNOWN:
                    self._save_debug(frame_full, "UNKNOWN_state")
                    now = time.time()
                    if now - self._last_battle_click < 15.0:
                        self._handle_mode_select(frame_full)
                        time.sleep(1.0)
                        continue
                    time.sleep(0.5)
                    continue

                # ── IN_MATCH: RL-augmented action execution ───────────
                if not self._in_match:
                    self._on_enter_match()
                    self._save_debug(frame_full, "MATCH_started")

                self._rl_tick(frame_full)
                time.sleep(_STEP_INTERVAL)

        finally:
            inf_thread.stop()
            inf_thread.join(timeout=5)
            self.mapper.release_all()
            self._save_checkpoint()
            print(f"[RLCombat] Stopped. Matches played: {self._total_matches}")

    # ==================================================================
    # RL tick -- one action chunk
    # ==================================================================
    def _rl_tick(self, frame_full: np.ndarray):
        """
        Execute one RL step:
        1. Get NitroGen base action chunk from queue
        2. Run RL policy to get residual
        3. Combine and execute all 16 steps
        4. Capture post-action frame, measure reward
        5. Store transition
        """
        # Get next NitroGen chunk
        if (self._chunk is None
                or self._step_idx >= self._chunk["j_left"].shape[0]):
            try:
                self._chunk = self.action_q.get_nowait()
                self._step_idx = 0
            except queue.Empty:
                return

        # ── Step 1: Prepare observation ───────────────────────────────
        frame_small = self.capture.get_frame_resized(config.CAPTURE_W,
                                                     config.CAPTURE_H)
        obs_np = frame_small.transpose(2, 0, 1).astype(np.float32) / 255.0

        # NitroGen's base action: mean across the chunk
        base_j_left = self._chunk["j_left"].mean(axis=0)    # (2,)
        base_j_right = self._chunk["j_right"].mean(axis=0)  # (2,)
        base_buttons = self._chunk["buttons"].mean(axis=0)   # (21,)
        base_action_np = np.concatenate(
            [base_j_left, base_j_right, base_buttons]).astype(np.float32)

        # ── Step 2: RL policy forward pass ────────────────────────────
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        base_t = torch.from_numpy(base_action_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rl_action, log_prob, value = self.policy.get_action(obs_t, base_t)

        rl_action_np = rl_action.squeeze(0).cpu().numpy()
        log_prob_val = log_prob.item()
        value_val = value.item()

        # ── Step 3: Combine NitroGen base + RL residual ───────────────
        residual_j_left = rl_action_np[0:2]
        residual_j_right = rl_action_np[2:4]
        rl_buttons = rl_action_np[4:]  # (21,) binary from Bernoulli

        adj_chunk = {
            "j_left": np.clip(
                self._chunk["j_left"] + residual_j_left, -1.0, 1.0),
            "j_right": np.clip(
                self._chunk["j_right"] + residual_j_right, -1.0, 1.0),
            "buttons": np.where(
                rl_buttons > 0.5,
                np.maximum(self._chunk["buttons"], rl_buttons),
                self._chunk["buttons"] * 0.5,
            ),
        }

        # ── Step 4: Execute all steps of the adjusted chunk ──────────
        n_steps = adj_chunk["j_left"].shape[0]
        for i in range(self._step_idx, n_steps):
            if not self._running:
                break
            self.mapper.execute(
                adj_chunk["j_left"][i],
                adj_chunk["j_right"][i],
                adj_chunk["buttons"][i],
            )
            if i < n_steps - 1:
                time.sleep(_STEP_INTERVAL)

        self._step_idx = n_steps  # mark chunk consumed

        # ── Step 5: Post-action reward measurement ────────────────────
        post_frame = self.capture.get_frame()
        reward_info = self.reward_det.step(post_frame)
        reward = reward_info["reward"]
        done = reward_info["died"]

        # ── Step 6: Store transition in rollout buffer ────────────────
        self.buffer.add(
            obs=obs_np,
            base_action=base_action_np,
            action=rl_action_np,
            log_prob=log_prob_val,
            reward=reward,
            value=value_val,
            done=done,
        )

        self._episode_reward += reward
        self._episode_steps += 1

        # Print periodic status
        if self._episode_steps % 10 == 0:
            print(f"[RLCombat] step={self._episode_steps:4d}  "
                  f"reward={self._episode_reward:+.2f}  "
                  f"hp={reward_info['health_pct']:.0%}  "
                  f"kills={reward_info['kills']}  "
                  f"buffer={self.buffer.size}/{self.buffer.buffer_size}")

        # Screenshot every 50 RL steps for debugging
        if self._episode_steps % 50 == 0:
            self._save_debug(post_frame,
                             f"match_step{self._episode_steps}")

        # ── Step 7: PPO update if buffer full ─────────────────────────
        if self.buffer.full:
            self._run_ppo_update()

    # ==================================================================
    # Match lifecycle
    # ==================================================================
    def _on_enter_match(self):
        """Called when transitioning to IN_MATCH."""
        print("[RLCombat] === Match started ===")
        self._in_match = True
        self._episode_reward = 0.0
        self._episode_steps = 0
        self.reward_det.reset()

    def _on_leave_match(self):
        """Called when transitioning out of IN_MATCH."""
        if not self._in_match:
            return

        self._in_match = False
        self._total_matches += 1
        print(f"[RLCombat] === Match ended ===  "
              f"reward={self._episode_reward:+.2f}  "
              f"steps={self._episode_steps}  "
              f"matches={self._total_matches}")

        self.trainer.log_reward(self._episode_reward, self._episode_steps)

        # Run PPO update with whatever is in the buffer
        if self.buffer.size >= self.trainer.batch_size:
            self._run_ppo_update()

        # Save checkpoint periodically
        if self._total_matches % config.RL_SAVE_EVERY == 0:
            self._save_checkpoint()

    # ==================================================================
    # PPO update
    # ==================================================================
    def _run_ppo_update(self):
        """Run a PPO update cycle on the current rollout buffer."""
        print(f"[RLCombat] Running PPO update  "
              f"(buffer={self.buffer.size} steps) ...")

        # Bootstrap value for the last state
        frame_small = self.capture.get_frame_resized(config.CAPTURE_W,
                                                     config.CAPTURE_H)
        obs_np = frame_small.transpose(2, 0, 1).astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        dummy_base = torch.zeros(1, ACTION_DIM, device=self.device)

        with torch.no_grad():
            _, _, _, last_value = self.policy(obs_t, dummy_base)
        last_val = last_value.item()

        metrics = self.trainer.update(self.buffer, last_val)
        self.buffer.reset()

        if "skipped" not in metrics:
            print(f"[RLCombat] PPO: policy_loss={metrics['policy_loss']:.4f}  "
                  f"value_loss={metrics['value_loss']:.4f}  "
                  f"entropy={metrics['entropy']:.4f}  "
                  f"kl={metrics['approx_kl']:.4f}")

    # ==================================================================
    # Checkpoint
    # ==================================================================
    def _save_checkpoint(self):
        self.trainer.save(self._rl_ckpt_path)

    # ==================================================================
    # Menu navigation (unchanged from CombatLoop)
    # ==================================================================
    def _handle_popup(self):
        now = time.time()
        if now - self._last_escape_press < 2.0:
            return
        print("[RLCombat] POPUP -- pressing Escape to dismiss.")
        pyautogui.press("escape")
        self._last_escape_press = now
        self._reset_actions()

    def _handle_lobby(self, frame):
        now = time.time()
        if now - self._last_battle_click < 5.0:
            return
        pos = self.detector.get_battle_button_pos(frame)
        if pos is not None:
            x, y = pos
            sx, sy = self._frame_to_screen(x, y)
            print(f"[RLCombat] LOBBY -- clicking BATTLE at ({sx}, {sy})")
            _click_at(sx, sy)
            self._last_battle_click = now
        else:
            if now - self._last_escape_press > 3.0:
                print("[RLCombat] LOBBY but no BATTLE found -- pressing Escape")
                pyautogui.press("escape")
                self._last_escape_press = now
        self._reset_actions()

    def _handle_mode_select(self, frame):
        now = time.time()
        if now - self._last_mode_click < 5.0:
            return
        if self._mode_template is not None:
            res = cv2.matchTemplate(
                frame, self._mode_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(res)
            if val > 0.60:
                th, tw = self._mode_template.shape[:2]
                x = loc[0] + tw // 2
                y = loc[1] + th + 80
                sx, sy = self._frame_to_screen(x, y)
                print(f"[RLCombat] MODE SELECT -- clicking at ({sx}, {sy})")
                _click_at(sx, sy)
                self._last_mode_click = now
                return
        self._last_mode_click = now

    def _frame_to_screen(self, x, y):
        region = self.capture._region
        return (region["left"] + x, region["top"] + y)

    def _reset_actions(self):
        self.wrapper.reset()
        while not self.action_q.empty():
            try:
                self.action_q.get_nowait()
            except queue.Empty:
                break
        self._chunk = None
        self._step_idx = 0
