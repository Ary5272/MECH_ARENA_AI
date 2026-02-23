"""
MechArena Bot — entry point.

Usage
─────
  python main.py              # run the full bot (frozen NitroGen, no learning)
  python main.py --train      # run with RL training loop (PPO on top of NitroGen)
  python main.py --dry-run    # load model + capture + inference test, no inputs sent
  python main.py --test       # full pipeline: launch game → capture → detect → infer
  python main.py --launch     # just launch Plarium Play and click Play for Mech Arena
"""

import argparse
import sys
import time
import os

import cv2
import numpy as np

import config


def _check_model():
    if not os.path.exists(config.MODEL_PATH):
        print(f"[Main] Model not found at {config.MODEL_PATH}")
        print("[Main] Download it with:")
        print("       python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download('nvidia/NitroGen', 'ng.pt', local_dir='models')\"")
        sys.exit(1)


def run_bot():
    _check_model()

    from launcher_agent.launcher import PlariumLauncher
    from models.nitrogen_wrapper import NitroGenWrapper
    from combat_agent.combat_loop import CombatLoop

    # 1. Launch game
    print("[Main] Launching game ...")
    PlariumLauncher().launch_and_play()

    # 2. Load model and start combat loop
    print(f"[Main] Device: {config.DEVICE}")
    wrapper = NitroGenWrapper(config.MODEL_PATH)

    loop = CombatLoop(wrapper)
    loop.run()


def run_train():
    """Run with RL training — PPO on top of frozen NitroGen."""
    _check_model()

    from launcher_agent.launcher import PlariumLauncher
    from models.nitrogen_wrapper import NitroGenWrapper
    from rl.rl_combat_loop import RLCombatLoop

    # 1. Launch game
    print("[Main] Launching game ...")
    PlariumLauncher().launch_and_play()

    # 2. Load NitroGen + start RL combat loop
    print(f"[Main] Device: {config.DEVICE}")
    print("[Main] Mode: RL TRAINING (PPO + NitroGen)")
    wrapper = NitroGenWrapper(config.MODEL_PATH)

    loop = RLCombatLoop(wrapper)
    loop.run()


def run_dry():
    """Load model, grab one frame, run one inference, print results — no inputs sent."""
    _check_model()

    from models.nitrogen_wrapper import NitroGenWrapper
    from capture.screen_capture import ScreenCapture
    from state_detection.state_detector import StateDetector

    print("[DryRun] Loading model ...")
    wrapper = NitroGenWrapper(config.MODEL_PATH)

    print("[DryRun] Capturing frame ...")
    cap = ScreenCapture()
    frame = cap.get_frame_resized(config.CAPTURE_W, config.CAPTURE_H)
    cv2.imwrite("dry_run_frame.png", frame)
    print(f"[DryRun] Frame saved to dry_run_frame.png  shape={frame.shape}")

    det = StateDetector()
    state = det.detect(frame)
    print(f"[DryRun] Detected state: {state}")

    print("[DryRun] Running inference (may take a while on CPU) ...")
    t0 = time.time()
    result = wrapper.predict(frame)
    elapsed = time.time() - t0

    print(f"[DryRun] Inference time : {elapsed:.2f}s")
    print(f"[DryRun] j_left  shape  : {result['j_left'].shape}   "
          f"sample={result['j_left'][0]}")
    print(f"[DryRun] j_right shape  : {result['j_right'].shape}   "
          f"sample={result['j_right'][0]}")
    print(f"[DryRun] buttons shape  : {result['buttons'].shape}   "
          f"sample={result['buttons'][0]}")
    print("[DryRun] Done — no inputs were sent to the game.")


def run_launch():
    """Just launch Plarium Play and click Play for Mech Arena."""
    from launcher_agent.launcher import PlariumLauncher
    launcher = PlariumLauncher()
    launcher.launch_and_play()


def run_test(skip_launch=False):
    """
    Debug loop: launch -> find window -> capture/detect/print every frame.
    No inputs are sent.  Press Ctrl-C to stop.
    """
    import signal
    import pygetwindow as gw
    from capture.screen_capture import ScreenCapture
    from state_detection.state_detector import StateDetector

    # ── Step 1: launch ────────────────────────────────────────────────
    if not skip_launch:
        print("\n=== STEP 1: Launch ===")
        from launcher_agent.launcher import PlariumLauncher
        PlariumLauncher().launch_and_play()
    else:
        print("\n=== STEP 1: Skipped ===")

    # ── Step 2: find the game window ───────────────────────────────────
    # Since Mech Arena runs fullscreen, we capture the primary monitor.
    # We still try to find the exact window for logging purposes.
    print("\n=== STEP 2: Finding game window ===")
    found_win = None
    for title in ["Mech Arena", "Plarium Play"]:
        wins = gw.getWindowsWithTitle(title)
        # Exact match only — avoid partial matches (e.g. terminal titles)
        exact = [w for w in wins if w.title.strip() == title and w.width > 0]
        if exact:
            found_win = exact[0]
            print(f"  Found window: '{title}'  {found_win.width}x{found_win.height}"
                  f" @ ({found_win.left},{found_win.top})")
            break

    if found_win is None:
        print("  No exact game window found -- using primary monitor.")
        capture_title = "__PRIMARY_MONITOR__"
    else:
        capture_title = found_win.title.strip()
        # Bring game to foreground so mss captures it, not whatever is on top
        try:
            found_win.activate()
            time.sleep(0.5)
            print(f"  Activated '{capture_title}' window.")
        except Exception as e:
            print(f"  Could not activate window: {e}")

    # ── Step 3: debug capture loop ────────────────────────────────────
    print("\n=== STEP 3: Capture/detect loop (Ctrl-C to stop) ===")
    print("  Each line: frame# | raw size | state | brightness | health-roi orange px")
    print("-" * 70)

    cap = ScreenCapture(window_title=capture_title)
    det = StateDetector()

    running = True
    def _stop(sig, _frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    frame_n    = 0
    last_state = None

    while running:
        raw      = cap.get_frame()
        frame256 = cap.get_frame_resized(config.CAPTURE_W, config.CAPTURE_H)
        state    = det.detect(raw)   # full-res for template matching
        brightness = float(cv2.cvtColor(frame256, cv2.COLOR_BGR2GRAY).mean())

        # Orange pixels in bottom-left quarter (health bar indicator)
        roi = frame256[192:, :64]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        orange_px = int(cv2.inRange(hsv,
            np.array([0,  100, 100]),
            np.array([25, 255, 255])).sum())

        # Print every frame for first 10, then on change or every 10
        if frame_n < 10 or state != last_state or frame_n % 10 == 0:
            print(f"  [{frame_n:05d}] "
                  f"raw={raw.shape[1]}x{raw.shape[0]}  "
                  f"state={state:<10}  "
                  f"bright={brightness:5.1f}  "
                  f"orange={orange_px:4d}px")
            last_state = state

        # Save snapshot every 30 frames
        if frame_n % 30 == 0:
            snap = f"assets/debug_{frame_n:05d}.png"
            cv2.imwrite(snap, raw)
            print(f"  [Snapshot -> {snap}]")

        frame_n += 1
        time.sleep(0.1)

    print(f"\n[Main] Stopped after {frame_n} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MechArena AI Bot")
    parser.add_argument("--train",       action="store_true",
                        help="Run with RL training (PPO on top of NitroGen)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Capture + inference test, no inputs sent (game must already be open)")
    parser.add_argument("--launch",      action="store_true",
                        help="Launch Plarium Play and click Play for Mech Arena")
    parser.add_argument("--test",        action="store_true",
                        help="Debug loop: launch → wait for window → print capture/state each frame")
    parser.add_argument("--skip-launch", action="store_true",
                        help="Use with --test to skip launching (game already open)")
    args = parser.parse_args()

    if args.train:
        run_train()
    elif args.dry_run:
        run_dry()
    elif args.launch:
        run_launch()
    elif args.test:
        run_test(skip_launch=args.skip_launch)
    else:
        run_bot()
