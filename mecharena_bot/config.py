import ctypes
import os
import torch

# Set DPI awareness early so window coords are correct on multi-monitor setups
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except Exception:
    pass

# Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "ng.pt")
NITROGEN_PATH   = "C:/AI/NitroGen"          # cloned NitroGen repo

# Game window — bot targets the Mech Arena window once launched
GAME_WINDOW_TITLE = "Mech Arena"            # window title to search for
LAUNCHER_WINDOW_TITLE = "Plarium Play"      # launcher window title

# Inference
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CFG_SCALE       = 1.0                       # 1.0 = no CFG guidance (faster)
CONTEXT_LENGTH  = 1                         # single-frame context

# Action execution
ACTION_FPS      = 15                        # how fast to replay the 16-step action chunk
JOYSTICK_DEAD   = 0.30                      # joystick deflection threshold for key presses
MOUSE_SCALE     = 10.0                      # pixels per unit of joystick deflection per action step
BUTTON_THRESH   = 0.5                       # button confidence threshold

# Screen capture
CAPTURE_W       = 256
CAPTURE_H       = 256

# ── Reinforcement Learning ──────────────────────────────────────────────
# Reward weights
RL_W_HEALTH     = 1.0           # reward per unit of health delta
RL_W_KILL       = 5.0           # reward per kill
RL_W_ALIVE      = 0.01          # small per-step survival bonus
RL_W_DEATH      = -5.0          # penalty on death

# PPO hyper-parameters
RL_LR           = 3e-4          # Adam learning rate
RL_GAMMA        = 0.99          # discount factor
RL_GAE_LAMBDA   = 0.95          # GAE lambda
RL_CLIP_EPS     = 0.2           # PPO clipping epsilon
RL_N_EPOCHS     = 4             # PPO epochs per update
RL_BATCH_SIZE   = 64            # mini-batch size
RL_ENTROPY_COEF = 0.01          # entropy bonus coefficient
RL_ROLLOUT_STEPS = 256          # steps per rollout before PPO update
RL_SAVE_EVERY   = 3             # save checkpoint every N matches
