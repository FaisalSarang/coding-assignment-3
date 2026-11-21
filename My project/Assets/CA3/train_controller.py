import json
from pathlib import Path
import time
import cma
import numpy as np
import torch
from torch import nn

from train import Dynamics

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 12          # UR5e: [q (6), qd (6)]
ACTION = 6
FEAT = STATE
N_PARAMS = ACTION * FEAT + ACTION

# From ur5e.xml: <key name="home"> and the joint ranges (elbow is +-pi, the rest +-2pi).
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)
JOINT_LO = np.array([-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319], dtype=np.float32)
JOINT_HI = -JOINT_LO
RANDOM_SCALE = 0.15   # keep goals in the collision-free region around home (see REPORT)

# Tuned for the UR5e (see REPORT): with VEL_PENALTY = 0.05 and MAX_DELTA = 0.05 the
# rate limit, not the policy, decides how fast the arm moves, and slow motion always
# wins, so CMA-ES could not beat open-loop tracking. A lighter velocity penalty and a
# looser rate limit leave room for the policy to learn lead / damping gains.
VEL_PENALTY = 0.01
ACTION_SMOOTH_PENALTY = 0.05
ACTION_SMOOTH = 0.3
MAX_DELTA = 0.1
HORIZON = 200       # 200 control steps x 0.02 s = 4 s per imagined episode
EPISODES = 64       # simulated as one batch, so this is cheap
CMA_ITERS = 60
CMA_SIGMA = 0.1
SEED = 0
# "diagonal": CMA-ES searches 3 numbers per joint (position gain, velocity gain, bias)
#             = 18 parameters, expanded to the full W (6x12) and b for saving.
# "full":     CMA-ES searches all 78 entries of W and b directly. Starting from W = 0 the
#             random cross-joint terms hurt far more than the useful diagonal gains help,
#             and in testing it never beat open-loop tracking.
# "gains":    like "diagonal" but with b fixed at 0 (12 parameters). This is the default:
#             with a free bias, CMA-ES learned a constant -0.03..-0.04 rad offset on the
#             shoulder/elbow that exploited a small steady-state error of the world model.
#             In Unity that left the hand ~3 cm off every IK target.
STRUCTURE = "gains"


def build_dynamics():
    model = Dynamics(STATE, ACTION)
    path = ROOT / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / "norm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
    b = theta[ACTION * FEAT :]
    return w, b


def expand(z):
    """Search vector -> full theta (W flattened row-major, then b)."""
    if STRUCTURE == "full":
        return np.asarray(z, dtype=np.float64)
    kp, kd = z[:ACTION], z[ACTION:2 * ACTION]
    bias = z[2 * ACTION:] if STRUCTURE == "diagonal" else np.zeros(ACTION)
    w = np.zeros((ACTION, FEAT))
    w[:, :ACTION] = np.diag(kp)            # gain on (q - goal) of the same joint
    w[:, ACTION:] = np.diag(kd)            # gain on qd of the same joint
    return np.concatenate([w.reshape(-1), bias])


def search_size():
    return {"full": N_PARAMS, "diagonal": 3 * ACTION, "gains": 2 * ACTION}[STRUCTURE]


def policy(state, goal, w, b):
    """Controller C: residual linear policy on f = [q - goal, qd].
    raw = clip(goal + W f + b). Works on a single state (S,) or a batch (B, S).
    W = b = 0 is exactly open-loop tracking, so CMA-ES starts from a sane controller
    and only has to learn corrections (damping, gravity-sag bias, anticipation)."""
    q = state[..., :ACTION]
    qd = state[..., ACTION:STATE]
    f = np.concatenate([q - goal, qd], axis=-1)
    raw = goal + f @ w.T + b
    return np.clip(raw, JOINT_LO, JOINT_HI).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    """One imagined step: s' = s + denorm(model(norm([s, a])))."""
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    delta_n = model(torch.as_tensor(x_n, dtype=torch.float32)).numpy()
    return (state + delta_n * y_std + y_mean).astype(np.float32)


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    goal = HOME + rng.uniform(-span, span)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, episodes=1):
    """Run `episodes` imagined episodes in parallel inside the world model and return
    their mean return. Each episode: start at rest at a random pose, chase a random goal
    (same distribution as data collection) for HORIZON steps.
      r_t = -||q - goal||^2 - VEL_PENALTY ||qd||^2 - ACTION_SMOOTH_PENALTY ||a_t - a_{t-1}||^2
    """
    w, b = unpack(theta)
    start_q = np.stack([sample_goal(rng) for _ in range(episodes)])
    goal = np.stack([sample_goal(rng) for _ in range(episodes)])
    state = np.concatenate([start_q, np.zeros_like(start_q)], axis=1).astype(np.float32)
    action = start_q.copy()          # the command starts where the arm is (as in Unity)

    total = np.zeros(episodes, dtype=np.float64)
    for _ in range(HORIZON):
        raw = policy(state, goal, w, b)
        new_action = smooth_action(action, raw)
        state = step_dynamics(model, state, new_action, x_mean, x_std, y_mean, y_std)
        q, qd = state[:, :ACTION], state[:, ACTION:]
        total -= ((q - goal) ** 2).sum(1)
        total -= VEL_PENALTY * (qd ** 2).sum(1)
        total -= ACTION_SMOOTH_PENALTY * ((new_action - action) ** 2).sum(1)
        action = new_action
    total /= HORIZON
    return float(total.mean())


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    # All EPISODES are simulated as one batch; the same seed is used for every
    # candidate of a generation (common random numbers -> fair comparison).
    rng = np.random.default_rng(seed)
    return -rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, episodes=EPISODES)


def save_controller(theta):
    w, b = unpack(theta)
    payload = {
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
        "action_smooth": ACTION_SMOOTH,
        "max_delta": MAX_DELTA,
    }
    path = ROOT / "controller.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = build_dynamics()
    x_mean, x_std, y_mean, y_std = (v.astype(np.float32) for v in load_norm())

    z0 = np.zeros(search_size())                    # theta = 0  ->  open-loop tracking
    eval_seed = 12345                               # fixed held-out goals for reporting
    base = -fitness(expand(z0), model, x_mean, x_std, y_mean, y_std, eval_seed)
    print(f"open-loop baseline (theta = 0) imagined return: {base:.4f}   "
          f"[searching {search_size()} params, structure={STRUCTURE}]")

    es = cma.CMAEvolutionStrategy(z0, CMA_SIGMA, {"seed": SEED + 1, "verbose": -9})
    best_theta, best_f = expand(z0), -base
    for it in range(CMA_ITERS):
        candidates = es.ask()
        seed = SEED * 100000 + it % 5               # rotate through 5 fixed goal sets
        scores = [fitness(expand(c), model, x_mean, x_std, y_mean, y_std, seed) for c in candidates]
        es.tell(candidates, scores)
        mean_theta = expand(es.result.xfavorite)
        held_out = fitness(mean_theta, model, x_mean, x_std, y_mean, y_std, eval_seed)
        if held_out < best_f:
            best_f, best_theta = held_out, mean_theta.copy()
        print(f"iter {it + 1:3d}  pop best {-min(scores):.4f}  pop mean {-np.mean(scores):.4f}  "
              f"held-out(mean theta) {-held_out:.4f}  sigma {es.sigma:.4f}")

    print()
    print(f"imagined return: open-loop {base:.4f} -> learned {-best_f:.4f}")
    save_controller(best_theta.astype(np.float32))


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
