import numpy as np
import torch
import time
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
# UR5e (MuJoCo Menagerie): 6 hinge joints, 6 position actuators.
JOINTS = 6
STATE = 2 * JOINTS   # [q, qd]
ACTION = 6           # q* (joint-position targets)

HIDDEN = 256
EPOCHS = 40
BATCH = 512
LR = 1e-3
ROLLOUT_HORIZON = 10      # steps for the multi-step (open-loop) loss
MULTI_STEP_WEIGHT = 0.5
VAL_FRACTION = 0.1
SEED = 0


class Dynamics(nn.Module):
    """MLP world model: normalized [s, a] -> normalized delta s.

    The UR5e actuators are position servos (torque = kp (q* - q) - kv qd), so the most
    informative input is the servo error q* - q. That is a small difference of two large
    numbers, which a plain MLP learns poorly, so we compute it explicitly inside the
    network (from the de-normalized input) and append it as extra features. The
    normalization constants live in buffers, so the exported ONNX graph still takes the
    same normalized [s, a] vector and Unity does not need to know about this trick.
    """

    def __init__(self, state=STATE, action=ACTION, hidden=HIDDEN):
        super().__init__()
        self.state, self.action = state, action
        self.register_buffer("x_mean", torch.zeros(state + action))
        self.register_buffer("x_std", torch.ones(state + action))
        self.register_buffer("err_std", torch.ones(action))
        self.net = nn.Sequential(
            nn.Linear(state + 2 * action, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state),
        )

    def set_norm(self, x_mean, x_std, err_std):
        self.x_mean.copy_(torch.as_tensor(x_mean))
        self.x_std.copy_(torch.as_tensor(x_std))
        self.err_std.copy_(torch.as_tensor(err_std))

    def forward(self, x):
        raw = x * self.x_std + self.x_mean
        q = raw[:, :self.action]
        q_star = raw[:, self.state:self.state + self.action]
        servo_error = (q_star - q) / self.err_std
        return self.net(torch.cat([x, servo_error], dim=-1))


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / "norm.json").write_text("\n".join(lines) + "\n")


def load_rollout():
    path = ROOT / "rollout.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])


def multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    """Open-loop rollout loss: feed the model its OWN predictions for `horizon` steps
    (with the recorded actions) and penalize drift from the recorded trajectory.

    data   : torch tensor (N, STATE + ACTION + STATE), rows are consecutive steps
    split  : numpy array of row indices this loss may start from
    The error is measured in normalized-state units (divide by the state std) so the
    joint angles and velocities contribute on a similar scale.
    """
    valid = split[split < len(data) - horizon]
    starts = torch.as_tensor(np.random.choice(valid, size=min(batch, len(valid)), replace=False))
    s_scale = x_std[:STATE]

    state = data[starts, 0:STATE]
    total = 0.0
    for t in range(horizon):
        rows = data[starts + t]
        action = rows[:, STATE:STATE + ACTION]
        x_n = (torch.cat([state, action], dim=-1) - x_mean) / x_std
        delta = model(x_n) * y_std + y_mean
        state = state + delta
        true_next = rows[:, STATE + ACTION:]
        total = total + loss_fn(state / s_scale, true_next / s_scale)
    return total / horizon


def train_world_model(data):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    s = data[:, 0:STATE]
    a = data[:, STATE:STATE + ACTION]
    s_next = data[:, STATE + ACTION:]
    x = np.concatenate([s, a], axis=1)
    y = s_next - s                       # predict the change in state

    # Chronological split: the last 10% of the recording is held out.
    n = len(data)
    n_val = int(n * VAL_FRACTION)
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    x_mean, x_std = mean_std(x[train_idx])
    y_mean, y_std = mean_std(y[train_idx])
    ROOT.mkdir(parents=True, exist_ok=True)
    save_norm_json(x_mean, x_std, y_mean, y_std)
    np.savez(ROOT / "norm.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    xt = torch.as_tensor(norm(x, x_mean, x_std))
    yt = torch.as_tensor(norm(y, y_mean, y_std))
    data_t = torch.as_tensor(data)
    xm, xs, ym, ys = (torch.as_tensor(v) for v in (x_mean, x_std, y_mean, y_std))

    model = Dynamics()
    err_std = (a[train_idx] - s[train_idx, :ACTION]).std(0) + 1e-6
    model.set_norm(x_mean, x_std, err_std)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps_per_epoch = max(1, len(train_idx) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    mse = nn.MSELoss()

    print(f"{n} transitions  ({len(train_idx)} train / {len(val_idx)} val), "
          f"model {sum(p.numel() for p in model.parameters())} params")

    best_val, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(train_idx)
        one_total, multi_total = 0.0, 0.0
        for k in range(steps_per_epoch):
            idx = torch.as_tensor(perm[k * BATCH:(k + 1) * BATCH])
            one = mse(model(xt[idx]), yt[idx])
            multi = multi_step_loss(model, data_t, train_idx, xm, xs, ym, ys,
                                    ROLLOUT_HORIZON, BATCH // 4, mse)
            loss = one + MULTI_STEP_WEIGHT * multi
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            one_total += one.item()
            multi_total += multi.item()

        model.eval()
        with torch.no_grad():
            vidx = torch.as_tensor(val_idx)
            val_one = mse(model(xt[vidx]), yt[vidx]).item()
            val_multi = multi_step_loss(model, data_t, val_idx, xm, xs, ym, ys,
                                        ROLLOUT_HORIZON, 1024, mse).item()
        val = val_one + MULTI_STEP_WEIGHT * val_multi
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:3d}  train 1-step {one_total / steps_per_epoch:.5f}  "
              f"{ROLLOUT_HORIZON}-step {multi_total / steps_per_epoch:.5f} | "
              f"val 1-step {val_one:.5f}  {ROLLOUT_HORIZON}-step {val_multi:.5f}")

    model.load_state_dict(best_state)
    model.eval()

    # ---- Report metrics (joint-angle RMSE in radians on the held-out data) ----
    val_start = int(val_idx[0])
    pred_next = data[val_idx, 0:STATE] + predict_delta(
        model, data[val_idx, 0:STATE], data[val_idx, STATE:STATE + ACTION], x_mean, x_std, y_mean, y_std)
    one_rmse = joint_rmse(pred_next, data[val_idx, STATE + ACTION:])
    baseline = joint_rmse(data[val_idx, 0:STATE], data[val_idx, STATE + ACTION:])
    print()
    print(f"One-step joint RMSE: {one_rmse:.5f} rad  (copy-last-state baseline {baseline:.5f})")
    for h in (10, 50, 100):
        r = open_loop_rmse(model, data, val_start, x_mean, x_std, y_mean, y_std, h)
        print(f"Open-loop {h:3d}-step joint RMSE: {r:.5f} rad")

    # ---- Save: PyTorch weights (for train_controller.py) + ONNX (for Unity) ----
    torch.save(model.state_dict(), ROOT / "dynamics.pt")
    dummy = torch.zeros(1, STATE + ACTION)
    torch.onnx.export(model, dummy, ROOT / "dynamics.onnx",
                      input_names=["input"], output_names=["delta"],
                      opset_version=15, dynamo=False)
    print(f"wrote {ROOT / 'dynamics.pt'}, {ROOT / 'dynamics.onnx'}, norm.json, norm.npz")
    return model


if __name__ == "__main__":
    current_time = time.time()
    train_world_model(load_rollout())
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
