import numpy as np
from PIL import Image
import torch
import base64
import re
import cv2
import os
from pathlib import Path
import json

def rescale_bridge_action(
    a,
    wv_lo=-0.05,
    wv_hi=+0.05,
    wv_post_scale_max=+1.75,
    wv_post_scale_min=-1.75,
    rd_lo=-0.25,
    rd_hi=+0.25,
    rd_post_scale_max=+1.4,
    rd_post_scale_min=-1.4):
    """
    Rescale Bridge (WidowX) action to the ranges expected by the world model.
    We need to call this function on the unnormalized action values returned by the policy.
    """
    # rescale end effector
    a[:3] = (a[:3] - wv_lo) / (wv_hi - wv_lo) * (
        wv_post_scale_max - wv_post_scale_min
    ) + wv_post_scale_min
    a[:3] = torch.clamp(a[:3], wv_post_scale_min, wv_post_scale_max)
    # rescale joint rotations
    a[3:6] = (a[3:6] - rd_lo) / (rd_hi - rd_lo) * (
        rd_post_scale_max - rd_post_scale_min
    ) + rd_post_scale_min
    a[3:6] = torch.clamp(a[3:6], rd_post_scale_min, rd_post_scale_max)
    # threshold the gripper
    a[6] = torch.where(a[6] > 0.8, -1.0, +1.0)
    return a

def encode_video(video, stride=20):
    frames, idx = [], 0
    for idx, frame in enumerate(video):
        if idx % stride == 0:
            if (frame == 0).all():
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buf).decode())
    return frames

def evaluate(scores):
    partial = scores[:, 0]  # First subtask scores
    complete = scores[:, 1]  # Second subtask scores
    print(f"Partial completion mean score: {np.round(100*np.mean(partial))=}")
    print(f"Partial completion STE: {np.round(100*np.std(partial) / len(partial)**0.5)=}")
    print(f"Completion mean score: {np.round(100*np.mean(complete))=}")
    print(f"Completion STE: {np.round(100*np.std(complete) / len(complete)**0.5)=}")

import requests
import os
import cv2
from PIL import Image

def predict(video, trial, n=5):
    instruction = trial["instruction"]
    has_partial = bool(trial.get("partial_criteria"))
    
    tmp_dir = "eval_frames_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    image_paths = []
    stride = 20
    for idx, frame in enumerate(video):
        if idx % stride == 0:
            if (frame == 0).all(): break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p = os.path.abspath(os.path.join(tmp_dir, f"frame_{idx}.jpg"))
            Image.fromarray(frame_rgb).save(p)
            image_paths.append(p)

    rubric = f"""
Score rubric:
0 = Failure: instruction "{instruction}" not completed.
1 = Success: instruction completed."""
    prompt = f"Instruction: {instruction}\n{rubric}\nFinal Score: "

    counts = {"0": 0, "0.5": 0, "1": 0}
    parsed = 0
    
    for i in range(n):
        try:
            response = requests.post(
                "http://localhost:5000/predict",
                json={"prompt": prompt, "image_paths": image_paths},
                timeout=60
            )
            if response.status_code == 200:
                content = response.json()["result"].strip()
                m = pattern.search(content)
                if m:
                    val = m.group(1).strip()
                    if val in ("0.0", "0"): key = "0"
                    elif val in ("1.0", "1"): key = "1"
                    elif val in ("0.5",): key = "0.5"
                    counts[key] += 1
                    parsed += 1
                else:
                    print(f"[Attempt {i}] No match in response: {content[:50]}...")
        except Exception as e:
            print(f"Server request failed: {e}")

    for p in image_paths:
        if os.path.exists(p): os.remove(p)

    if parsed == 0: return 0.0
    
    ordered = ["1", "0.5", "0"] if has_partial else ["1", "0"]
    best_key = max(ordered, key=lambda k: (counts[k], ordered.index(k)*-1))
    score = {"0": 0.0, "0.5": 0.5, "1": 1.0}[best_key]
    
    print(f"Parsed {parsed}/{n}. Counts {counts} -> Final Score: {score}")
    return score

def load_tasks(root):
    for file in os.listdir(root):
        if file.endswith(".png"):
            base = os.path.splitext(file)[0]
            yield os.path.join(root, base)

def _titleize(name: str):
    name = name.replace("--", " ")
    return " ".join(w.capitalize() for w in re.split(r"[_\\-]+", name))

def discover_trials(root_dir):
    root_path = Path(root_dir).resolve()
    trials = []
    for png in root_path.rglob("*.png"):
        task_dir = png.parent
        base = png.stem
        json_same = task_dir / f"{base}.json"
        meta_path = json_same if json_same.exists() else None
        if not meta_path:
            print(f"[WARN] No JSON for {png}")
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"[WARN] Bad JSON {meta_path}: {e}")
            continue
        instruction = meta.get("instruction")
        if not instruction:
            print(f"[WARN] No instruction in {meta_path}")
            continue
        partial = meta.get("partial_credit_criteria")
        task_key = str(task_dir.relative_to(root_path))
        trials.append({
            "trial_png": str(png),
            "instruction": instruction,
            "partial_criteria": partial,
            "task_key": task_key,
            "task_display": _titleize(task_dir.name),
        })
    return trials

def aggregate_model_results(results):
    tasks = {}
    for r in results:
        key = r["task_key"]
        if key not in tasks:
            tasks[key] = {
                "Task": r["task_display"],
                "# Trials": 0,
                "# Successes": 0.0,
            }
        tasks[key]["# Trials"] += 1
        tasks[key]["# Successes"] += float(r["score"])

    task_list = sorted(tasks.values(), key=lambda x: x["Task"])

    per_trial_scores = []
    for t in task_list:
        trials = t["# Trials"]
        succ = t["# Successes"]
        if trials > 0:
            per_trial_scores.extend([succ / trials] * trials)
    per_trial_scores = np.array(per_trial_scores)
    N = len(per_trial_scores)
    mean_rate = 100.0 * per_trial_scores.mean() if N else 0.0
    ste = 100.0 * (per_trial_scores.std() / np.sqrt(N)) if N else 0.0
    return {"tasks": task_list, "mean_success_rate": mean_rate, "ste": ste}

def print_results_table(agg):
    tasks = agg["tasks"]
    header = ["Task", "# Trials", "# Successes"]
    col_w = [len(h) for h in header]
    for t in tasks:
        col_w[0] = max(col_w[0], len(t["Task"]))
        col_w[1] = max(col_w[1], len(str(t["# Trials"])))
        col_w[2] = max(col_w[2], len(str(t["# Successes"])))
    fmt = lambda row: " | ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row))
    print(fmt(header))
    print("-" * (sum(col_w) + 3 * (len(header) - 1)))
    for t in tasks:
        print(fmt([
            t["Task"],
            t["# Trials"],
            t["# Successes"]
        ]))
    print("-" * (sum(col_w) + 3 * (len(header) - 1)))
    print(f"Mean Success Rate: {agg['mean_success_rate']:.1f}±{agg['ste']:.1f}%")