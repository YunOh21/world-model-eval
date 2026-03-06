import argparse
from typing import Sequence

from .utils import (
    rescale_bridge_action,
    discover_trials,
    predict,
    aggregate_model_results,
    print_results_table,
)
from transformers import AutoModelForVision2Seq, AutoProcessor
from .world_model import WorldModel
import numpy as np
from PIL import Image
import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path
import json
from accelerate import Accelerator


def evaluate_openvla(wm, vla, processor, trials, device, retries=1, rollout_length=40,
                     save_video=False, video_out_dir=None, root_dir=None):
    """
    Rollout an OpenVLA model on a list of tasks, and return the score on each task.
    Arguments:
        wm: WorldModel
        vla: An OpenVLA model from `transformers`
        tasks: A list of N tasks in loaded from a json. See "put_carrot_on_plate.json" for an example of the format.
    Returns:
        scores: A list of N scores from the VLM corresponding to each input task.
    """
    results = []
    if save_video and video_out_dir:
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for trial in tqdm(trials, desc="Openvla trials"):
            start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))
            for r in range(retries):
                wm.reset(torch.from_numpy(start_frame).to(device).float() / 255.0)
                frames = [start_frame]
                for step in range(rollout_length):
                    curr_frame = Image.fromarray(frames[-1])
                    prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
                    inputs = processor(prompt, curr_frame).to(
                        device=device, dtype=torch.bfloat16
                    )
                    actions = vla.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
                    a = torch.tensor(actions).to(device)
                    # NOTE: OpenVLA outputs 7-dim actions, while the world model was trained with up to 10-dim actions.
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad with zeros
                    a = rescale_bridge_action(a)
                    for i, x in wm.generate_chunk(a):
                        new_frame = x[0, 0].cpu().numpy()
                        new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
                        frames.append(new_frame)
                rollout_video = np.stack(frames)
                if save_video and video_out_dir:
                    trial_png = Path(trial["trial_png"])
                    target_dir = Path(video_out_dir)
                    if root_dir is not None:
                        try:
                            rel_parent = trial_png.parent.relative_to(Path(root_dir))
                            target_dir = target_dir / rel_parent
                        except ValueError:
                            target_dir = target_dir / trial_png.parent.name
                    else:
                        target_dir = target_dir / trial_png.parent.name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    vid_name = trial_png.stem
                    out_name = f"{vid_name}.mp4"
                    media.write_video(str(target_dir / out_name), rollout_video, fps=20)
                score = predict(rollout_video, trial)
                results.append({
                    "task_key": vid_name,
                    "task_display": trial["instruction"],
                    "score": float(score),
                })
                
                print(f"  [task_key] {vid_name} [Task] {trial['instruction']} | Score: {score}")
                
                with open("results.jsonl", "a") as f:
                    f.write(json.dumps(results[-1]) + "\n")
    return results

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {
        "use_pixel_rope": True,
    },
    "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt": {
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}


def run(
    checkpoint_path: str = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt",
    model_name: str = "openvla-7b",
    root_dir: str | None = None,
    *,
    rollout_length: int = 40,
    retries: int = 1,
    save_video: bool = False,
    video_out_dir: str | None = None,
) -> dict[str, dict[str, float]]:
    """Run the OpenVLA evaluation loop."""
    accelerator = Accelerator()  # 추가
    device = accelerator.device  # "cuda" 대신 사용할 디바이스

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}; download it manually and retry.")
    ckpt_key = ckpt_path.name
    ckpt_kwargs = CHECKPOINTS_TO_KWARGS.get(ckpt_key, {})
    wm = WorldModel(ckpt_path, device=device, **ckpt_kwargs).eval()

    processor = AutoProcessor.from_pretrained(f"openvla/{model_name}", trust_remote_code=True)
    with accelerator.zero_3_init_context():
        vla = AutoModelForVision2Seq.from_pretrained(
            f"openvla/{model_name}",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    
    vla = accelerator.prepare(vla)

    if root_dir is None:
        raise ValueError("root_dir must be provided; pass --root-dir to point at the evaluation dataset.")
    trials = discover_trials(root_dir)
    print(f"Discovered {len(trials)} trials.")

    results = evaluate_openvla(
        wm,
        vla,
        processor,
        trials,
        device,
        rollout_length=rollout_length,
        retries=retries,
        save_video=save_video,
        video_out_dir=video_out_dir,
        root_dir=root_dir,
    )

    agg = aggregate_model_results(results)
    print_results_table(agg)
    return agg


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an OpenVLA policy in the Bridge world model")
    parser.add_argument("--checkpoint-path", default="mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt")
    parser.add_argument("--model-name", default="openvla-7b")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--rollout-length", type=int, default=40)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, dict[str, float]]:  # pragma: no cover - CLI entry point
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return run(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        root_dir=args.root_dir,
        rollout_length=args.rollout_length,
        retries=args.retries,
        save_video=args.save_video,
        video_out_dir=args.video_out_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
