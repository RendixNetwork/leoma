"""Pinned, multi-GPU offline captioning for corpus-v2 samples.

This module is never imported by the validator or eval server.  Caption generation
is an offline corpus-build step: the resulting text is frozen into manifest shards,
so validators do not need the caption model and cannot disagree about its output.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Sequence

from leoma.infra.corpus_v2 import CAPTION_PROMPT, PilotLedger, PilotSpec


_IMMUTABLE_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# Resolved from Hugging Face on 2026-07-23.  Keeping the commit here makes the
# default CLI invocation reproducible; callers can still provide another pinned
# open-source model and immutable revision explicitly.
DEFAULT_CAPTION_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def validate_model_revision(revision: str) -> None:
    """Require a Hugging Face commit hash, never a mutable branch or tag."""
    if not _IMMUTABLE_HF_REVISION.fullmatch(str(revision).strip().lower()):
        raise ValueError(
            "caption model revision must be an immutable 40-character Hugging Face "
            "commit hash (not main, latest, or a mutable tag)"
        )


class TransformersVideoCaptioner:
    """Native temporal-video captioning through a Transformers processor.

    Qwen3-VL receives one video item rather than eight unrelated image items.  Its
    processor samples exactly ``frame_count`` frames while retaining temporal
    positions, which is important for distinguishing actions and camera motion.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        spec: PilotSpec,
        max_new_tokens: int = 96,
        frame_count: int = 8,
    ):
        validate_model_revision(revision)
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:  # compatibility with older pinned transformer stacks
            from transformers import AutoModelForVision2Seq as ModelClass

        self.model_name = model
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.frame_count = int(frame_count)
        self.spec = spec
        self.processor = AutoProcessor.from_pretrained(
            model, revision=revision, trust_remote_code=False,
        )
        self.model = ModelClass.from_pretrained(
            model,
            revision=revision,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        self.model.eval()

    def caption(self, clip_path: str) -> str:
        import torch
        path = Path(clip_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prepared clip is missing: {path}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(path)},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]

        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError(
                "caption model processor has no chat template; choose an instruction-tuned "
                "video model compatible with AutoModelForImageTextToText"
            )
        # This is Qwen3-VL's native video path. Setting num_frames (and clearing
        # fps) prevents a backend/default change from silently changing what the
        # caption model sees. The source clip itself is already pinned to 81 frames.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"num_frames": self.frame_count, "fps": None},
        )
        inputs = inputs.to("cuda:0")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        generated = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        text = text.strip()
        if text.lower().startswith("assistant"):
            text = text[len("assistant"):].lstrip(" :\n")
        return text


def _caption_worker(
    *,
    ledger_path: str,
    gpu: str,
    model: str,
    revision: str,
    frame_count: int,
    max_new_tokens: int,
    max_attempts: int,
    follow: bool,
    poll_seconds: float,
    idle_timeout_seconds: float,
) -> None:
    # Set visibility before importing torch in the captioner constructor.  Each
    # process sees exactly one device and therefore always uses cuda:0 internally.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    worker = f"gpu-{gpu}-pid-{os.getpid()}"
    ledger = PilotLedger(ledger_path)
    spec = ledger.read_spec()
    captioner = TransformersVideoCaptioner(
        model, revision, spec=spec, frame_count=frame_count, max_new_tokens=max_new_tokens,
    )
    empty_since: float | None = None
    while True:
        task = ledger.claim_caption(worker, max_attempts=max_attempts)
        if task is None:
            if not follow:
                return
            if empty_since is None:
                empty_since = time.monotonic()
            if time.monotonic() - empty_since >= idle_timeout_seconds:
                return
            time.sleep(poll_seconds)
            continue
        empty_since = None
        try:
            if not Path(task.clip_path).is_file():
                raise FileNotFoundError(f"prepared clip is missing: {task.clip_path}")
            caption = captioner.caption(task.clip_path)
            ledger.finish_caption(
                task.sample_id,
                caption=caption,
                model=model,
                revision=revision,
                worker=worker,
                frame_count=frame_count,
            )
        except Exception as exc:  # the ledger makes bounded retry explicit and visible
            ledger.fail_caption(task.sample_id, str(exc), max_attempts=max_attempts)


def run_caption_workers(
    ledger_path: str,
    *,
    gpus: Sequence[str | int],
    model: str,
    revision: str,
    frame_count: int = 16,
    max_new_tokens: int = 96,
    max_attempts: int = 3,
    recover: bool = True,
    follow: bool = False,
    poll_seconds: float = 2.0,
    idle_timeout_seconds: float = 600.0,
) -> dict:
    """Run one persistent caption process per physical GPU and return ledger stats.

    With ``follow=True``, workers keep their models loaded while the preparation
    process adds new rows. They exit only after the queue remains empty for the
    configured idle timeout.
    """
    validate_model_revision(revision)
    gpu_ids = [str(g).strip() for g in gpus if str(g).strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("provide one or more unique GPU indices")
    if poll_seconds <= 0 or idle_timeout_seconds <= 0:
        raise ValueError("caption follow timing values must be positive")
    ledger = PilotLedger(ledger_path)
    ledger.bind_caption_spec(
        model=model,
        revision=revision,
        frame_count=frame_count,
        max_new_tokens=max_new_tokens,
    )
    if recover:
        ledger.recover_caption_claims()
    pending = ledger.caption_queue_size(max_attempts=max_attempts)
    if not pending and not follow:
        return ledger.stats()
    # Do not load an expensive model replica on a GPU that cannot receive work.
    if not follow:
        gpu_ids = gpu_ids[:pending]

    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_caption_worker,
            kwargs={
                "ledger_path": ledger_path,
                "gpu": gpu,
                "model": model,
                "revision": revision,
                "frame_count": frame_count,
                "max_new_tokens": max_new_tokens,
                "max_attempts": max_attempts,
                "follow": follow,
                "poll_seconds": float(poll_seconds),
                "idle_timeout_seconds": float(idle_timeout_seconds),
            },
            name=f"leoma-caption-gpu-{gpu}",
        )
        for gpu in gpu_ids
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed_workers = [p.name for p in processes if p.exitcode != 0]
    if failed_workers:
        raise RuntimeError(f"caption workers exited unsuccessfully: {', '.join(failed_workers)}")
    return ledger.stats()


__all__ = [
    "DEFAULT_CAPTION_MODEL",
    "DEFAULT_CAPTION_REVISION",
    "TransformersVideoCaptioner",
    "run_caption_workers",
    "validate_model_revision",
]
