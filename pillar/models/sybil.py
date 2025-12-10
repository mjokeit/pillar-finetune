import os
import warnings
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd

from argparse import Namespace
from pillar.models.multi_stage import MultiStage

import rve
from rve import Config, get_processor, SeriesInfo

from tqdm import tqdm
from pillar.metrics.survival import SurvivalMetric
from easydict import EasyDict

class Sybil15:
    def __init__(self,
        model_repo_id="YalaLab/Pillar0-Sybil-1.5",
        model_revision="main",
        local_dir="logs/checkpoints",
        **kwargs
    ):
        repo_id = kwargs.pop("model_repo_id", "YalaLab/Pillar0-Sybil-1.5")
        revision = kwargs.pop("model_revision", "main")
        local_dir = kwargs.pop("local_dir", "logs/checkpoints")
        # Keep remaining kwargs to build the underlying model architecture
        self._base_model_kwargs = dict(kwargs)

        self.ckpt_paths = []
        self.checkpoints = []
        self.ensemble_state_dicts = {}
        self.ensemble_models = nn.ModuleList()

        # Try to download the checkpoints from Hugging Face to the expected local directory
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            warnings.warn(
                f"Could not download checkpoints from {repo_id}@{revision}: {e}. "
                f"Proceeding assuming they already exist at {local_dir}.",
                stacklevel=1,
            )

        # Discover up to three Sybil checkpoints; prefer seed0/1/2 epoch ckpts if present
        candidate_paths = []
        for seed_name in ["seed0", "seed1", "seed2"]:
            seed_dir = os.path.join(local_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            ckpts = [f for f in os.listdir(seed_dir) if f.endswith(".ckpt")]
            if len(ckpts) == 0:
                continue
            # Prefer epoch=2.ckpt if present to mirror test scripts; otherwise choose the last lexicographically
            preferred = "epoch=2.ckpt" if "epoch=2.ckpt" in ckpts else sorted(ckpts)[-1]
            candidate_paths.append(os.path.join(seed_dir, preferred))

        self.ckpt_paths = candidate_paths

        # Load checkpoints into CPU memory, mirroring scripts/train.py loader semantics
        for path in self.ckpt_paths:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older torch versions may not support weights_only
                ckpt = torch.load(path, map_location="cpu")
            self.checkpoints.append(ckpt)
            if isinstance(ckpt, dict) and "model" in ckpt:
                self.ensemble_state_dicts[path] = ckpt["model"]

        # Build three model instances with identical architecture and load the seed weights
        # We reuse the MultiStage architecture kwargs provided to Sybil15
        try:
            print()
            for path in self.ensemble_state_dicts:
                state_dict = self.ensemble_state_dicts[path]
                print(f"Loading checkpoint from {path}")
                model_i = MultiStage(
                    args=Namespace(),
                    pool_name="MultiAttentionPool",
                    pool_kwargs={"hidden_dim": 1152},
                    backbone_model_type="MultimodalAtlas",
                    backbone_kwargs={
                        "pretrained": True,
                        "model_repo_id": "YalaLab/Pillar0-ChestCT",
                        "model_revision": "main",
                        "device": "cpu",
                    },
                    head_models={
                        "survival": {
                            "type": "CumulativeProbabilityLayer",
                            "kwargs": {"input_dim": 1152, "max_followup": 6},
                            "apply_pooling": True,
                            "use_pooled_features": False,
                            "enable_at_eval": True,
                        },
                        "detr": {
                            "type": "DETR3D",
                            "use_pooled_features": False,
                            "apply_pooling": False,
                            "kwargs": {
                                "input_dim": 1152,
                                "num_classes": 1,
                                "num_queries": 1,
                                "transformer_kwargs": {
                                    "d_model": 128,
                                    "dropout": 0.1,
                                    "nhead": 1,
                                    "dim_feedforward": 128,
                                    "num_encoder_layers": 2,
                                    "num_decoder_layers": 2,
                                    "normalize_before": False,
                                },
                            },
                            "enable_at_eval": False,
                        },
                    },
                )
                model_i.load_state_dict(state_dict)
                model_i.eval()
                self.ensemble_models.append(model_i)
                print()
        except Exception as e:
            warnings.warn(f"Failed to build ensemble models for Sybil15: {e}", stacklevel=1)


        self.ct_processor = get_processor("CT", config=Config.from_yaml("configs/rve/ct_chest.yaml"))


    def predict(self, inputs_csv_path=None, rve_sample=None, **extras):
        scores = {"accession": [], "logit": [], "y": [], "time_at_event": []}
        if rve_sample is not None:
            if isinstance(rve_sample, str):
                rve_sample = [rve_sample]
            inputs = [{"output_path": x} for x in rve_sample]
            progress_bar = tqdm(inputs, desc="Generating Cancer Risk Scores")
        elif isinstance(inputs_csv_path, str):
            inputs = pd.read_csv(inputs_csv_path)
            os.system(f"vision-engine process --config configs/rve/ct_chest.yaml --input-series-csv {inputs_csv_path} --output rve-output --workers 4")
            processed = pd.read_csv("rve-output/mapping.csv")
            inputs = inputs.merge(processed, left_on="series_path", right_on="source_path")
            progress_bar = tqdm(inputs.iterrows(), total=len(inputs), desc="Generating Cancer Risk Scores")

        batch = {"anatomy": ["chest_ct"]}

        for row in progress_bar:
            if len(row) == 2:
                row = row[1]
            scores["accession"].append(row.get('accession', None))
            scores["y"].append(row.get('y', None))
            scores["time_at_event"].append(row.get('time_at_event', None))

            processed_series = rve.load_sample(row['output_path'],  use_hardware_acceleration=False)

            D, H, W = processed_series.shape
            if H > 256:
                crop_side = (H - 256) // 2
                processed_series = processed_series[:, crop_side:-crop_side, crop_side:-crop_side]
            if D < 256:
                pad_total = 256 - D
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left  # Handles odd padding amounts
                processed_series = F.pad(processed_series, (0, 0, 0, 0, pad_left, pad_right))

            x = rve.apply_windowing(processed_series, "all", "CT").unsqueeze(0)
            logits = []
            for model_i in self.ensemble_models:
                with torch.no_grad():
                    out = model_i.forward(x, batch=batch, split="test", **extras)['survival'].cpu()
                    logits.append(out.squeeze(0))

            scores["logit"].append(torch.stack(logits).mean(dim=0))
        return scores

    def evaluate(self, inputs_csv_path=None, rve_sample=None, **extras):
        scores = self.predict(inputs_csv_path=inputs_csv_path, rve_sample=rve_sample, **extras)
        print(scores)
        metric = SurvivalMetric(args=EasyDict({"dataset": {"shared_dataset_kwargs": {"max_followup": 6}}}), dataset_info=None, split="test")
        metric_scores = metric(
            logit=torch.stack(scores['logit']),
            y=torch.tensor([int(v) for v in scores['y']], dtype=torch.long),
            time_at_event=torch.tensor([int(v) for v in scores['time_at_event']], dtype=torch.long)
        )
        for k, v in metric_scores.items():
            print(f"{k}: {v}")
        return scores



