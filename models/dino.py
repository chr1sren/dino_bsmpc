import sys

import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

# Latest dinov2 main uses PEP604 unions (float | None), which need Python 3.10+.
# Pin a known-good commit that still works on Python 3.9.
_DINOV2_REPO = "facebookresearch/dinov2"
_DINOV2_PY39_REF = "b48308ae6c1e15e08751675effe31ae290af8cd0"


def _load_dinov2(name: str):
    if sys.version_info < (3, 10):
        repo = f"{_DINOV2_REPO}:{_DINOV2_PY39_REF}"
        print(f"[DinoV2Encoder] Python {sys.version_info.major}.{sys.version_info.minor}: "
              f"loading dinov2 from {repo}")
        return torch.hub.load(repo, name, trust_repo=True, force_reload=False)
    return torch.hub.load(_DINOV2_REPO, name, trust_repo=True)


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = _load_dinov2(name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
