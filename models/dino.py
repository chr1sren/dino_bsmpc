import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

_DINOV2_REPO = "facebookresearch/dinov2"
# Known Python-3.9-friendly commit (short: b48308a)
_DINOV2_PY39_REF = "b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"


def _patch_pep604_unions(root: Path) -> int:
    """Rewrite common `X | Y` annotations for Python 3.9 importability."""
    union_pat = re.compile(
        r"(?P<pre>[:,\[(=\s])"
        r"(?P<a>[A-Za-z_][\w\.]*)"
        r"\s*\|\s*"
        r"(?P<b>None|[A-Za-z_][\w\.]*)"
        r"(?P<post>[\s,)=\]])"
    )

    def repl(m):
        a, b = m.group("a"), m.group("b")
        if b == "None":
            t = f"Optional[{a}]"
        elif a == "None":
            t = f"Optional[{b}]"
        else:
            t = f"Union[{a}, {b}]"
        return f"{m.group('pre')}{t}{m.group('post')}"

    touched = 0
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if " | " not in text:
            continue
        new_text, n = union_pat.subn(repl, text)
        if n == 0:
            continue
        needs_optional = "Optional[" in new_text
        needs_union = "Union[" in new_text
        if needs_optional or needs_union:
            if "from typing import" not in new_text:
                imports = []
                if needs_optional:
                    imports.append("Optional")
                if needs_union:
                    imports.append("Union")
                new_text = f"from typing import {', '.join(imports)}\n" + new_text
            else:
                def _extend_typing(match):
                    names = [x.strip() for x in match.group(1).split(",")]
                    if needs_optional and "Optional" not in names:
                        names.append("Optional")
                    if needs_union and "Union" not in names:
                        names.append("Union")
                    return "from typing import " + ", ".join(names)

                new_text = re.sub(
                    r"from typing import ([^\n]+)",
                    _extend_typing,
                    new_text,
                    count=1,
                )
        path.write_text(new_text, encoding="utf-8")
        touched += 1
    return touched


def _find_hub_dinov2_dir():
    hub_dir = Path(torch.hub.get_dir())
    cands = sorted(
        hub_dir.glob("facebookresearch_dinov2*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _download_hub_source(ref: str = "main"):
    """Ensure github source exists in hub cache (ignore model build errors)."""
    repo = _DINOV2_REPO if ref == "main" else f"{_DINOV2_REPO}:{ref}"
    try:
        torch.hub.load(repo, "dinov2_vits14", trust_repo=True, force_reload=False)
    except Exception:
        # Download may succeed even if import fails on Python 3.9.
        pass
    return _find_hub_dinov2_dir()


def _load_dinov2(name: str):
    if sys.version_info >= (3, 10):
        return torch.hub.load(_DINOV2_REPO, name, trust_repo=True)

    pinned = f"{_DINOV2_REPO}:{_DINOV2_PY39_REF}"
    print(
        f"[DinoV2Encoder] Python {sys.version_info.major}.{sys.version_info.minor}: "
        f"trying {pinned}"
    )
    try:
        return torch.hub.load(pinned, name, trust_repo=True, force_reload=False)
    except Exception as err:
        print(f"[DinoV2Encoder] pinned load failed: {err}")

    print("[DinoV2Encoder] falling back to hub main + local PEP604 patch")
    hub_root = _download_hub_source("main")
    if hub_root is None:
        raise RuntimeError("Could not locate torch.hub dinov2 cache directory")
    n = _patch_pep604_unions(hub_root)
    print(f"[DinoV2Encoder] patched {n} files under {hub_root}")
    return torch.hub.load(
        str(hub_root),
        name,
        source="local",
        trust_repo=True,
    )


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
