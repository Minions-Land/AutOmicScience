from __future__ import annotations

from typing import Any

from pydantic import Field

from .contracts import BaseEmbeddingSkillInput
from .data_utils import resolve_device
from .foundation_embedding_common import run_embedding_skill
from .model_backends import ScGPTGenericEmbeddingEncoder, scgpt_vocab_upper


SKILL_NAME = "scgpt_generic_embedding_skill"
SKILL_DESCRIPTION = (
    "Generate reference/query embeddings using scGPT generic checkpoint and emit an EmbeddingPackage."
)


class ScGPTGenericEmbeddingSkillInput(BaseEmbeddingSkillInput):
    model_id: str = Field(default="scgpt_generic")


def scgpt_generic_embedding_skill(**kwargs) -> dict[str, Any]:
    args = ScGPTGenericEmbeddingSkillInput(**kwargs)
    resolved_device = resolve_device(args.device)

    def _build_encoder(gene_names: list[str], species: str) -> ScGPTGenericEmbeddingEncoder:
        del species
        return ScGPTGenericEmbeddingEncoder(
            gene_names=gene_names,
            device=resolved_device,
            random_seed=args.random_seed,
        )

    return run_embedding_skill(
        skill_name=SKILL_NAME,
        args=args,
        build_encoder=_build_encoder,
        vocab_fn=lambda: scgpt_vocab_upper("scgpt_generic"),
        use_query_panel_for_both_sides=True,
    )


SKILL_TOOLS = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
        "tool": scgpt_generic_embedding_skill,
        "args_schema": ScGPTGenericEmbeddingSkillInput,
        "return_direct": False,
    }
]

SKILL_TOOL_CATALOG = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
    }
]
