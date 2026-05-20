from __future__ import annotations

from typing import Any

from pydantic import Field

from .contracts import BaseEmbeddingSkillInput
from .data_utils import resolve_device
from .foundation_embedding_common import run_embedding_skill
from .model_backends import GeneformerEmbeddingEncoder, geneformer_vocab_upper


SKILL_NAME = "geneformer_embedding_skill"
SKILL_DESCRIPTION = (
    "Generate reference/query embeddings using Geneformer and emit an EmbeddingPackage for shared KNN transfer."
)


class GeneformerEmbeddingSkillInput(BaseEmbeddingSkillInput):
    model_id: str = Field(default="geneformer")


def geneformer_embedding_skill(**kwargs) -> dict[str, Any]:
    args = GeneformerEmbeddingSkillInput(**kwargs)
    resolved_device = resolve_device(args.device)

    def _build_encoder(gene_names: list[str], species: str) -> GeneformerEmbeddingEncoder:
        del species
        return GeneformerEmbeddingEncoder(gene_names=gene_names, device=resolved_device)

    return run_embedding_skill(
        skill_name=SKILL_NAME,
        args=args,
        build_encoder=_build_encoder,
        vocab_fn=geneformer_vocab_upper,
        use_query_panel_for_both_sides=False,
    )


SKILL_TOOLS = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
        "tool": geneformer_embedding_skill,
        "args_schema": GeneformerEmbeddingSkillInput,
        "return_direct": False,
    }
]

SKILL_TOOL_CATALOG = [
    {
        "name": SKILL_NAME,
        "description": SKILL_DESCRIPTION,
    }
]
