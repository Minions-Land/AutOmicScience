from .embedding import (
    geneformer_embedding_skill,
    nicheformer_embedding_skill,
    scgpt_generic_brain_embedding_skill,
    scgpt_generic_embedding_skill,
    scgpt_human_embedding_skill,
    uce_33l_embedding_skill,
    uce_embedding_skill,
)
from .foundation_embedding_geneformer_skill import (
    SKILL_TOOL_CATALOG as GENEFORMER_SKILL_TOOL_CATALOG,
)
from .foundation_embedding_geneformer_skill import SKILL_TOOLS as GENEFORMER_SKILL_TOOLS
from .foundation_embedding_nicheformer_skill import (
    SKILL_TOOL_CATALOG as NICHEFORMER_SKILL_TOOL_CATALOG,
)
from .foundation_embedding_nicheformer_skill import SKILL_TOOLS as NICHEFORMER_SKILL_TOOLS
from .foundation_embedding_scgpt_generic_skill import (
    SKILL_TOOL_CATALOG as SCGPT_GENERIC_SKILL_TOOL_CATALOG,
)
from .foundation_embedding_scgpt_generic_skill import SKILL_TOOLS as SCGPT_GENERIC_SKILL_TOOLS
from .shared_knn_transfer_skill import SKILL_TOOL_CATALOG as KNN_SKILL_TOOL_CATALOG
from .shared_knn_transfer_skill import SKILL_TOOLS as KNN_SKILL_TOOLS
from .shared_knn_transfer_skill import shared_knn_transfer_skill
from .shared_prediction_analysis_skill import SKILL_TOOL_CATALOG as ANALYSIS_SKILL_TOOL_CATALOG
from .shared_prediction_analysis_skill import SKILL_TOOLS as ANALYSIS_SKILL_TOOLS
from .shared_prediction_analysis_skill import shared_prediction_analysis_skill


SKILL_TOOLS = [
    *GENEFORMER_SKILL_TOOLS,
    *NICHEFORMER_SKILL_TOOLS,
    *SCGPT_GENERIC_SKILL_TOOLS,
    *KNN_SKILL_TOOLS,
    *ANALYSIS_SKILL_TOOLS,
]

SKILL_TOOL_CATALOG = [
    *GENEFORMER_SKILL_TOOL_CATALOG,
    *NICHEFORMER_SKILL_TOOL_CATALOG,
    *SCGPT_GENERIC_SKILL_TOOL_CATALOG,
    *KNN_SKILL_TOOL_CATALOG,
    *ANALYSIS_SKILL_TOOL_CATALOG,
]


__all__ = [
    "SKILL_TOOLS",
    "SKILL_TOOL_CATALOG",
    "geneformer_embedding_skill",
    "nicheformer_embedding_skill",
    "scgpt_generic_embedding_skill",
    "scgpt_human_embedding_skill",
    "scgpt_generic_brain_embedding_skill",
    "uce_embedding_skill",
    "uce_33l_embedding_skill",
    "shared_knn_transfer_skill",
    "shared_prediction_analysis_skill",
]
