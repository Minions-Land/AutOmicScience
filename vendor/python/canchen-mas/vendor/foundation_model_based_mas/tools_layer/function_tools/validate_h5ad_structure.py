from __future__ import annotations

import numpy as np
import scipy.sparse
from langchain_core.tools import tool
from pydantic import BaseModel

try:
    import scanpy as sc
except ImportError:
    try:
        import anndata as ad
    except ImportError as exc:
        class _MissingReadH5AD:
            @staticmethod
            def read_h5ad(data_path: str):
                raise ImportError(
                    'Install scanpy or anndata before using validate_h5ad_structure_tool to read h5ad files.'
                ) from exc

        sc = _MissingReadH5AD()
    else:
        class _AnnDataCompat:
            @staticmethod
            def read_h5ad(data_path: str):
                return ad.read_h5ad(data_path)

        sc = _AnnDataCompat()


class Input(BaseModel):
    data_path: str = ''


class OutputData(BaseModel):
    description: str = ''
    decision: bool = False


class Output(BaseModel):
    output: OutputData = OutputData()


@tool(args_schema=Input)
def validate_h5ad_structure_tool(data_path) -> Output:
    """
    Use this tool to verify whether the h5ad file contains a valid expression matrix (ndarray/csr/csc).

    Args:
        data_path: The path to the h5ad file.

    Returns:
        output: A dictionary containing the verification result description and the boolean decision.
    """
    try:
        print(data_path)
        adata = sc.read_h5ad(data_path)
        assert isinstance(
            adata.X,
            (
                np.ndarray,
                scipy.sparse.csr_matrix,
                scipy.sparse.csc_matrix,
            ),
        ), 'X must be an ndarray/csr/csc matrix.'
        output = {
            'description': 'The h5ad file contains a valid expression matrix.',
            'decision': True,
        }
        return output
    except Exception as e:
        output = {
            'description': f'The h5ad file does not contain a valid expression matrix: {str(e)}',
            'decision': False,
        }
        return output
    finally:
        if 'adata' in locals():
            del adata


TOOLS = [validate_h5ad_structure_tool]
TOOL_CATALOG = [
    {
        'name': 'validate_h5ad_structure_tool',
        'kind': 'function',
        'description': 'Validate whether a h5ad file contains an expression matrix stored as ndarray, csr_matrix, or csc_matrix.',
        'source_repo': 'BiOmics-master',
        'source_file': 'tools/valid_h5ad.py',
    }
]
