from __future__ import annotations

import os

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel
from scipy.sparse import isspmatrix

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
                    'Install scanpy or anndata before using summarize_data_tool to read h5ad files.'
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
    preview_n: int = 5


class Output(BaseModel):
    output: dict = {}


@tool(args_schema=Input)
def summarize_data_tool(data_path, preview_n=5) -> Output:
    """
    Use this tool to automatically analyze the various attributes of AnnData and extract unique values (up to the maximum of preview_n).

    Parameters:
        data_path: Path of the h5ad file
        preview_n: Maximum number of unique values to display for each column

    Returns:
        output: A dictionary containing the key "data_info" and its value is a dictionary containing the unique values of each attribute
    """
    if not os.path.exists(data_path):
        return {'data_info': 'No target data found.'}

    adata = sc.read_h5ad(data_path)
    attributes = ['X', 'obs', 'var', 'uns', 'obsm', 'varm', 'layers', 'obsp']
    data_info = {}

    def convert_to_serializable(obj):
        """Recursive conversion of non-serializable objects to basic types"""
        if isspmatrix(obj):
            return obj.toarray().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='list')
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return str(obj)

    try:
        for attr in attributes:
            if not hasattr(adata, attr):
                continue

            try:
                sub = getattr(adata, attr)
                attr_info = {}

                if isinstance(sub, pd.DataFrame):
                    for col in sub.columns:
                        try:
                            values = sub[col].unique()
                            values_list = values[:preview_n].tolist()
                            attr_info[col] = {
                                'preview_unique': values_list,
                                'total_unique': int(len(values)),
                            }
                        except Exception as e:
                            attr_info[col] = {'Error': str(e)}

                elif attr == 'X':
                    try:
                        if isspmatrix(sub):
                            sub = sub.toarray()
                        uniq = np.unique(sub)
                        attr_info = {
                            'preview_unique': uniq[:preview_n].tolist(),
                            'total_unique': int(len(uniq)),
                        }
                    except Exception as e:
                        attr_info = {'Error': str(e)}

                elif isinstance(sub, dict) or (hasattr(sub, 'keys') and callable(sub.keys)):
                    for key in sub.keys():
                        try:
                            val = sub[key]
                            if isinstance(val, pd.DataFrame):
                                col_info = {}
                                for col in val.columns:
                                    try:
                                        values = val[col].unique()
                                        col_info[col] = {
                                            'preview_unique': values[:preview_n].tolist(),
                                            'total_unique': int(len(values)),
                                        }
                                    except Exception as e:
                                        col_info[col] = {'Error': str(e)}
                                attr_info[key] = {'type': 'DataFrame', 'columns': col_info}
                            elif isinstance(val, np.ndarray):
                                uniq = np.unique(val)
                                attr_info[key] = {
                                    'preview_unique': uniq[:preview_n].tolist(),
                                    'total_unique': int(len(uniq)),
                                }
                            elif isspmatrix(val):
                                dense = val.toarray()
                                uniq = np.unique(dense)
                                attr_info[key] = {
                                    'preview_unique': uniq[:preview_n].tolist(),
                                    'total_unique': int(len(uniq)),
                                }
                            else:
                                preview = str(convert_to_serializable(val))[:25]
                                attr_info[key] = {'preview': preview}
                        except Exception as e:
                            attr_info[key] = {'Error': str(e)}

                else:
                    attr_info = {'type': str(type(sub))}

                data_info[attr] = convert_to_serializable(attr_info)

            except Exception as e:
                data_info[attr] = {'Error': str(e)}

        output = {'data_info': data_info}
        return output
    finally:
        del adata


TOOLS = [summarize_data_tool]
TOOL_CATALOG = [
    {
        'name': 'summarize_data_tool',
        'kind': 'function',
        'description': 'Summarize key AnnData attributes including X, obs, var, uns, obsm, varm, layers, and obsp.',
        'source_repo': 'BiOmics-master',
        'source_file': 'tools/summarize_data.py',
    }
]
