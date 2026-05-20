from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from llm_runtime import build_brick_chat_model
from tools_layer.vectorstores import ensure_default_vectorstores, load_default_vectorstores
from tools_layer.vectorstores.registry import DEFAULT_VECTORSTORE_SPECS

output_parser = JsonOutputParser()
str_output_parser = StrOutputParser()


class RetrieveBrickReferencesInput(BaseModel):
    query: str = Field(..., description="The BRICK or Scanpy usage question to answer.")
    vectorstore_names: list[str] | None = Field(
        default=None,
        description="Optional vectorstore names. Defaults to the local BRICK code and notebook indexes.",
    )
    search_k: int = Field(default=5, ge=1, le=20, description="Top-k retrieved chunks per vectorstore.")
    score_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum normalized retrieval score for keeping a retrieved chunk.",
    )


class RAGState(BaseModel):
    query: str = ""
    vectorstore: Optional[Union[list[str], str, dict]] = None
    vect_name: Union[list[str], str, dict] = ""
    code_output: Union[list[str], str, dict] = ""
    notebook_output: Union[list[str], str, dict] = ""
    status: str = "START"
    search_k: int = 5
    score_threshold: float = 0.2
    run_msg: list = ["START"]
    final_result: str = ""
    next: Literal[
        "load_vectorstore",
        "search_code",
        "search_notebook",
        "generate_final_answer",
    ] = "load_vectorstore"


def _get_model():
    return build_brick_chat_model()


def _normalized_similarity_search(vectorstore: Any, query: str, k: int) -> list[tuple[Any, float]]:
    raw_results = vectorstore.similarity_search_with_score(query, k=k)
    normalized_results: list[tuple[Any, float]] = []
    for doc, distance in raw_results:
        score = 1.0 / (1.0 + float(distance))
        normalized_results.append((doc, score))
    return normalized_results


def fuzzy_ratio_match(keyword, text, threshold=0):
    return fuzz.token_sort_ratio(keyword.lower(), text.lower()) >= threshold


def _default_vectorstore_names() -> list[str]:
    return list(DEFAULT_VECTORSTORE_SPECS.keys())


def _load_default_vectorstore_map() -> dict[str, Any]:
    ensure_default_vectorstores()
    return load_default_vectorstores(_default_vectorstore_names())


def load_vectorstore(state: RAGState):
    if isinstance(state.vectorstore, str):
        if not state.vectorstore:
            raise ValueError("Provided vectorstore path cannot be empty.")
        if state.vectorstore in DEFAULT_VECTORSTORE_SPECS:
            loaded = load_default_vectorstores([state.vectorstore])
            state.vect_name = loaded
        else:
            store_name = Path(state.vectorstore).stem
            loaded = load_default_vectorstores([store_name]) if store_name in DEFAULT_VECTORSTORE_SPECS else None
            if loaded:
                state.vect_name = loaded
            else:
                raise FileNotFoundError(f"Vectorstore path or name does not exist: {state.vectorstore}")
        state.run_msg.append("Successfully loaded vectorstore.")
        return {
            "run_msg": state.run_msg,
            "vect_name": state.vect_name,
        }

    if isinstance(state.vectorstore, list):
        if not state.vectorstore:
            raise ValueError("Provided vectorstore list cannot be empty.")
        loaded = load_default_vectorstores(state.vectorstore)
        state.vect_name = loaded
        state.run_msg.append("Successfully loaded vectorstore.")
        return {
            "run_msg": state.run_msg,
            "vect_name": state.vect_name,
        }

    if isinstance(state.vectorstore, dict):
        if not state.vectorstore:
            raise ValueError("Provided vectorstore dict cannot be empty.")
        if all(hasattr(value, "similarity_search_with_score") for value in state.vectorstore.values()):
            state.vect_name = state.vectorstore
        else:
            requested_names = list(state.vectorstore.keys())
            state.vect_name = load_default_vectorstores(requested_names)
        state.run_msg.append("Successfully loaded vectorstore.")
        return {
            "run_msg": state.run_msg,
            "vect_name": state.vect_name,
        }

    if state.vectorstore is None:
        state.vect_name = _load_default_vectorstore_map()
        state.run_msg.append("Successfully loaded vectorstore.")
        return {
            "run_msg": state.run_msg,
            "vect_name": state.vect_name,
        }

    raise TypeError("vectorstore must be str, list[str], dict, or None.")


def search_code(state: RAGState):
    vect_temp = f""" 
    #Task# 
    Follow the instruction to extract the name of the vectorstore that stores the code information from all the vectorstore.

    #Content#
    {state.vect_name.keys()}

    #Instruction# 
    Determine whether the name of the vectorstore is related to "code". If it is related, extract it; otherwise, do not extract it.

    #Example# 
    1)
    vectorstore = ["BRICK_code","python","jupyter_notebook","markdown","txt","py","notebook"]
    names = ["BRICK_code", "python", "py"] 
    2)
    vectorstore = ["jupyter_notebook","plot","format","png"]
    names = [] 

    #Output#
    Return the following JSON object: {{"names": "use list object to only store the name of the vectorstore that stores the code information "}}
    Do not include any other text in your response, only the JSON object.
    """
    chain = _get_model() | output_parser
    names = chain.invoke(vect_temp)

    result = []
    vs = names.get("names", [])
    if len(vs) > 0:
        key_temp = f""" 
        #Task# 
        Extract the keywords from the problem, such as function names, package names, and purposes.

        #Question#
        {state.query}

        #Example# 
        query = "Integration by BRICK.pp.complete_results() and Visualization by BRICK.pl.visualization()"
        keywords = ["Integration", "BRICK.pp.complete_results()", "BRICK", "BRICK.pp", "complete_results()", "complete_results", "Visualization", "BRICK.pl.visualization()", "BRICK.pl", "visualization()", "visualization"] 

        #Output#
        Return the following JSON object: {{"keywords": "use list object to only store the keywords in the query"}}
        Do not include any other text in your response, only the JSON object.
        """
        chain = _get_model() | output_parser
        keywords_payload = chain.invoke(key_temp)
        keywords = keywords_payload.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        for name in vs:
            if name not in state.vect_name:
                continue
            results = _normalized_similarity_search(state.vect_name[name], state.query, k=state.search_k)
            filtered_sorted_results = sorted(
                [(doc, score) for doc, score in results if score >= state.score_threshold],
                key=lambda x: x[1],
                reverse=True,
            )

            keyword_score_results = []
            for doc, _ in filtered_sorted_results:
                if keywords:
                    max_score = max(
                        fuzz.partial_ratio(keyword.lower(), doc.page_content.lower())
                        for keyword in keywords
                    )
                else:
                    max_score = 0
                keyword_score_results.append((doc, max_score))

            if len(keyword_score_results) == 0:
                result.append([doc.page_content for doc, _ in filtered_sorted_results])
            else:
                max_score = max(score for _, score in keyword_score_results)
                top_k_results = [(doc, score) for doc, score in keyword_score_results if score == max_score]
                result.append([doc.page_content for doc, _ in top_k_results])

        template = f"""
        #Task#
        Only based on the code context, select the approriate answer to generate a valid python code for the question.
        
        #Context#
        {result}
        
        #Question#
        {state.query}

        #Instruction#
        1. Use the most appropriate code retrieved for the response, and the output code must be exactly the same as the most suitable code. 
        2. The specific function content must be included, and only the function name cannot be answered.
        
        #Output#
        Return your answer in a valid str object.
        """
        chain = _get_model() | str_output_parser
        final_result = chain.invoke(template)
        state.run_msg.append("Successfully search code.")
    else:
        final_result = ""
        state.run_msg.append("There is no vector library name related to the code, so skipping search code.")
    return {
        "run_msg": state.run_msg,
        "code_output": final_result,
    }


def search_notebook(state: RAGState):
    vect_temp = f""" 
    #Task# 
    Follow the instruction to extract the name of the vectorstore that stores the notebook information from all the vectorstore.

    #Content#
    {state.vect_name.keys()}

    #Instruction# 
    Determine whether the name of the vectorstore is related to "notebook". If it is related, extract it; otherwise, do not extract it.

    #Example# 
    1)
    vectorstore = ["BRICK_code","python","jupyter_notebook","markdown","txt","py","notebook"]
    names = ["jupyter_notebook", "markdown", "txt", "notebook"] 
    2)
    vectorstore = ["BRICK","BRICK_code","BRICK_code2","code","plot","format","png"]
    names = [] 

    #Output#
    Return the following JSON object: {{"names": "use list object to only store the name of the vectorstore that stores the notebook information"}}
    Do not include any other text in your response, only the JSON object.
    """
    chain = _get_model() | output_parser
    names = chain.invoke(vect_temp)

    result = []
    vs = names.get("names", [])
    if len(vs) > 0:
        key_temp = f""" 
        #Task# 
        Extract the keywords from the problem, such as function names, package names, and purposes.

        #Question#
        {state.query}

        #Example# 
        query = "Integration by BRICK.pp.complete_results() and Visualization by BRICK.pl.visualization()"
        keywords = ["Integration", "BRICK.pp.complete_results()", "BRICK", "BRICK.pp", "complete_results()", "complete_results", "Visualization", "BRICK.pl.visualization()", "BRICK.pl", "visualization()", "visualization"] 

        #Output#
        Return the following JSON object: {{"keywords": "use list object to only store the keywords in the query"}}
        Do not include any other text in your response, only the JSON object.
        """
        chain = _get_model() | output_parser
        keywords_payload = chain.invoke(key_temp)
        keywords = keywords_payload.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        for name in vs:
            if name not in state.vect_name:
                continue
            results = _normalized_similarity_search(state.vect_name[name], state.query, k=state.search_k)
            filtered_sorted_results = sorted(
                [(doc, score) for doc, score in results if score >= state.score_threshold],
                key=lambda x: x[1],
                reverse=True,
            )

            keyword_score_results = []
            for doc, _ in filtered_sorted_results:
                if keywords:
                    max_score = max(
                        fuzz.partial_ratio(keyword.lower(), doc.page_content.lower())
                        for keyword in keywords
                    )
                else:
                    max_score = 0
                keyword_score_results.append((doc, max_score))

            if len(keyword_score_results) == 0:
                result.append([doc.page_content for doc, _ in filtered_sorted_results])
            else:
                max_score = max(score for _, score in keyword_score_results)
                top_k_results = [(doc, score) for doc, score in keyword_score_results if score == max_score]
                result.append([doc.page_content for doc, _ in top_k_results])

        template = f"""
            #Task#
            Only based on the notebook context, select the approriate answer to generate a valid answer for the question.
            
            #Context#
            {result}
            
            #Question#
            {state.query}
            
            #Instruction# 
            1. Use the most appropriate context retrieved for the response, and the output context must be exactly the same as the most suitable context. 
            2. The output context needs to be complete. 

            #Output#
            Return your answer in a valid str object.
        """
        chain = _get_model() | str_output_parser
        final_result = chain.invoke(template)
        state.run_msg.append("Successfully search notebook.")
    else:
        final_result = ""
        state.run_msg.append("There is no vector library name related to the notebook, so skipping search notebook.")
    return {
        "run_msg": state.run_msg,
        "notebook_output": final_result,
    }


def generate_final_answer(state: RAGState):
    template = f"""
    #Content#
    - query:{state.query}
    - code information:{state.code_output}
    - code execution examples:{state.notebook_output}
    
    #Instruction#
    1. Check the code information and the code execution examples are relevant or not.
        If relevant, complete the code context by using these two information, but avoid using repetitive parts.
            For example: you found 
                ```python
                import scanpy as sc
                adata = sc.read_h5ad(file_path)
                ``` 
                and 
                ```python
                def load_preprocessed_h5ad(file_path):
                adata = sc.read_h5ad(file_path)
                return adata
                ```
                The second piece of code employs the same code as the first one: adata = sc.read_h5ad(file_path). The difference between these two pieces of code lies in that the second one uses another load_preprocessed_h5ad function to wrap the sc.read_h5ad function.
                Thus, you can use the first code with real file_path.
        Else, check the query and select the best and relevant code to answer this query.
    2. Not only provide the function name, but also provide the function content.

    #Example#
    1) The code information provides a function `run_paga` that encapsulates the execution of `sc.tl.paga`, while the code execution example directly shows how to run `sc.tl.paga` on an `adata` object.
    To complete the code context without repetition, we can use the function `sc.tl.paga` to run the PAGA.
    
    2) The code information provides a function `sc.read_h5ad` that can load h5ad file, while the code execution example shows a function `load_preprocessed_h5ad` that encapsulates the execution of `sc.read_h5ad`
    To complete the code context without repetition, we can use the function `sc.read_h5ad` to load h5ad.

    #Format#
    Return your final answer in a valid str object.
    """
    chain = _get_model() | str_output_parser
    final_result = chain.invoke(template)
    return {"final_result": final_result}


@lru_cache(maxsize=1)
def _build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("load_vectorstore", load_vectorstore)
    builder.add_node("search_code", search_code)
    builder.add_node("search_notebook", search_notebook)
    builder.add_node("generate_final_answer", generate_final_answer)
    builder.add_edge(START, "load_vectorstore")
    builder.add_edge("load_vectorstore", "search_code")
    builder.add_edge("search_code", "search_notebook")
    builder.add_edge("search_notebook", "generate_final_answer")
    builder.add_edge("generate_final_answer", END)
    return builder.compile()


def run_original_style_rag(
    query: str,
    vectorstore=None,
    search_k: int = 5,
    score_threshold: float = 0.2,
):
    initial_state = RAGState(
        query=query,
        vectorstore=vectorstore,
        search_k=search_k,
        score_threshold=score_threshold,
        run_msg=["START"],
    )
    graph = _build_rag_graph()
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 200}
    answer = graph.invoke(initial_state, config)
    return answer


def run_brick_reference_pipeline(
    *,
    query: str,
    vectorstore_names: list[str] | None = None,
    search_k: int = 5,
    score_threshold: float = 0.2,
) -> dict:
    selected_names = vectorstore_names or _default_vectorstore_names()
    result = run_original_style_rag(
        query=query,
        vectorstore=selected_names,
        search_k=search_k,
        score_threshold=score_threshold,
    )
    selected_code_vectorstores = [name for name in result.get("vect_name", {}).keys() if "code" in name.lower()]
    selected_notebook_vectorstores = [
        name for name in result.get("vect_name", {}).keys() if "notebook" in name.lower()
    ]
    return {
        "success": True,
        "query": query,
        "vectorstore_names": selected_names,
        "selected_code_vectorstores": selected_code_vectorstores,
        "selected_notebook_vectorstores": selected_notebook_vectorstores,
        "code_output": result.get("code_output", ""),
        "notebook_output": result.get("notebook_output", ""),
        "final_answer": result.get("final_result", ""),
        "run_log": result.get("run_msg", []),
    }


@tool(args_schema=RetrieveBrickReferencesInput)
def retrieve_brick_references_tool(
    query: str,
    vectorstore_names: list[str] | None = None,
    search_k: int = 5,
    score_threshold: float = 0.2,
) -> dict:
    """
    Retrieve BRICK implementation references from local code and notebook vectorstores,
    then synthesize a final answer for the query.

    Internal orchestration now follows the original BRICK RAG search flow much more closely:
    load vectorstore -> search code -> search notebook -> generate final answer.
    """
    return run_brick_reference_pipeline(
        query=query,
        vectorstore_names=vectorstore_names,
        search_k=search_k,
        score_threshold=score_threshold,
    )


TOOLS = [retrieve_brick_references_tool]
TOOL_CATALOG = [
    {
        "name": "retrieve_brick_references_tool",
        "kind": "function",
        "description": (
            "Retrieve BRICK-related code and notebook references from local FAISS indexes "
            "and synthesize an implementation-focused answer."
        ),
        "source_repo": "BiOmics-master",
        "source_file": "tools/brick_rag_searcher.py",
        "migrated_from": ["perform_rag_search_tool", "run_rag_pipeline_tool"],
    }
]
