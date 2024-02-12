import numpy as np
from datasets import Dataset, DatasetDict
from transformers import pipeline
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from .embeddings import get_embeddings_model_and_tokenizer, get_embeddings
from typing import Any

import torch


def get_embedded_query(checkpoint: str, query: str, batch_size: int = 400):
    """ Given a user query create embeddings for it """
    model, tokenizer = get_embeddings_model_and_tokenizer(checkpoint)
    embedding = get_embeddings(model, tokenizer, [query], batch_size).cpu().detach().numpy()
    return embedding


def find_related_laws(
    ds: Dataset | DatasetDict,
    embedding: np.ndarray | Any,
    index_name: str = 'embeddings',
    top_k: int = 3
) -> tuple:
    """ Apply FAISS index to find related laws given a question. """

    if isinstance(ds, DatasetDict):
        ds = ds['train']    # select the train split if it is a dict

    # Add FAISS Index as vector db
    ds = ds.add_faiss_index(column=index_name)
    scores, samples = ds.get_nearest_examples(index_name, embedding, k=top_k)

    return scores, samples


def find_related_sections(
    ds: dict | Dataset | DatasetDict,
    embedding: np.ndarray | Any,
    index_name: str = 'section_embeddings',
    top_k: int = 3
) -> tuple:

    if isinstance(ds, dict):
        ds = Dataset.from_dict(ds)

    if isinstance(ds, DatasetDict):
        ds = ds['train']

    ds = (ds.with_format("pandas").select_columns(['text', 'sections', 'section_embeddings'])
            .map(lambda df: df.explode(["sections", "section_embeddings"])))

    ds.reset_format()

    # Add FAISS Index as vector db
    ds = ds.add_faiss_index(column=index_name)
    scores, samples = ds.get_nearest_examples(index_name, embedding, k=top_k)
    return scores, samples


def info_logger(ds: dict | Dataset):
    from nltk import sent_tokenize, word_tokenize
    import nltk

    def _get_infos():
        return ds.map(
            lambda x: {
                "char_count": len(x['text']),
                "word_count": len(word_tokenize(x['text'], language='german')),
                "sent_count": len(sent_tokenize(x['text'], language='german')),
            }
        )

    if isinstance(ds, dict):
        ds = Dataset.from_dict(ds)

    try:
        ds = _get_infos()
    except LookupError:
        # nltk fails install packages
        nltk.download('punkt')
        ds = _get_infos()

    for i, law in enumerate(ds):
        print(
            f"Context {i}: "
            f"Total Characters: {law['char_count']}, "
            f"Total Words: {law['word_count']} "
            f"Total Sentences: {law['sent_count']}"
        )


def answer(
    question: str,
    embeddings_checkpoint: str,
    qa_checkpoint: str,
    vectordb: Dataset | DatasetDict,
    index: dict = None,
    batch_size: int = 400,
    context: dict = None,
    log_info: bool = True,
) -> list[dict] | dict:
    """ Given a user query, retrieve the most likely answer.

    Parameters
    ----------
    question: str
        A user question to retrieve answer for
    embeddings_checkpoint: str
        Model checkpoint for sentence embeddings.
    qa_checkpoint: str
        Model checkpoint for Q&A model
    vectordb: Dataset | DatasetDict
        A Dataset that contains the embeddings. Here we simply use FAISS index to search
    index: dict
        Name of the columns that would be used in FAISS as index. This should contain the text embeddings.
    batch_size: int
        Batch size for long context processing. If memory issues occur, reducing these value would help.
    context: dict
        Contains a dict for relevant top_k results for both embeddings model and Q&A model
    log_info: bool
        Define whether useful information about the background should be logged or not

    Returns
    -------
    answers:
        If context['answer'] is > 1 then returns list of dicts, otherwise dict
    """

    qa_pipeline = pipeline('question-answering', model=qa_checkpoint, tokenizer=qa_checkpoint)

    query_embedding = get_embedded_query(checkpoint=embeddings_checkpoint, query=question, batch_size=batch_size)

    _, laws = find_related_laws(
        vectordb, embedding=query_embedding, top_k=context['embeddings']['laws'], index_name=index['laws']
    )

    if log_info:
        info_logger(laws)

    _, sections = find_related_sections(
        laws, query_embedding, index_name=index['sections'], top_k=context['embeddings']['sections']
    )

    # Join top k sections together as a context to QA model
    context = "\n".join(sections['sections'])

    answers = qa_pipeline(question=question, context=context, top_k=context['answer'])

    return answers


def print_answers(answers: list[dict] | dict):
    def _print(r: dict):
        print(f"Response: {r['answer']} Confidence: {r['score']}")

    if isinstance(answers, list):
        for a in answers:
            _print(a)
    else:
        _print(answers)
