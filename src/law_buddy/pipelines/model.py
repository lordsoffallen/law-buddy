from datasets import Dataset, DatasetDict
from transformers import pipeline
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from .embeddings import get_embeddings_model_and_tokenizer, get_embeddings
from typing import Any

import torch


def find_related_laws(
    ds: Dataset | DatasetDict,
    checkpoint: str,
    query: str,
    index_name: str = 'embeddings',
    batch_size: int = 400,
    top_k: int = 3
) -> tuple:
    """ Apply FAISS index to find related laws given a question. """

    model, tokenizer = get_embeddings_model_and_tokenizer(checkpoint)
    embedding = get_embeddings(model, tokenizer, [query], batch_size).cpu().detach().numpy()

    if isinstance(ds, DatasetDict):
        ds = ds['train']    # select the train split if it is a dict

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
    index_name: str = 'embeddings',
    batch_size: int = 400,
    top_k_context: int = 3,
    top_k_answer: int = 3,
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
    index_name: str
        Name of the column that would be used in FAISS as index. This should contain the text embeddings.
    batch_size: int
        Batch size for long context processing. If memory issues occur, reducing these value would help.
    top_k_context: int
        Number of relevant information to use in the context for the Q&A model. Higher context means longer processing
        times. Also note that depending on the input language order of the context can be different so for small values
        using German or English in the question can result in a different result.
    top_k_answer: int
        Number of possible answers to return from the QA model.
    log_info: bool
        Define whether useful information about the background should be logged or not

    Returns
    -------
    answers:
        If top_k_answer is > 1 then returns list of dicts, otherwise dict
    """

    qa_pipeline = pipeline('question-answering', model=qa_checkpoint, tokenizer=qa_checkpoint)

    _, laws = find_related_laws(
        vectordb,
        checkpoint=embeddings_checkpoint,
        query=question,
        top_k=top_k_context,
        batch_size=batch_size,
        index_name=index_name,
    )

    if log_info:
        info_logger(laws)

    # Join top k laws together as a context to QA model
    context = "\n".join(laws['text'])

    answers = qa_pipeline(question=question, context=context, top_k=top_k_answer)

    return answers


def print_answers(answers: list[dict] | dict):
    def _print(r: dict):
        print(f"Response: {r['answer']} Confidence: {r['score']}")

    if isinstance(answers, list):
        for a in answers:
            _print(a)
    else:
        _print(answers)
