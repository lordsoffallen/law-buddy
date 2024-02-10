from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from .tools import device, batch_forward
from .embeddings import get_embeddings_model_and_tokenizer, get_embeddings
from typing import Any
import torch


def get_qa_model_and_tokenizer(checkpoint: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint).to(device)
    # model = torch.compile(model)    # Speed up inference

    return model, tokenizer


def find_related_laws(
    ds: Dataset,
    checkpoint: str,
    query: str,
    index_name: str = 'embeddings',
    batch_size: int = 400,
    top_k: int = 3
) -> tuple:
    """ Apply FAISS index to find related laws given a question. """

    model, tokenizer = get_embeddings_model_and_tokenizer(checkpoint)
    embedding = get_embeddings(model, tokenizer, [query], batch_size).cpu().detach().numpy()

    # Add FAISS Index as vector db
    ds = ds.add_faiss_index(column=index_name)
    scores, samples = ds.get_nearest_examples(index_name, embedding, k=top_k)

    return scores, samples


def process_qa_output(output: QuestionAnsweringModelOutput, inputs: dict | Any):
    start_logits, end_logits = output.start_logits, output.end_logits

    # we first mask the tokens that are not part of the context before taking the softmax.
    # We also mask all the padding tokens (as flagged by the attention mask)
    sequence_ids = inputs.sequence_ids()

    mask = [i != 1 for i in sequence_ids]     # Mask everything except the tokens of the context
    mask[0] = False     # Unmask the [CLS] token

    # Mask all the [PAD] tokens
    mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

    start_logits[mask] = -10000
    end_logits[mask] = -10000

    # Convert logits to probs
    start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

    #  Attribute a score to all possible spans of answer, then take the span with the best score
    candidates = []

    for start_probs, end_probs in zip(start_probabilities, end_probabilities):
        scores = start_probs[:, None] * end_probs[None, :]
        idx = torch.triu(scores).argmax().item()

        start_idx = idx // scores.shape[1]
        end_idx = idx % scores.shape[1]
        score = scores[start_idx, end_idx].item()
        candidates.append((start_idx, end_idx, score))

    print(candidates)


def answer(
    question: str,
    embeddings_checkpoint: str,
    qa_checkpoint: str,
    vectordb: Dataset,
    index_name: str = 'embeddings',
    batch_size: int = 400,
    top_k_context: int = 3,
):
    """

    Parameters
    ----------
    question: str
        A user question to retrieve answer for
    embeddings_checkpoint: str
        Model checkpoint for sentence embeddings.
    qa_checkpoint: str
        Model checkpoint for Q&A model
    vectordb: Dataset
        A Dataset that contains the embeddings. Here we simply use FAISS index to search
    index_name: str
        Name of the column that would be used in FAISS as index. This should contain the text embeddings.
    batch_size: int
        Batch size for long context processing. If memory issues occur, reducing these value would help.
    top_k_context: int
        Number of relevant information to use in the context for the Q&A model. Higher context means longer processing
        times. Also note that depending the input language order of the context can be different so for small values
        using German or English in the question can result in a different result.

    Returns
    -------

    """
    model, tokenizer = get_qa_model_and_tokenizer(qa_checkpoint)

    _, laws = find_related_laws(
        vectordb,
        checkpoint=embeddings_checkpoint,
        query=question,
        top_k=top_k_context,
        batch_size=batch_size,
        index_name=index_name,
    )

    # Join top 3 laws together as a context to QA model
    context = "\n".join(laws['text'])

    inputs = tokenizer(
        question,
        context,
        stride=int(tokenizer.model_max_length / 2),
        max_length=tokenizer.model_max_length,
        padding="longest",
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors='pt',
    )

    output = batch_forward(
        model=model,
        encoded_input=inputs,
        input_filter_keys=['offset_mapping', 'overflow_to_sample_mapping'],
        batch_size=batch_size
    )

    process_qa_output(output, inputs)
