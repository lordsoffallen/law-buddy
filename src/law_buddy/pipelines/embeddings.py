from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Any
from .tools import device, batch_forward


def get_embeddings_model_and_tokenizer(checkpoint: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    # model = torch.compile(model)    # Speed up inference

    return model, tokenizer


def mean_pooling(model_output, attention_mask):
    """ Mean Pooling - Take attention mask into account for correct averaging """
    token_embeddings, attention_mask = model_output.to('cpu'), attention_mask.to('cpu')
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text_list: list[str] | str | Any,
    batch_size: int = 400,
):
    encoded_input = tokenizer(
        text_list,
        padding=True,
        max_length=tokenizer.model_max_length,
        stride=int(tokenizer.model_max_length / 2),
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )

    output = batch_forward(
        model=model,
        encoded_input=encoded_input,
        input_filter_keys='overflow_to_sample_mapping',
        batch_size=batch_size,
    )

    # Check if the output is batch of Output or not
    if isinstance(output, list):
        output = torch.concat([v.last_hidden_state for v in output])
    else:
        output = output.last_hidden_state

    pooled_embeddings = mean_pooling(output, encoded_input['attention_mask'])

    if pooled_embeddings.dim() == 2 and pooled_embeddings.shape[0] > 1:
        # Big chunk of text was processed in batches, so we average them again.
        pooled_embeddings = torch.mean(pooled_embeddings, dim=0, keepdim=True)

    return pooled_embeddings


def compute_embeddings(ds: Dataset, checkpoint: str, batch_size: int = 400) -> Dataset:
    """ Compute embeddings for the existing GitHub issues """
    model, tokenizer = get_embeddings_model_and_tokenizer(checkpoint)

    ds = ds.map(
        lambda x: {
            "embeddings": get_embeddings(model, tokenizer, x["text"], batch_size).detach().cpu().numpy()[0]
        }
    )

    return ds
