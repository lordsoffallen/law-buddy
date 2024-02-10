from transformers import PreTrainedModel
import torch
from typing import Any


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def forward(
    model: PreTrainedModel,
    encoded_input: dict | Any,
    input_filter_keys: list | str = 'overflow_to_sample_mapping',
):
    if isinstance(input_filter_keys, str):
        input_filter_keys = [input_filter_keys]

    encoded_input = {k: v.to(device) for k, v in encoded_input.items() if k not in input_filter_keys}

    with torch.no_grad():
        model_output = model(**encoded_input)

    return model_output


def batch_forward(
    model: PreTrainedModel,
    encoded_input: dict | Any,
    input_filter_keys: list | str = 'overflow_to_sample_mapping',
    batch_size: int = 400,
):
    # Huge chunks of text gets split into higher dimensions and cause GPU OOM. Use batch param to fix it
    if encoded_input['input_ids'].shape[0] > batch_size:
        total_batch = encoded_input['input_ids'].shape[0]
        model_output = []

        for idx in range((total_batch + batch_size - 1) // batch_size):
            start, end = idx * batch_size, min((idx + 1) * batch_size, total_batch)
            batch = {k: v[start:end] if v.dim() == 1 else v[start:end, :] for k, v in encoded_input.items()}
            model_output.append(forward(model, batch, input_filter_keys=input_filter_keys))
    else:
        model_output = forward(model, encoded_input, input_filter_keys=input_filter_keys)

    return model_output
