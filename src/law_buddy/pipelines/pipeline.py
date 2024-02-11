from kedro.pipeline import Pipeline, pipeline, node
from .data import create_dataset
from .embeddings import compute_embeddings
from .model import answer, print_answers


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(lambda x: x, inputs='laws#api', outputs='laws#pkl', name='fetch_laws'),
        node(create_dataset, inputs='laws#pkl', outputs='laws', name='create_dataset'),
        node(
            compute_embeddings,
            inputs=['laws', 'params:embeddings_model_checkpoint', 'params:batch_size'],
            outputs='embeddings',
            name='compute_embeddings',
        ),
        node(
            answer,
            inputs=dict(
                question='params:query',
                embeddings_checkpoint='params:embeddings_model_checkpoint',
                qa_checkpoint='params:qa_model_checkpoint',
                vectordb='embeddings#hf',       # If end to end pipeline executed, simply `embeddings` would work too.
                index_name='params:index_name',
                batch_size='params:batch_size',
                top_k_context='params:top_k_context',
                top_k_answer='params:top_k_answer',
                log_info='params:log_info',
            ),
            outputs='answers',
            name='answer',
        ),
        node(print_answers, inputs='answers', outputs=None, name='print_answers'),
    ])
