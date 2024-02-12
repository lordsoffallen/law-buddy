from kedro.pipeline import Pipeline, pipeline, node
from .data import create_dataset
from .embeddings import compute_embeddings
from .model import answer, print_answers


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(lambda x: x, inputs='laws#api', outputs='laws#pkl', name='fetch_laws'),
        node(create_dataset, inputs='laws#pkl', outputs='laws#hf', name='create_dataset'),
        node(
            compute_embeddings,
            inputs=['laws#hf', 'params:embeddings_model_checkpoint', 'params:batch_size'],
            outputs='embeddings#hf',
            name='compute_embeddings',
        ),
        node(
            answer,
            inputs=dict(
                question='params:query',
                embeddings_checkpoint='params:embeddings_model_checkpoint',
                qa_checkpoint='params:qa_model_checkpoint',
                vectordb='embeddings#hf',
                index='params:index',
                batch_size='params:batch_size',
                context='params:context',
                log_info='params:log_info',
            ),
            outputs='answers',
            name='answer',
        ),
        node(print_answers, inputs='answers', outputs=None, name='print_answers'),
    ])
