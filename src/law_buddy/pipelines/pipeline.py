from kedro.pipeline import Pipeline, pipeline, node
from .data import create_dataset
from .embeddings import compute_embeddings
from .model import answer, print_answers, get_question_context


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
            get_question_context,
            inputs=dict(
                question='params:query',
                embeddings_checkpoint='params:embeddings_model_checkpoint',
                vectordb='embeddings#hf',
                index='params:index',
                batch_size='params:batch_size',
                top_k='params:top_k',
                log_info='params:log_info',
            ),
            outputs='qa_context',
            name='get_question_context',
        ),
        node(
            answer,
            inputs=dict(
                question='params:query',
                embeddings_checkpoint='params:embeddings_model_checkpoint',
                qa_checkpoint='params:qa_model_checkpoint',
                qa_context='qa_context',
                vectordb='embeddings#hf',
                index='params:index',
                batch_size='params:batch_size',
                top_k='params:top_k',
                log_info='params:log_info',
            ),
            outputs='answers',
            name='answer',
        ),
        node(print_answers, inputs='answers', outputs=None, name='print_answers'),
    ])
