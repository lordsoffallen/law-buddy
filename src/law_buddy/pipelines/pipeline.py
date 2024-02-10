from kedro.pipeline import Pipeline, pipeline, node
from .data import create_dataset
from .embeddings import compute_embeddings
from .model import add_faiss_index, find_related_laws


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(lambda x: x, inputs='laws#api', outputs='laws#pkl', name='fetch_laws'),
        node(create_dataset, inputs='laws#pkl', outputs='laws#hf', name='create_dataset'),
        node(
            compute_embeddings,
            inputs=['laws#hf', 'params:embeddings_model_checkpoint', 'params:batch_size'],
            outputs='embeddings',
            name='compute_embeddings',
        ),
        node(
            add_faiss_index,
            inputs=['embeddings#hf', 'params:index_name'],
            outputs='vectordb',
            name='add_faiss_index'
        ),
        node(
            find_related_laws,
            inputs=dict(
                ds='vectordb',
                checkpoint='params:embeddings_model_checkpoint',
                query='params:query',
                index_name='params:index_name',
                batch_size='params:batch_size',
                top_k='params:top_k',
            ),
            outputs='scores_and_samples',
            name='find_related_laws',
        ),
    ])
