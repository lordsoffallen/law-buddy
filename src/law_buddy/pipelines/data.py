from datasets import Dataset
import pandas as pd


def create_dataset(laws: dict[dict]) -> Dataset:
    """ Takes the scrapped data as input and converts that to Dataset object """

    df = pd.DataFrame.from_dict(laws, orient='index')
    df = df.reset_index(names='page_url')
    df = df.rename(columns={'url': 'law_page_url'})

    return Dataset.from_pandas(df)
