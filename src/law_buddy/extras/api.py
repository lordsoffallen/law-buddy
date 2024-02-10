from kedro.io.core import AbstractDataset
from typing import Any
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from requests import HTTPError
from unstructured.partition.html import partition_html
from pathlib import Path, PurePosixPath
from kedro.io.core import get_filepath_str, get_protocol_and_path

import fsspec
import requests
import logging
import pickle


logger = logging.getLogger(__name__)


def check_if_http_call_successful(response: requests.Response):
    if response.status_code != 200:
        raise HTTPError(f"Failed to retrieve the webpage={response.url}. Status code: {response.status_code}")


def check_content_length(content, url):
    if len(content) > 1:
        raise ValueError(f'Returned content={url} has multiple divs matching. There should be 1 match only')


class GermanLawAPIDataset(AbstractDataset):
    def __init__(self,
                 cachepath: str,
                 url: str = "https://www.gesetze-im-internet.de",
                 bs4_parser: str = 'lxml'):

        protocol, path = get_protocol_and_path(cachepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

        self.BASE_URL = url
        self.bs4_parser = bs4_parser

    def _save(self, data) -> None:
        raise NotImplementedError("Does not support save method")

    def _describe(self) -> dict[str, Any]:
        return {
            "url": self.BASE_URL,
            "bs4_parser": self.bs4_parser,
            "_protocol": self._protocol,
            "_filepath": self._filepath,
            "_fs": self._fs,
        }

    def _load(self) -> dict[dict[str, str]]:
        return self.crawl()

    def _read_cache(self, path: str | Path) -> Any | None:
        fp = self._filepath.joinpath(path).joinpath("data.pkl")
        if Path(fp).exists():
            load_path = get_filepath_str(fp, self._protocol)
            with self._fs.open(load_path, mode="rb") as f:
                cache = pickle.load(f)
            return cache

    def _write_cache(self, obj: Any, path: str | Path):
        fp = Path(self._filepath).joinpath(path)
        save_path = get_filepath_str(fp, self._protocol)

        Path(save_path).mkdir(parents=True, exist_ok=True)

        with self._fs.open(Path(save_path).joinpath("data.pkl"), mode="wb") as f:
            pickle.dump(obj, f)

    def get_homepage_links(self) -> list[str]:
        """ Extract alphabet links from the main page """
        cache_path = 'homepage'

        cache = self._read_cache(cache_path)
        if cache is not None:
            return cache

        logger.info(f"Cache={cache_path} does not exist. Pulling new data")

        response = requests.get(f"{self.BASE_URL}/aktuell.html")

        check_if_http_call_successful(response)
        soup = BeautifulSoup(response.content, self.bs4_parser)
        links = [
            f"{self.BASE_URL}{link['href'][1:]}" for link in soup.find_all('a', href=True, class_='alphabet')
        ]
        self._write_cache(links, cache_path)

        return links

    def get_page_links(self, url: str) -> list[str]:
        """ Given a page link (not main page) it extracts all the relevant law links """
        unique_id = url.replace(".html", "").split('/')[-1]
        cache_path = Path('pages').joinpath(unique_id)

        cache = self._read_cache(cache_path)
        if cache is not None:
            return cache

        logger.info(f"Cache={cache_path} does not exist. Pulling new data")

        response = requests.get(url)

        check_if_http_call_successful(response)

        soup = BeautifulSoup(response.content, self.bs4_parser)
        # Ignore other divs
        content = soup.find_all('div', id='content_2022')
        check_content_length(content, url)

        links = [
            f"{self.BASE_URL}{match['href'][1:]}" for match in content[0].find_all('a', href=True)
            if match['href'].endswith('index.html')
        ]
        self._write_cache(links, cache_path)

        return links

    def get_law_link(self, url: str) -> str:

        response = requests.get(url)

        check_if_http_call_successful(response)

        soup = BeautifulSoup(response.content, self.bs4_parser)
        # Ignore other divs
        content = soup.find_all('div', id='content_2022')
        check_content_length(content, url)

        content = content[0].find_all('h2', class_='headline')
        check_content_length(content, url)

        links = content[0].find_all('a', href=True)

        return f"{url.replace('/index.html', '')}/{links[0]['href']}"

    @staticmethod
    def get_data(url: str) -> str:

        data = partition_html(url=url)

        # Remove table data as it's hard to parse properly
        # remove footer links text as well (last 6 text data)
        return "\n".join([t.text for t in data if t.tag != 'table'][:-6])

    def crawl(self) -> dict[dict[str, str]]:
        cache_path = Path('crawl')
        law_texts = self._read_cache(cache_path) or {}

        for home_link in tqdm(self.get_homepage_links(), desc="Home Page"):
            for page_link in tqdm(self.get_page_links(home_link), desc=f"{home_link}"):
                if page_link not in law_texts:
                    url = self.get_law_link(page_link)
                    logger.info(f'Crawling page={page_link}, url={url}')

                    try:
                        text = self.get_data(url)
                        law_texts[page_link] = dict(url=url, text=text)
                    except Exception as e:
                        self._write_cache(law_texts, cache_path)
                        raise Exception(f"Following exception occurred={e}")

        return law_texts
