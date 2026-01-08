import requests
import threading
import time
import logging
from typing import List, Dict, Optional, TypedDict, Any

from modules.config import (
    WIKIPEDIA_USER_AGENT,
    WIKIPEDIA_TIMEOUT,
    WIKIPEDIA_MAX_RETRIES,
    WIKIPEDIA_REQUESTS_PER_SECOND,
)

logger = logging.getLogger(__name__)


class WikipediaPageInfo(TypedDict):
    error: bool
    title: str
    links: List[str]
    extract: Optional[str]
    redirections: List[str]


def rate_limited_request(
    session: requests.Session,
    lock: threading.Lock,
    last_request_time: List[float],
    url: str,
    params: Dict[str, Any],
    timeout: int,
    max_retries: int,
    requests_per_second: int,
) -> Optional[requests.Response]:
    """
    Send a GET request with rate limiting and retries.
    """
    for _ in range(max_retries):
        with lock:
            now = time.time()
            min_interval = 1.0 / requests_per_second
            elapsed = now - last_request_time[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        try:
            resp: Optional[requests.Response] = session.get(
                url, params=params, timeout=timeout
            )
        except Exception as e:
            logger.debug("Request exception: %s", e)
            resp = None
        with lock:
            last_request_time[0] = time.time()
        if resp and resp.status_code == 200:
            logger.debug(f"Successful request to {url}")
            return resp
    logger.warning(f"Failed request to {url} after {max_retries} retries")
    return None


class WikiApiRequestHandler:
    def __init__(
        self,
        user_agent: str = WIKIPEDIA_USER_AGENT,
        timeout: int = WIKIPEDIA_TIMEOUT,
        max_retries: int = WIKIPEDIA_MAX_RETRIES,
        requests_per_second: int = WIKIPEDIA_REQUESTS_PER_SECOND,
        access_token: Optional[str] = None,
    ) -> None:
        self.user_agent: str = user_agent
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.requests_per_second: int = requests_per_second
        self.session: requests.Session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        if access_token:
            self.session.headers.update({"Authorization": f"Bearer {access_token}"})

        # Configure connection pool for concurrent requests
        # Pool size matches rate limit to handle max concurrent requests
        from requests.adapters import HTTPAdapter

        adapter = HTTPAdapter(
            pool_connections=self.requests_per_second,
            pool_maxsize=self.requests_per_second,
            max_retries=0,  # We handle retries ourselves
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self._lock: threading.Lock = threading.Lock()
        self._last_request_time: List[float] = [
            0
        ]  # Use list for mutability in function

    def fetch_wikidata_titles(
        self, ids: List[str], languages: List[str], batch_size: int = 50
    ) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
        results: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
        logger.info(
            f"Fetching Wikidata titles for {len(ids)} IDs in {len(languages)} languages"
        )
        for i in range(0, len(ids), batch_size):
            batch_ids: List[str] = ids[i : i + batch_size]
            ids_str: str = "|".join(batch_ids)
            lang_str: str = "|".join(languages)
            sitefilter_str: str = "|".join([f"{lang}wiki" for lang in languages])
            url: str = "https://www.wikidata.org/w/api.php"
            params: Dict[str, Any] = {
                "action": "wbgetentities",
                "format": "json",
                "ids": ids_str,
                "redirects": "yes",
                "props": "sitelinks|descriptions",
                "languages": lang_str,
                "sitefilter": sitefilter_str,
                "formatversion": 2,
            }
            response: Optional[requests.Response] = rate_limited_request(
                self.session,
                self._lock,
                self._last_request_time,
                url,
                params,
                self.timeout,
                self.max_retries,
                self.requests_per_second,
            )
            logger.debug(f"Fetched titles batch {i+len(batch_ids)}/{len(ids)} IDs")
            if response is None or response.status_code != 200:
                logger.warning("Failed Wikidata titles batch of %d IDs", len(batch_ids))
                continue
            response_data: Dict[str, Any] = response.json()
            for entity_id, entity_data in response_data.get("entities", {}).items():
                language_titles: Dict[str, Dict[str, Optional[str]]] = {}
                sitelinks: Dict[str, Any] = entity_data.get("sitelinks", {})
                descriptions: Dict[str, Any] = entity_data.get("descriptions", {})
                for lang in languages:
                    entry: Dict[str, Optional[str]] = {}
                    site_key: str = f"{lang}wiki"
                    entry["title"] = sitelinks.get(site_key, {}).get("title")
                    entry["description"] = descriptions.get(lang, {}).get("value")
                    language_titles[lang] = entry
                results[entity_id] = language_titles
        logger.debug(f"Fetched titles for {len(results)} IDs")
        return results

    def fetch_wikidata_sitelinks(
        self, ids: List[str], languages: List[str], batch_size: int = 50
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Fetch Wikipedia titles (sitelinks) for given Wikidata IDs and languages.
        Returns a dictionary mapping each Wikidata ID to a dictionary of language codes and their corresponding titles.
        """
        results: Dict[str, Dict[str, Optional[str]]] = {}
        logger.info(
            f"Fetching Wikidata sitelinks for {len(ids)} IDs in {len(languages)} languages"
        )
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            ids_str = "|".join(batch_ids)
            sitefilter_str = "|".join([f"{lang}wiki" for lang in languages])
            url = "https://www.wikidata.org/w/api.php"
            params: Dict[str, str | int] = {
                "action": "wbgetentities",
                "format": "json",
                "ids": ids_str,
                "redirects": "yes",
                "props": "sitelinks",
                "sitefilter": sitefilter_str,
                "formatversion": 2,
            }
            logger.debug(f"Requesting sitelinks batch {len(batch_ids)} IDs")
            response = rate_limited_request(
                self.session,
                self._lock,
                self._last_request_time,
                url,
                params,
                self.timeout,
                self.max_retries,
                self.requests_per_second,
            )
            if response is None:
                logger.warning(f"No response for batch ({len(batch_ids)} IDs)")
                continue
            if response.status_code != 200:
                logger.warning(f"Bad status code {response.status_code} for batch")
                continue
            response_data = response.json()
            if "entities" not in response_data:
                logger.warning(f"No 'entities' in response for batch")
                continue
            for entity_id, entity_data in response_data.get("entities", {}).items():
                sitelinks = entity_data.get("sitelinks", {})
                language_titles = {}
                for lang in languages:
                    site_key = f"{lang}wiki"
                    title = sitelinks.get(site_key, {}).get("title")
                    if title is None:
                        logger.debug(f"No sitelink for {entity_id} in '{lang}'")
                    language_titles[lang] = title
                results[entity_id] = language_titles
            logger.debug(f"Processed batch {i+len(batch_ids)}/{len(ids)} IDs")
        logger.debug(f"Fetched sitelinks for {len(results)} IDs")
        return results

    def fetch_page_extract(self, title: str, language: str) -> Optional[str]:
        params: Dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts",
            "titles": title,
            "exlimit": 1,
            "explaintext": 1,
            "exsectionformat": "plain",
        }
        url = f"https://{language}.wikipedia.org/w/api.php"
        resp = rate_limited_request(
            self.session,
            self._lock,
            self._last_request_time,
            url,
            params,
            self.timeout,
            self.max_retries,
            self.requests_per_second,
        )
        if resp is None or resp.status_code != 200:
            return None
        data = resp.json()
        if "query" in data and "pages" in data["query"]:
            for page in data["query"]["pages"]:
                if "missing" in page:
                    return None
                if "extract" in page:
                    return page["extract"]
        return None

    def fetch_page_details(self, title: str, language: str) -> WikipediaPageInfo:
        links: List[str] = []
        extract: Optional[str] = None
        redirections: List[str] = []
        plcontinue: Optional[str] = None
        excontinue: Optional[str] = None
        rdcontinue: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "links|extracts|redirects",
                "titles": title,
                "plnamespace": 0,
                "pllimit": "max",
                "exlimit": 1,
                "explaintext": 1,
                "exsectionformat": "plain",
                "rdprop": "title",
                "rdnamespace": 0,
                "rdlimit": "max",
            }
            if plcontinue:
                params["plcontinue"] = plcontinue
            if excontinue:
                params["excontinue"] = excontinue
            if rdcontinue:
                params["rdcontinue"] = rdcontinue
            url: str = f"https://{language}.wikipedia.org/w/api.php"
            resp: Optional[requests.Response] = rate_limited_request(
                self.session,
                self._lock,
                self._last_request_time,
                url,
                params,
                self.timeout,
                self.max_retries,
                self.requests_per_second,
            )
            logger.debug(f"Requested page details for '{title}' ({language})")
            if resp is None or resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch page details for '{title}' ({language})"
                )
                return {
                    "error": True,
                    "title": title,
                    "links": [],
                    "extract": None,
                    "redirections": [],
                }
            data: Dict[str, Any] = resp.json()
            if "query" in data and "pages" in data["query"]:
                for page in data["query"]["pages"]:
                    if "missing" in page:
                        logger.debug(f"Page '{title}' ({language}) is missing")
                        return {
                            "error": True,
                            "title": title,
                            "links": [],
                            "extract": None,
                            "redirections": [],
                        }
                    if "links" in page:
                        links.extend(
                            [l["title"] for l in page["links"] if l.get("ns", 0) == 0]
                        )
                    if "extract" in page and extract is None:
                        extract = page["extract"]
                    if "redirects" in page:
                        redirections.extend(
                            [r.get("title") for r in page["redirects"] if "title" in r]
                        )
            cont: Dict[str, Any] = data.get("continue", {})
            plcontinue = cont.get("plcontinue")
            excontinue = cont.get("excontinue")
            rdcontinue = cont.get("rdcontinue")
            if not plcontinue and not excontinue and not rdcontinue:
                break
        logger.debug(f"Fetched page details for '{title}' ({language})")
        return {
            "error": False,
            "title": title,
            "links": links,
            "extract": extract,
            "redirections": redirections,
        }

    def fetch_links_qids(self, title: str, language: str) -> list[tuple[str, str]]:
        """
        Fetch QIDs for all links on a given Wikipedia page.
        Returns a list of (page_title, QID) tuples.
        """
        params: Dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "prop": "pageprops",
            "titles": title,
            "generator": "links",
            "redirects": 1,
            "formatversion": 2,
            "gplnamespace": 0,
            "gpllimit": "500",
        }
        url = f"https://{language}.wikipedia.org/w/api.php"
        qids: list[tuple[str, str]] = []
        gplcontinue = None

        while True:
            if gplcontinue:
                params["gplcontinue"] = gplcontinue
            resp = rate_limited_request(
                self.session,
                self._lock,
                self._last_request_time,
                url,
                params,
                self.timeout,
                self.max_retries,
                self.requests_per_second,
            )
            if resp is None or resp.status_code != 200:
                break
            data = resp.json()
            if "query" in data and "pages" in data["query"]:
                for page in data["query"]["pages"]:
                    if "missing" in page:
                        continue
                    page_title = page.get("title")
                    qid = page.get("pageprops", {}).get("wikibase_item")
                    if page_title and qid:
                        qids.append((page_title, qid))
            cont = data.get("continue", {})
            gplcontinue = cont.get("gplcontinue")
            if not gplcontinue:
                break
        return qids

    def fetch_page_links(
        self, title: str, language: str, redirect: bool = False
    ) -> List[str]:
        links: List[str] = []
        plcontinue: Optional[str] = None
        while True:
            params: Dict[str, Any] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "links",
                "titles": title,
                "plnamespace": 0,
                "pllimit": "max",
            }
            if redirect:
                params["redirects"] = 1
            else:
                params["redirects"] = 0
            if plcontinue:
                params["plcontinue"] = plcontinue
            url: str = f"https://{language}.wikipedia.org/w/api.php"
            resp: Optional[requests.Response] = rate_limited_request(
                self.session,
                self._lock,
                self._last_request_time,
                url,
                params,
                self.timeout,
                self.max_retries,
                self.requests_per_second,
            )
            logging.debug(
                f"Requested page links for '{title}' in '{language}' with redirects={redirect}"
            )
            if resp is None or resp.status_code != 200:
                logging.warning(
                    f"Failed to fetch page links for '{title}' in '{language}'"
                )
                return []
            data: Dict[str, Any] = resp.json()
            if "query" in data and "pages" in data["query"]:
                for page in data["query"]["pages"]:
                    if "missing" in page:
                        logging.debug(f"Page '{title}' in '{language}' is missing.")
                        return []
                    if "links" in page:
                        links.extend(
                            [l["title"] for l in page["links"] if l.get("ns", 0) == 0]
                        )
            cont: Dict[str, Any] = data.get("continue", {})
            plcontinue = cont.get("plcontinue")
            if not plcontinue:
                break
        logging.debug(
            f"Finished fetching page links for '{title}' in '{language}' with redirects={redirect}"
        )
        return links

    def fetch_pageviews(
        self, title: str, language: str, start: str, end: str
    ) -> Optional[int]:
        """
        Fetch total pageviews for a Wikipedia article between start and end dates.

        Args:
            title: Article title (URL-encoded internally)
            language: Language code (e.g., 'en', 'fr')
            start: Start date in YYYYMMDD format (e.g., '20230101')
            end: End date in YYYYMMDD format (e.g., '20251214')

        Returns:
            Total pageviews as an integer, or None if request fails
        """
        import urllib.parse

        # URL encode the title for the API request
        encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="")

        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{language}.wikipedia.org/all-access/all-agents/{encoded_title}/daily/{start}/{end}"
        )

        # Note: Wikimedia REST API doesn't require User-Agent in the same way
        # but we'll use our session which already has it configured
        resp = rate_limited_request(
            self.session,
            self._lock,
            self._last_request_time,
            url,
            {},  # No query params needed - everything is in the URL
            self.timeout,
            self.max_retries,
            self.requests_per_second,
        )

        if resp is None or resp.status_code != 200:
            logging.warning(
                f"Failed to fetch pageviews for '{title}' in '{language}': "
                f"status={resp.status_code if resp else 'None'}"
            )
            return None

        try:
            data = resp.json()
            total = sum(item.get("views", 0) for item in data.get("items", []))
            logging.debug(
                f"Fetched {total} total pageviews for '{title}' in '{language}'"
            )
            return total
        except Exception as e:
            logging.warning(
                f"Error parsing pageviews response for '{title}' in '{language}': {e}"
            )
            return None


# ...existing code...

if __name__ == "__main__":
    handler = WikiApiRequestHandler()

    # Example: Fetch sitelinks for Wikidata IDs
    ids = ["Q86010314"]
    languages = ["en", "fr"]
    sitelinks = handler.fetch_wikidata_sitelinks(ids, languages)
    print("Sitelinks:", sitelinks)

    # For each title found from the sitelinks, fetch extract and links' Qids
    for entity_id, lang_titles in sitelinks.items():
        for lang, title in lang_titles.items():
            if title:
                extract = handler.fetch_page_extract(title, lang)
                if extract:
                    print(f"Extract ({lang}, {title}):", extract[:80], "...")
                else:
                    print(f"Extract ({lang}, {title}): None")
                qids = handler.fetch_links_qids(title, lang)
                print(f"Links' Qids ({lang}, {title}):", qids)
