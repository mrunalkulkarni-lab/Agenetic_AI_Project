# agents/research_agent.py
import wikipedia
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    ResearchAgent: given a topic, returns a list of summarized sources:
    [{'title':..., 'url':..., 'summary':...}, ...]
    """

    def __init__(self, max_sources: int = 5):
        self.max_sources = max_sources

    def search_wikipedia(self, topic: str) -> List[str]:
        try:
            results = wikipedia.search(topic, results=self.max_sources)
            logger.info(f"ResearchAgent: wiki search results: {results}")
            return results
        except Exception as e:
            logger.exception("ResearchAgent: search_wikipedia failed")
            return []

    def summarize_page(self, title: str) -> Dict:
        try:
            page = wikipedia.page(title)
            summary = wikipedia.summary(title, sentences=3)
            return {"title": page.title, "url": page.url, "summary": summary}
        except Exception as e:
            logger.warning(f"ResearchAgent: failed to load page {title}: {e}")
            return {"title": title, "url": "", "summary": ""}

    def run(self, topic: str) -> List[Dict]:
        """Main entry point"""
        titles = self.search_wikipedia(topic)
        sources = []
        for t in titles:
            if len(sources) >= self.max_sources:
                break
            s = self.summarize_page(t)
            if s["summary"]:
                sources.append(s)
        logger.info(f"ResearchAgent: returning {len(sources)} sources")
        return sources
