# agents/writer_agent.py
import os
import openai
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class WriterAgent:
    """
    WriterAgent: given research snippets (list of dicts), returns a draft article.
    Uses OpenAI ChatCompletion (gpt-3.5/4 style). You can swap provider easily.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1500):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens

    def build_research_prompt(self, topic: str, sources: List[Dict]) -> str:
        src_text = "\n\n".join([f"Title: {s['title']}\nURL: {s['url']}\nSummary: {s['summary']}" for s in sources])
        prompt = (
            f"Write a 700-1200 word informative article on: {topic}.\n\n"
            "Use the research snippets below and include inline citations like [1], [2] matching the sources.\n"
            "Make the tone professional and accessible. Add a short 2-3 sentence intro and a 1-paragraph conclusion.\n\n"
            f"Research sources:\n{src_text}\n\n"
            "Return only the article text (no json)."
        )
        return prompt

    def call_llm(self, prompt: str) -> str:
        logger.info("WriterAgent: calling OpenAI")
        r = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.6,
        )
        text = r["choices"][0]["message"]["content"].strip()
        return text

    def run(self, topic: str, research: List[Dict]) -> str:
        prompt = self.build_research_prompt(topic, research)
        try:
            draft = self.call_llm(prompt)
            logger.info(f"WriterAgent: produced draft of length {len(draft)}")
            return draft
        except Exception as e:
            logger.exception("WriterAgent: LLM call failed")
            # fallback simple assembly
            fallback = "Fallback draft:\n\n" + "\n\n".join([s["summary"] for s in research])
            return fallback
