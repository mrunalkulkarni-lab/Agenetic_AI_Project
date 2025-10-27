# agents/critic_agent.py
import os
import openai
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

class CriticAgent:
    """
    CriticAgent: takes a draft and returns (improved_draft, critique_summary).
    """

    def __init__(self, model="gpt-4o-mini", max_tokens=1200):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens

    def build_prompt(self, draft: str, research: List[Dict]) -> str:
        src_list = "\n".join([f"[{i+1}] {s['title']} — {s['url']}" for i,s in enumerate(research)])
        prompt = (
            "You are an editor. Given an article draft and the list of research sources, do two things:\n\n"
            "1) Produce a short critique (bulleted): factual errors, missing citations, tone/style, unclear sections.\n"
            "2) Produce an improved version of the article that fixes the issues and adds inline citations.\n\n"
            "Format your answer as:\nCRITIQUE:\n- ...\n\nIMPROVED ARTICLE:\n<article text>\n\n"
            f"Draft:\n{draft}\n\nSources:\n{src_list}\n\n"
            "Be concise in the critique (5-8 bullets) and keep the improved article approx same length.\n"
        )
        return prompt

    def call_llm(self, prompt: str) -> str:
        logger.info("CriticAgent: calling OpenAI")
        r = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.4,
        )
        return r["choices"][0]["message"]["content"].strip()

    def run(self, draft: str, research: List[Dict]) -> Tuple[str, str]:
        prompt = self.build_prompt(draft, research)
        try:
            result = self.call_llm(prompt)
            # naive split: separate CRITIQUE and IMPROVED ARTICLE
            if "IMPROVED ARTICLE" in result:
                critique, improved = result.split("IMPROVED ARTICLE", 1)
                critique = critique.replace("CRITIQUE:", "").strip()
                improved = improved.strip(": \n")
            else:
                # fallback
                critique = "No clear critique extracted"
                improved = result
            logger.info("CriticAgent: finished critique")
            return improved, critique
        except Exception as e:
            logger.exception("CriticAgent failed")
            return draft, "Critic failed — returning original draft"
