# orchestrator.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import logging
from pathlib import Path
from datetime import datetime
from agents.research_agent import ResearchAgent
from agents.writer_agent import WriterAgent
from agents.critic_agent import CriticAgent
import re
import json
from tqdm import tqdm

# --- Setup logging
root = Path(__file__).parent
out_dir = root / "outputs"
logs_dir = out_dir / "logs"
out_dir.mkdir(exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)

log_file = logs_dir / "agentic.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator")

# --- Helpers
def slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','_', s.lower()).strip('_')

def save_output(topic: str, final_text: str, critique: str):
    filename = out_dir / f"final_article_{slugify(topic)}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\n\n")
        f.write("=== FINAL ARTICLE ===\n\n")
        f.write(final_text + "\n\n")
        f.write("=== CRITIQUE SUMMARY ===\n\n")
        f.write(critique + "\n")
    logger.info(f"Saved final article to {filename}")
    return filename

# --- Pipeline
def run_pipeline(topic: str, iterative: bool = False):
    logger.info(f"=== Starting pipeline for topic: {topic} ===")
    # Instantiate agents
    research_agent = ResearchAgent(max_sources=5)
    writer_agent = WriterAgent(model=os.getenv("LLM_MODEL","gpt-3.5-turbo"))
    critic_agent = CriticAgent(model=os.getenv("LLM_MODEL","gpt-3.5-turbo"))

    # 1: Research
    research = research_agent.run(topic)
    logger.info(f"Research returned {len(research)} items.")

    # 2: Write
    draft = writer_agent.run(topic, research)

    # 3: Critic
    improved, critique = critic_agent.run(draft, research)

    # Optional iterative loop (critic -> writer)
    if iterative:
        logger.info("Starting iterative rewrite (Critic -> Writer)")
        draft2 = writer_agent.run(topic + " (revise based on critic)", research + [{"title":"critic","url":"","summary":critique}])
        improved2, critique2 = critic_agent.run(draft2, research)
        final_article = improved2
        final_critique = critique + "\n\nITERATION:\n" + critique2
    else:
        final_article = improved
        final_critique = critique

    saved = save_output(topic, final_article, final_critique)
    # Save a small json log with timestamps
    log_json = {
        "topic": topic,
        "timestamp": datetime.utcnow().isoformat(),
        "final_file": str(saved),
        "research_count": len(research),
        "draft_length": len(draft),
        "final_length": len(final_article)
    }
    with open(logs_dir / f"log_{slugify(topic)}.json", "w", encoding="utf-8") as f:
        json.dump(log_json, f, indent=2)
    logger.info(f"Pipeline finished for topic: {topic}")
    return saved

if __name__ == "__main__":
    # Example two topics (requirement: test system on two topics)
    topics = [
        "How retrieval-augmented generation improves factual accuracy in language models",
        "Ethical challenges in autonomous AI agents"
    ]
    for t in topics:
        try:
            run_pipeline(t, iterative=False)
        except Exception as e:
            logger.exception(f"Run failed for topic: {t}")
