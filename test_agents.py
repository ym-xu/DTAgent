from pathlib import Path
from pprint import pprint
from src.agents_v2.cli import run_question

doc_dir = Path("./../../../data/users/yiming/dtagent/MinerU_25_MMLB/nova_y70/indexes/")
question = "Under the pro mode to shoot, what is the function of the icon on right hand side of the icon that select a focus mode?"
question = "In the demostration of how to use a Knuckle to Take a Scrolling Screenshot, what buildings appear in the first picture?"

doc_dir = Path("./../../../data/users/yiming/dtagent/MinerU_25_MMLB/2305.13186v3/indexes/")
question = "How many reasoning steps are involved in the figure 1 in the paper?"

doc_dir = Path("./../../../data/users/yiming/dtagent/MinerU_25_MMLB/8e7c4cb542ad160f80fb3d795ada35d8/indexes/")
question = "What percentage of land area was rezoned in the Bronx from 2003-2007?"

answer, orchestrator, plans = run_question(
    doc_dir,
    question,
    max_iterations=3,
    use_llm=True,
    llm_backend="gpt",
    llm_model="gpt-4o-mini",
)

# obs = orchestrator.memory.observations.get("vlm.answer:vlm_answer_s1")
# pprint(obs.payload)
