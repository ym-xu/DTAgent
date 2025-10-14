"""命令行入口：单文档问答调试工具。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .loaders import build_observer_store, build_resources_from_index
from .observer import Observer, build_llm_image_analyzer
from .orchestrator import AgentConfig, AgentOrchestrator
from .planner import Planner
from .reasoner import LLMCallable, Reasoner, ReasonerLLMConfig
from .schemas import ReasonerAnswer, StrategyPlan
from .retriever import RetrieverLLMCallable, RetrieverManager
from .router import QuestionRouter, RouterLLMCallable
from .strategy_planner import RetrievalStrategyPlanner, StrategyLLMCallable


def run_question(
    doc_dir: Path | str,
    question: str,
    *,
    max_iterations: int = 3,
    use_llm: bool = True,
    llm_backend: str = "gpt",
    llm_model: str = "gpt-4o-mini",
    llm_callable: Optional[LLMCallable] = None,
    image_analyzer=None,
    strategy_llm_callable: Optional[StrategyLLMCallable] = None,
    router_llm_callable: Optional[RouterLLMCallable] = None,
    retriever_llm_callable: Optional[RetrieverLLMCallable] = None,
) -> Tuple[ReasonerAnswer, AgentOrchestrator, List[StrategyPlan]]:
    """在给定文档索引目录上运行问题，返回答案与完整 orchestrator。"""

    base = Path(doc_dir)
    resources, graph = build_resources_from_index(base)
    store = build_observer_store(base)

    reasoner = Reasoner(
        use_llm=use_llm,
        llm_config=ReasonerLLMConfig(backend=llm_backend, model=llm_model),
        llm_callable=llm_callable,
    )

    orchestrator = AgentOrchestrator(
        router=QuestionRouter(llm_callable=router_llm_callable),
        strategy_planner=RetrievalStrategyPlanner(llm_callable=strategy_llm_callable),
        planner=Planner(graph=graph),
        retriever_manager=RetrieverManager(resources, llm_callable=retriever_llm_callable),
        observer=Observer(store=store, image_analyzer=image_analyzer),
        reasoner=reasoner,
        config=AgentConfig(max_iterations=max_iterations),
    )

    plans_emitted: List[StrategyPlan] = []

    def _on_plan(plan: StrategyPlan) -> None:
        plans_emitted.append(plan)
        label = "Strategy Plan" if len(plans_emitted) == 1 else f"Strategy Plan (iteration {len(plans_emitted)})"
        print(f"\n{label}:")
        print(_format_strategy(plan))

    answer = orchestrator.run_with_callback(question, on_plan=_on_plan)
    return answer, orchestrator, plans_emitted


def _format_strategy(plan) -> str:
    header = f"  - confidence={plan.confidence:.2f}, strategy={plan.strategy.value}"
    if getattr(plan, "thinking", None):
        header += f" | thinking: {plan.thinking}"
    if getattr(plan, "retrieval_keys", None):
        header += f" | keys: {', '.join(plan.retrieval_keys)}"
    lines = [header]
    for stage in getattr(plan, "stages", []) or []:
        lines.append(
            f"    stage={stage.stage}, methods={stage.methods}, k_pages={stage.k_pages}, "
            f"k_nodes={stage.k_nodes}, page_window={stage.page_window}"
        )
    for step in plan.steps:
        lines.append(f"    * {step.describe()}")
    return "\n".join(lines)


def _format_router_history(orchestrator: AgentOrchestrator) -> str:
    if not orchestrator.memory.router_history:
        return "  <no router decision>"
    lines = []
    for idx, decision in enumerate(orchestrator.memory.router_history, start=1):
        signals = decision.signals
        risk = decision.risk
        constraints = decision.constraints
        lines.append(
            f"  [{idx}] type={decision.query_type}, confidence={decision.confidence:.2f}, "
            f"signals={{page_hint={signals.page_hint}, figure_hint={signals.figure_hint}, "
            f"table_hint={signals.table_hint}, objects={signals.objects}, units={signals.units}, "
            f"years={signals.years}, operations={signals.operations}, expected={signals.expected_format}, "
            f"section_cues={signals.section_cues}, keywords={signals.keywords}, scope={signals.objects_scope}}}, "
            f"risk={{ambiguity={risk.ambiguity:.2f}, need_visual={risk.need_visual}, "
            f"need_table={risk.need_table}, need_chart={risk.need_chart}}}, "
            f"constraints={{allow_unanswerable={constraints.allow_unanswerable}, must_cite={constraints.must_cite}}}"
        )
    return "\n".join(lines)


def _format_hits(orchestrator: AgentOrchestrator) -> str:
    lines = []
    for key, hits in orchestrator.memory.retrieval_cache.items():
        lines.append(f"  Step {key}:")
        for hit in hits[:10]:
            lines.append(f"    - {hit.node_id} (score={hit.score:.3f}, tool={hit.tool})")
    return "\n".join(lines) if lines else "  <no hits>"


def _format_observations(orchestrator: AgentOrchestrator) -> str:
    if not orchestrator.memory.observations:
        return "  <no observations>"
    lines = []
    for node_id, obs in orchestrator.memory.observations.items():
        text = obs.payload.get("text") if isinstance(obs.payload, dict) else None
        snippet = (text[:100] + "...") if text and len(text) > 100 else (text or "")
        lines.append(f"  - {node_id} ({obs.modality}): {snippet}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agents_v2 QA on a single document index")
    parser.add_argument("--doc-dir", required=True, help="Path to document index directory")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum planner-reasoner iterations")
    parser.add_argument("--llm-backend", default="gpt", help="LLM backend identifier (gpt/qwen)")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--use-image-llm", action="store_true", help="Enable LLM-based image summarization")
    parser.add_argument("--image-llm-backend", default=None, help="Image analyzer backend (defaults to --llm-backend)")
    parser.add_argument("--image-llm-model", default=None, help="Image analyzer model (defaults to --llm-model)")
    args = parser.parse_args()

    image_analyzer = None
    if args.use_image_llm:
        backend = args.image_llm_backend or args.llm_backend
        model = args.image_llm_model or args.llm_model
        image_root = Path(args.doc_dir)
        image_analyzer = build_llm_image_analyzer(backend=backend, model=model, image_root=image_root)

    answer, orchestrator, plans = run_question(
        args.doc_dir,
        args.question,
        max_iterations=args.max_iterations,
        use_llm=True,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        image_analyzer=image_analyzer,
    )

    print(f"Question: {args.question}")
    print("\nRouter Decision:")
    print(_format_router_history(orchestrator))
    if not plans:
        print("\nStrategy Plan:")
        for plan in orchestrator.memory.strategy_history:
            print(_format_strategy(plan))

    print("\nRetriever Hits:")
    print(_format_hits(orchestrator))

    print("\nObservations:")
    print(_format_observations(orchestrator))

    print("\nFinal Answer:")
    if answer.thinking:
        print(f"  Thinking: {answer.thinking}")
    print(f"  Answer: {answer.answer}")
    print(f"  Confidence: {answer.confidence:.2f}")
    if answer.support_nodes:
        print(f"  Support Nodes: {', '.join(answer.support_nodes)}")
    if answer.reasoning_trace:
        print(f"  Reasoning: {answer.reasoning_trace[0]}")


if __name__ == "__main__":
    main()
