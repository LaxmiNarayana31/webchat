from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend.app.agents.router_agent import router_agent
from backend.app.agents.supervisor_agent import supervisor_agent
from backend.app.core.errors import WebChatException
from backend.app.core.logging import logger
from backend.app.dtos.agent_dto import (
    AgentCapabilityDto,
    AgentQueryRequestDto,
    AgentQueryResponseDto,
    AgentRole,
    AgentRouteDecisionDto,
    AgentTraceDto,
    AgentTraceStepDto,
)
from backend.app.dtos.chat_dto import CitationItemDto

router = APIRouter(prefix="/agent", tags=["Multi-Agent System"])


@router.post("/query", response_model=AgentQueryResponseDto)
async def query_agent_system(req: AgentQueryRequestDto):
    """Executes multi-agent conversational RAG workflow with full execution tracing."""
    try:
        result = await supervisor_agent.run_workflow_async(
            query=req.query,
            url=req.url,
            document_content=req.document_content,
            chat_history=[msg.model_dump() for msg in req.chat_history] if req.chat_history else [],
            selected_model=req.selected_model,
            user_id=req.user_id,
        )

        trace_data = None
        if req.include_trace and result.trace:
            trace_data = AgentTraceDto(
                trace_id=result.trace.trace_id,
                query=result.trace.query,
                steps=[
                    AgentTraceStepDto(
                        step_index=step.step_index,
                        agent_role=step.agent_role.value if hasattr(step.agent_role, "value") else str(step.agent_role),
                        action=step.action,
                        input_summary=step.input_summary,
                        output_summary=step.output_summary,
                        duration_ms=step.duration_ms,
                        timestamp=step.timestamp,
                        metadata=step.metadata,
                    )
                    for step in result.trace.steps
                ],
                total_duration_ms=result.trace.total_duration_ms,
                agent_handoffs=result.trace.agent_handoffs,
                started_at=result.trace.started_at,
            )

        citations_dtos = [
            CitationItemDto(
                chunk_index=c.get("chunk_index", idx),
                content=c.get("content", ""),
                title=c.get("title", ""),
                url=c.get("url", ""),
            )
            for idx, c in enumerate(result.citations)
        ]

        return AgentQueryResponseDto(
            answer=result.answer,
            route=result.route,
            citations=citations_dtos,
            model_used=result.model_used,
            provider=result.provider,
            fallback_triggered=result.fallback_triggered,
            latency_sec=result.latency_sec,
            trace=trace_data,
            telemetry=result.telemetry,
        )
    except WebChatException as wce:
        logger.warning(f"Agent query handled exception: {wce.message}")
        raise HTTPException(
            status_code=wce.status_code,
            detail=wce.details if wce.details else wce.message,
        )
    except Exception as e:
        logger.error(f"Unexpected agent query error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/route", response_model=AgentRouteDecisionDto)
async def classify_route(req: dict[str, Any]):
    """Classifies user query intent using AI-driven routing with sub-50ms fallback."""
    try:
        query = req.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

        has_document = bool(req.get("has_document", False))
        doc_title = req.get("doc_title", "")

        decision = await router_agent.classify_intent_async(
            query=query,
            has_document=has_document,
            doc_title=doc_title,
        )

        return AgentRouteDecisionDto(
            route=decision.get("route", "DOCUMENT_RAG"),
            confidence=float(decision.get("confidence", 1.0)),
            reasoning=decision.get("reasoning", ""),
            latency_ms=float(decision.get("latency_ms", 0.0)),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected route classification error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/capabilities", response_model=list[AgentCapabilityDto])
def list_capabilities():
    """Lists specialized agents registered in the multi-agent system and their capabilities."""
    return [
        AgentCapabilityDto(
            role=AgentRole.SUPERVISOR.value,
            name="Supervisor Agent",
            description="Central orchestrator delegating tasks, managing handoffs, and aggregating traces.",
            autonomous=True,
            tools=["task_delegation", "agent_handoff", "trace_aggregation"],
        ),
        AgentCapabilityDto(
            role=AgentRole.ROUTER.value,
            name="Router Agent",
            description="AI-driven intent classifier with sub-50ms heuristic fast-path fallback.",
            autonomous=True,
            tools=["llm_classification", "regex_fastpath"],
        ),
        AgentCapabilityDto(
            role=AgentRole.PLANNER.value,
            name="Planner Agent",
            description="Query decomposition for multi-hop retrieval and iterative query rewriting.",
            autonomous=True,
            tools=["query_decomposition", "query_rewriting"],
        ),
        AgentCapabilityDto(
            role=AgentRole.RESEARCHER.value,
            name="Research Agent",
            description="Hybrid vector & keyword retriever with autonomous web search fallback.",
            autonomous=True,
            tools=["dense_search", "bm25_sparse_search", "flashrank_rerank", "duckduckgo_search", "wikipedia_search"],
        ),
        AgentCapabilityDto(
            role=AgentRole.CRITIC.value,
            name="Critic Agent",
            description="CRAG document relevance grader and Self-RAG hallucination verifier.",
            autonomous=True,
            tools=["crag_document_grading", "self_rag_hallucination_check", "critique_generation"],
        ),
        AgentCapabilityDto(
            role=AgentRole.SYNTHESIZER.value,
            name="Synthesis Agent",
            description="Merges verified chunks, citations, and conversation memories into grounded prompt.",
            autonomous=True,
            tools=["context_isolation", "prompt_synthesis", "memory_integration"],
        ),
    ]
