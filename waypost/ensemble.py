"""Ensemble and Self-Consistency Engine for Waypost v5 (Spec Section L).

Implements:
- L.1: Self-Consistency for verifiable tasks (code, json, math) using N samples of a single best model.
- L.2: Fan-Out for complex open-ended reasoning across 2-3 distinct model families with mandatory timeout.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .families import get_model_family
from .schemas import ChatRequest, ChatResponse, RequestProfile
from .verify import Verifier

if TYPE_CHECKING:
    from .executor import Executor
    from .router import Candidate

log = logging.getLogger("waypost.ensemble")


# The family table lives in waypost/families.py: the plan ladder needs it
# too, and two copies of "is this the same model" would drift apart.


async def self_consistency_sample(
    executor: Executor,
    req: ChatRequest,
    profile: RequestProfile,
    candidate: Candidate,
    n_samples: int = 3,
    verifier: Verifier | None = None,
) -> ChatResponse | None:
    """Generates N samples from one model with non-zero temperature and picks the verified best (Spec L.1).

    Used for verifiable tasks (code, json, math). Consumes quota of a single provider.
    """
    if n_samples <= 1:
        plan = [candidate]
        return await executor.execute(req, profile, plan)

    sample_req = req.model_copy(update={"temperature": max(0.6, req.temperature)})
    tasks = [
        asyncio.create_task(
            executor.execute(
                sample_req,
                profile,
                [candidate],
            )
        )
        for _ in range(n_samples)
    ]

    results: list[ChatResponse] = []
    done, pending = await asyncio.wait(tasks, timeout=10.0)
    for t in done:
        try:
            res = t.result()
            if res is not None:
                results.append(res)
        except Exception as exc:
            log.debug("self-consistency sample failed: %s", exc)

    # Cancel any lingering tasks
    for p in pending:
        p.cancel()

    if not results:
        return None

    if verifier is None:
        return results[0]

    # Pick first candidate that passes Level-A verification
    for r in results:
        ok, _ = verifier.verify(req, profile, r.model_dump())
        if ok:
            return r

    # If none passed cleanly, return the first
    return results[0]


async def fanout_ensemble(
    executor: Executor,
    req: ChatRequest,
    profile: RequestProfile,
    candidates: list[Candidate],
    max_proposers: int = 3,
    fanout_timeout_ms: int = 4000,
) -> ChatResponse | None:
    """Dispatches request concurrently to 2-3 distinct model families with mandatory timeout (Spec L.2)."""
    # Select up to max_proposers of distinct model families
    selected: list[Candidate] = []
    seen_families: set[str] = set()

    for c in candidates:
        fam = get_model_family(c.offering.model_id)
        if fam not in seen_families:
            seen_families.add(fam)
            selected.append(c)
            if len(selected) >= max_proposers:
                break

    if not selected:
        return None

    if len(selected) == 1:
        return await executor.execute(req, profile, selected)

    tasks = [
        asyncio.create_task(executor.execute(req, profile, [cand])) for cand in selected
    ]

    timeout_s = fanout_timeout_ms / 1000.0
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)

    responses: list[ChatResponse] = []
    for t in done:
        try:
            res = t.result()
            if res is not None:
                responses.append(res)
        except Exception as exc:
            log.debug("fanout branch failed: %s", exc)

    for p in pending:
        p.cancel()

    if not responses:
        return None

    # Return top responsive candidate
    return responses[0]
