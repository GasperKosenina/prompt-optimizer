# Project Decisions Log

This document tracks key decisions made during project planning.

## Session: 2025-01-03

### Context
- Academic thesis project
- 1-2 week timeline
- Beginner Python level
- Budget constraints

### Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dataset** | Bitext Customer Support (27K) | Labeled data for classification + generation |
| **Dataset size** | Start with 100-200, scale up | Budget-conscious, fast iteration |
| **Tasks** | Intent Classification + Response Generation | Both required for thesis |
| **Optimizers** | BootstrapFewShot + MIPRO v2 | Compare simple vs advanced |
| **LLM backends** | Multiple (GPT-3.5, Claude Haiku, etc.) | Academic comparison |
| **Evaluation (classification)** | Accuracy, F1-score | Objective metrics |
| **Evaluation (generation)** | Semantic similarity | Budget-friendly |
| **RAG component** | No | Keep focused on core problem |
| **Complexity features** | Ablation studies + Prompt analysis | Adds academic depth |

### What We're NOT Doing
- Cross-model transfer analysis (cut for time)
- LLM-as-judge evaluation (expensive)
- Multiple evaluation metrics beyond core ones
- Production deployment

### Risk Mitigation
- If time runs short: Focus only on classification + BootstrapFewShot
- If budget runs out: Use only local models (Ollama)
- If MIPRO v2 too expensive: Use only "light" mode or skip

### Open Questions (To Resolve)
- [ ] Verify Bitext dataset structure and column names
- [ ] Decide on specific intent categories to use
- [ ] Set up LLM API access
- [ ] Choose local model for development (if using)

---

## Documents Created

1. `01-project-overview.md` - High-level project description
2. `02-technical-architecture.md` - System design and structure
3. `03-implementation-plan.md` - Day-by-day timeline
4. `04-quickstart-guide.md` - Beginner-friendly code examples
