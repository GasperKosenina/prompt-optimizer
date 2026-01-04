# Prompt Optimization with DSPy - Project Overview

## Project Goal

**Research Question:**
> Can DSPy optimization algorithms improve LLM prompt performance on customer support tasks, and which optimizer provides the best quality/cost trade-off?

## Background

### Original Vision
The initial goal was to optimize chatbot prompts in production using real user data and feedback. Due to time constraints, this project pivots to simulating prompt optimization on an existing labeled dataset.

### Why DSPy?
DSPy (Declarative Self-improving Python) is a framework for programming language models declaratively. Instead of manually crafting prompts, DSPy:
- Defines signatures (input/output specifications)
- Compiles optimized prompts automatically
- Uses various optimization algorithms (MIPRO v2, BootstrapFewShot, etc.)

## Dataset

**Primary candidate:** [Bitext Customer Support Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset)
- ~27K customer support interactions
- Contains: customer queries, intents, appropriate responses
- Provides ground truth for evaluation

## Project Scope

### Tasks

**Task 1: Intent Classification**
- Input: Customer query
- Output: Intent category (e.g., "REFUND_REQUEST", "ORDER_STATUS")
- Evaluation: Accuracy, F1-score
- Why: Objective evaluation, cheap to compute

**Task 2: Response Generation**
- Input: Customer query (+ optionally classified intent)
- Output: Helpful customer support response
- Evaluation: Semantic similarity, LLM-as-judge
- Why: More realistic, shows generation optimization

### Comparison Dimensions

| Dimension | Options |
|-----------|---------|
| **Optimizers** | BootstrapFewShot, MIPRO v2, (COPRO?) |
| **LLM Backends** | GPT-3.5-turbo, GPT-4-mini, Claude Haiku, Local (Ollama/Llama) |
| **Evaluation Metrics** | Accuracy, Semantic Similarity, LLM-as-Judge |

### Constraints

- **Budget-conscious**: This is an academic project with limited API budget
- **No RAG component**: Keep focus on pure prompt optimization
- **Reproducibility**: All experiments should be reproducible

## Expected Deliverables

1. Working DSPy implementation for both tasks
2. Comparison of optimization algorithms
3. Comparison across LLM backends
4. Cost analysis (tokens used, API calls)
5. Academic writeup with methodology and findings

## Open Questions

- [ ] Final dataset selection - is Bitext the right choice?
- [ ] Which local model to use for budget development?
- [ ] How many optimization iterations are feasible within budget?
- [ ] Should we compare few-shot vs zero-shot baselines?

---

*Document created: 2025-01-03*
*Status: Draft - Discussion Phase*
