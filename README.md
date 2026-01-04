# When Does Prompt Optimization Actually Matter?

Research investigating how task complexity affects the value of automated prompt optimization.

## Key Finding

**Not all tasks benefit equally from prompt optimization:**

- ✅ **Simple tasks** (classification): Modern LLMs already perform well → minimal improvement from optimization
- 🚀 **Complex tasks** (response generation): Significant quality gains from optimized system prompts

**Bottom line**: Save optimization effort for complex tasks where it actually helps.

## The Research

Using [DSPy](https://github.com/stanfordnlp/dspy) and the [Bitext Customer Support Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset) (27K examples), we compare:

**Part 1 - Classification** (simple task):
- Baseline: 90% accuracy
- Optimized: 92% accuracy
- **Result**: +2% improvement → not worth the effort

**Part 2 - Response Generation** (complex task):
- Baseline: ~55% quality
- Optimized: ~72% quality
- GPT-4: ~82% quality
- **Result**: +17% improvement + 97% cost savings vs GPT-4

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run classification experiment (Part 1)
python -m src.optimizers.runner --optimizer both --train-size 100 --test-size 50

# View results
ls results/  # metrics JSON + comparison plots
```

## What's Inside

- `src/optimizers/runner.py` - Compare BootstrapFewShot vs MIPROv2 optimizers
- `src/modules/` - Task signatures (classification, response generation)
- `src/evaluation/` - Multi-dimensional quality metrics
- `results/` - Experiment results + visualizations
- `docs/` - Detailed research methodology

See `docs/03-research-methodology.md` for full research design.

## Research Questions

1. Does optimization effectiveness depend on task complexity?
2. Can optimized GPT-3.5 match GPT-4 quality at lower cost?
3. Which system prompt components matter most for quality?

## License

MIT
