# DSPy Optimizer Comparison Guide

## Quick Reference: Which Optimizer Should I Use?

### 🎯 Goal: Better Few-Shot Examples
**→ Use: MIPROv2 (auto="light" or "medium")**
- Focuses on finding optimal demonstrations
- Fast and effective for most cases
- Your current approach ✓

### 📝 Goal: Better Instructions/Prompts
**→ Use: COPRO or MIPROv2 (auto="heavy" + high temperature)**
- COPRO specializes in instruction refinement
- MIPROv2 heavy explores more instruction variations
- Takes longer but finds creative prompt phrasings

### ⚡ Goal: Fast Prototyping
**→ Use: BootstrapFewShot or MIPROv2 (auto="light")**
- Quickest to run (~1-5 minutes)
- Good enough for iteration and development

### 🏆 Goal: Maximum Performance
**→ Use: MIPROv2 (auto="heavy") + Large Dataset**
- Most thorough exploration
- Optimizes both instructions and demonstrations
- Takes 1-2 hours but gives best results

---

## Detailed Optimizer Comparison

| Optimizer | Optimizes | Speed | Best For | Typical Runtime |
|-----------|-----------|-------|----------|-----------------|
| **BootstrapFewShot** | Few-shot examples only | ⚡️ Fast | Quick prototyping | 1-5 min |
| **MIPROv2 Light** | Instructions + few-shot | ⚡️⚡️ Fast | Development, iteration | 5-15 min |
| **MIPROv2 Medium** | Instructions + few-shot | ⏱️ Medium | Production use | 20-40 min |
| **MIPROv2 Heavy** | Instructions + few-shot | 🐌 Slow | Maximum performance | 60+ min |
| **COPRO** | Instructions primarily | ⏱️ Medium | Creative prompt rewriting | 30-60 min |
| **SignatureOptimizer** | Instructions only | ⏱️ Medium | Instruction refinement | 20-40 min |
| **BetterTogether** | Balanced both | ⏱️ Medium | Balanced optimization | 30-50 min |

---

## How to Get More Instruction Optimization

### 1. **Increase MIPROv2 Exploration**

```python
# Current (Conservative)
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    auto="light",  # 3 instruction candidates
    num_threads=4,
)

# More Aggressive (Recommended)
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    auto="medium",  # 6 instruction candidates
    num_threads=4,
)

# Maximum Exploration
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    auto="heavy",  # 10 instruction candidates
    num_threads=4,
)

# Custom Fine-Tuning
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    num_trials=30,
    num_candidates=12,  # More instruction variations
    init_temperature=1.5,  # More creative instructions
    num_threads=4,
)
```

### 2. **Use COPRO for Instruction Focus**

COPRO (Coordinate Prompt Optimization) specializes in refining instructions:

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=math_accuracy_metric,
    breadth=10,  # Number of variations per round
    depth=3,     # Refinement rounds
    init_temperature=1.4,  # Creativity level
)

optimized = optimizer.compile(
    student=create_math_solver(),
    trainset=trainset,
)
```

### 3. **Increase Dataset Size**

More data = better pattern discovery:

```python
# Current
trainset, testset = load_math_problems(n_train=200, n_test=50)

# Better
trainset, testset = load_math_problems(n_train=500, n_test=100)

# Best
trainset, testset = load_math_problems(n_train=1000, n_test=200)
```

### 4. **Use Higher Temperature**

Higher temperature → more creative/diverse instructions:

```python
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    init_temperature=1.5,  # Default: ~1.0
    num_threads=4,
)
```

### 5. **Start with Weaker Baseline**

If your baseline is already good, there's less room for improvement:

```python
# Instead of this well-crafted instruction:
class MathSolver(dspy.Signature):
    """Solve math word problems using step-by-step reasoning."""
    question: str = dspy.InputField(desc="A math word problem requiring arithmetic reasoning")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning showing how to solve the problem")
    answer: str = dspy.OutputField(desc="The final numerical answer (just the number)")

# Try starting with something more generic:
class MathSolver(dspy.Signature):
    """Answer the question."""  # Vague - optimizer must improve this
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()
```

---

## Why Your Optimization Kept the Original Instruction

Looking at your results:

```
Baseline accuracy:    64.0%
Optimized accuracy:   84.0%
Improvement:          +20.0%
```

**The optimizer tested 3 instructions and found the original was best!**

This happened because:

1. ✅ **Your baseline instruction was already good**
   - "Solve math word problems using step-by-step reasoning"
   - Clear, specific, includes "step-by-step" which helps with CoT

2. ✅ **Few-shot examples made the bigger difference**
   - Went from zero-shot → 4 high-quality demonstrations
   - This alone gave you the +20% boost

3. 📊 **Limited exploration with "light" mode**
   - Only 3 instruction candidates tested
   - More would test: 6 (medium), 10 (heavy), 12+ (custom)

---

## Practical Recommendations

### For Your Current Task (Math Solver):

**If you want to see instruction changes:**

```python
# Option A: More aggressive MIPROv2
optimizer = MIPROv2(
    metric=math_accuracy_metric,
    auto="heavy",
    num_threads=4,
)
optimized = optimizer.compile(
    student=create_math_solver(),
    trainset=load_math_problems(n_train=500, n_test=100)[0],  # More data
)

# Option B: COPRO for instruction focus
from dspy.teleprompt import COPRO
optimizer = COPRO(
    metric=math_accuracy_metric,
    breadth=10,
    depth=3,
    init_temperature=1.5,
)
optimized = optimizer.compile(
    student=create_math_solver(),
    trainset=load_math_problems(n_train=500, n_test=100)[0],
)
```

**Time Investment:**
- Light: 10 min → Quick iteration ✓
- Medium: 30 min → Production use ✓✓
- Heavy: 60 min → Research/max performance ✓✓✓
- COPRO: 45 min → Creative instruction discovery ✓✓

---

## Understanding the Optimization Logs

From your run, you saw:

```
Proposed Instructions for Predictor 0:
0: Solve math word problems using step-by-step reasoning.  [YOUR ORIGINAL]
1: Given a scenario where a group of friends has to split expenses...
2: Imagine you are in a high-stakes competition where you must quickly...

Trial 6: Instruction 0 + Few-Shot Set 5 → 91.4% (WINNER!)
```

**What this tells us:**
- Instruction 0 (original) performed best
- The optimizer tried more elaborate instructions (1 & 2) but they hurt performance
- Few-Shot Set 5 was the key ingredient
- **Your initial prompt design was excellent!** 🎉

---

## When Instructions DO Change Dramatically

Instructions change more when:

1. **Baseline is vague/weak**
   ```python
   # Weak baseline
   """Answer the question."""
   
   # Optimizer might produce:
   """Given a mathematical word problem, break it down into steps,
   perform calculations systematically, and provide the final number."""
   ```

2. **Task is complex/unusual**
   - Novel domains where standard phrasing doesn't work
   - Multi-step reasoning tasks
   - Domain-specific jargon needed

3. **Using optimizer focused on instructions**
   - COPRO
   - SignatureOptimizer
   - MIPROv2 with high temperature + heavy mode

4. **Large, diverse training set**
   - 500+ examples
   - Covers edge cases
   - Shows optimizer what works and what doesn't

---

## Bottom Line

**Your current result is actually a success!** 

The optimizer validated that your instruction was already well-designed and added the missing piece (few-shot examples) to boost performance by 20%.

**To see bigger instruction changes**, try:
1. Use `auto="heavy"` or COPRO
2. Use 500-1000 training examples
3. Increase temperature to 1.5
4. Start with a deliberately vague baseline

But remember: **The goal is better performance, not necessarily different instructions**. If the original works best, that's a win! 🎯

