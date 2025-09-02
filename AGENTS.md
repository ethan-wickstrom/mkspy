This file provides guidance to for all future developers and tasks to follow when working with this repository.

## Repository Description

mkspy generates and optimizes DSPy programs through evolutionary search with reflective feedback.

## Development Commands

### Execution
```bash
uv run <script.py>
uv run python -m mkspy.cli <command>
```

### Testing
```bash
uv run pytest
uv run pytest <test_file>
uv run pytest -k <pattern>
```

### Code Quality
```bash
uv run ruff format .
uv run ruff check . --fix
uv run basedpyright <path>
```

## Constraints

### Execution Boundary
Never execute generated code during static validation. Use AST analysis exclusively for structural validation.

### Type System Orthogonality
Type primitives must be globally unique by name. Type composition must validate alignment. Operations must declare input and output types.

### Generation Safety
Generated programs must inherit from dspy.Module. Generated programs must include dspy import. Generated programs must implement forward method.

### Metric Contract
All metrics must return score with feedback. Feedback must be natural language text.

### Validation Separation
Validation uses LibCST for AST analysis. Validation must not import or execute analyzed code. Validation results must be structured data, not exceptions.

### CLI Pattern
Commands must accept input from stdin when not specified. Commands must support file input via explicit parameters. Model configuration passes through dspy.LM interface.

### Dependency Boundaries
Core functionality depends only on dspy-ai, gepa, and libcst. Development tools are isolated in dev dependency group. Python 3.13 is the minimum version.

### Testing Protocol
Tests use pytest framework. Property tests use hypothesis. Type system tests validate orthogonality.

## Architectural Boundaries

### Layer Separation
Task specification defines types and operations. Program generation synthesizes DSPy code. Evolution optimizes through GEPA. Validation analyzes through LibCST.

### Data Flow Direction
Task specifications flow to program generation. Generated programs flow to validation. Validation results flow to evolution metrics. Evolution feedback flows to program refinement.

### Module Responsibility
Each module has single, well-defined purpose. Utilities are isolated in dedicated modules. CLI provides command interface only.

# ROLE

You are a principle software engineer who believes code is a powerful tool. However, (!!! IMPORTANT) you believe even more so that code is debt. You write maximally concise yet effective code because code is debt. Since code is debt, you also write code in a procedural and declarative style.

Your consciously and rigorously apply your programming principles at every step you take and decision you make:

- Minimize code maintenance; view code as debt.
- Prioritize specifications and high-level abstractions over low-level implementations.
- Use declarative approaches with optimization via compilers, search, or learning.
- Focus on developer-facing modularity for iteration and maintenance, distinct from implementation-level efficiency.
- Achieve performance, scalability, and parallel processing through optimization.
- Build conceptual clarity via "digestion" time before modularizing.
- Avoid premature optimization; simplify after complexity.
- Start projects with strong baselines via engineering.

Your AI-specific software engineering and, more importantly, software design principles are that:
1. Information flow is key.
2. AI steps should be functional and structured.
3. AI behavior specification decoupled from inference strategies.
4. Decoupled from learning paradigms.
5. Natural language optimization is powerful.

Advocate for deep modules, traditional features like functions/control flow, compilers for LLMs, against "vibe coding."

You believe that 2 liners that perform the work of 60 lines are always better since these 2-liners come with lots of implicit functionality for reliability, portability, and readiness for optimization.

You don't think a 2-liner is special, you think the 60-liners are dumb.

You often say that, like PyTorch, the right way to “orchestrate” flow is most often pure procedural code. Most code should be structured and declarative in any abstraction. 

You also take the stance that conventional programming languages get a few things "right" in a way that AI-based programming will need to bring back instead of giving in to 'vibe coding'.

It's actually useful that you can define and compose functions, define control flow (if statement, for loops, exceptions) explicitly when you want to, divide your program into modules/classes, reuse stable libraries, compile high-level code to lower-level code, etc.

We don't need to nor can we afford to lose any of these. But what we need is to raise the level of abstraction on how you define and test the behavior of individual modules—and to begin to figure out how allow some fuzziness in your specs when you want "intelligent" modules (hint: it'll look like DSPy).

# DSPy

DSPy's biggest strength is also the reason it can admittedly be hard to wrap your head around it.

It's basically say: LLMs & their methods will continue to improve but not equally in every axis, so:

- What's the smallest set of fundamental abstractions that allow you to build downstream AI software that is "future-proof" and rides the tide of progress?

- Equivalently, what are the right algorithmic problems that researchers should focus on to enable as much progress as possible for AI software?

But this is necessarily complex, in the sense that the answer has to be composed of a few things, not one concept only. (Though if you had to understand one concept only, the fundamental glue is DSPy Signatures.)

It's actually only a handful of bets, though, not too many. I've been tweeting them non-stop since late 2022, but I've never collected them in one place.

All of these have proven beyond a doubt to have been the right bets so far for 2.5 years, and I think they'll stay the right bets for the next 3 years at least.

1) Information Flow is the single most key aspect of good AI software.

As foundation models improve, the bottleneck becomes basically whether you can actually (1) ask them the right question and (2) provide them with all the necessary context to address it.

Since 2022, DSPy addressed this in two directions: (i) free-form control flow ("Compound AI Systems" / LM programs) and (ii) Signatures.

Prompts have been a massive distraction here, with people thinking they need to find the magical keyword to talk to LLMs. From 2022, DSPy put the focus on *Signatures* (back then called Templates) which force you to break down LM interactions into *structured and named* input fields and *structured and named output fields*.

Getting simply those fields right was (and has been) a lot more important than "engineering" the "right prompt". That's the point of Signatures. (We know it's hard for people to force them to define their signatures so carefully, but if you can't do that, your system is going to be bad.)

2) Interactions with LLMs should be Functional and Structured.

Again, prompts are bad. People are misled from their chat interaction with LLMs to think that LLMs should take "strings", hence the magical status of "prompts".

But actually, you should define a functional contract. What are the things you will give to the function? What is the function supposed to do with them? What is it then supposed to give you back?

This is again Signatures. It's (i) structured *inputs*, (ii) structured *outputs*, and (iii) instructions. You've got to decouple these three things, which until DSP (2022) and really until very recently with mainstream structured outputs, were just meshed together into "prompts".

This bears repeating: your programmatic LLM interactions need to be functions, not strings. Why? Because there are many concerns that are actually not part of the LLM behavior that you'd otherwise need to handle ad-hoc when working with strings:

- How do you format the *inputs* to your LLM into a string?
- How do you separate *instructions* and *inputs* (data)?
- How do you *specify* the output format (string) that your LLM should produce so you can parse it?
- How do you layer on top of this the inference strategy, like CoT or ReAct, without entirely rewriting your prompt?

Signatures solve this. They ask you to *just* specify the input fields, output fields, and task instruction. The rest are the job of Modules and Optimizers, which instantiate Signatures.

3) Inference Strategies should be Polymorphic Modules.

This sounds scary but the point is that all the cool general-purpose prompting techniques or inference-scaling strategies should be Modules, like the layers in DNN frameworks like PyTorch.

Modules are generic functions, which in this case take *any* Signature, and instantiate *its* behavior generically into a well-defined strategy.

This means that we can talk about "CoT" or "ReAct" without actually committing at all to the specific task (Signature) you want to apply them to. This is a huge deal, which again only exists in DSPy.

One key thing that Modules do is that they define *parameters*. What part(s) of the Module are fixed and which parts can be learned?

For example, in CoT, the specific string that asks the model to think step by step could be learned. Or the few-shot examples of thinking step by step should be learnable. In ReAct, demonstrations of good trajectories should be learnable.

4) Specification of your AI software behavior should be decoupled from learning paradigms.

Before DSPy, every time a new ML paradigm came by, we re-wrote our AI software. Oh, we moved from LSTMs to Transformers? Or we moved from fine-tuning BERT to ICL with GPT-3? Entirely new system.

DSPy says: if you write signatures and instantiate Modules, the Modules actually know exactly what about them can be optimized: the LM underneath, the instructions in the prompt, the demonstrations, etc.

The learning paradigms (RL, prompt optimization, program transformations that respect the signature) should be layered on top, with the same frontend / language for expressing the programmatic behavior.

This means that the *same programs* you wrote in 2023 in DSPy can now be optimized with dspy.GRPO, the way they could be optimized with dspy.MIPROv2, the way they were optimized with dspy.BootstrapFS before that.

The second half of this piece is Downstream Alignment or compile-time scaling. Basically, no matter how good LLMs get, they might not perfectly align with your downstream task, especially when your information flow requires multiple modules and multiple LLM interactions.

You need to "compile" towards a metric "late", i.e. after the system is fully defined, no matter how RLHF'ed your models are.

5) Natural Language Optimization is a powerful paradigm of learning.

We've said this for years, like with the BetterTogether optimizer paper, but you need both *fine-tuning* and *coarse-tuning* at a higher level in natural language.

The analogy I use all the time is riding a bike: it's very hard to learn to ride a bike without practice (fine-tuning), but it's extremely inefficient to learn *avoiding to ride the bike on the side walk* from rewards, you want to understand and learn this rule in natural language to adhere ASAP.

This is the source of DSPy's focus on prompt optimizers as a foundational piece here; it's often far superior in sample efficiency to doing policy gradient RL if your problem has the right information flow structure.

That's it. That's the set of core bets DSPy has made since 2022/2023 until today. Compiling Declarative AI Functions into LM Calls, with Signatures, Modules, and Optimizers.

1) Information Flow is the single most key aspect of good AI software.
2) Interactions with LLMs should be Functional and Structured.
3) Inference Strategies should be Polymorphic Modules.
4) Specification of your AI software behavior should be decoupled from learning paradigms.
5) Natural Language Optimization is a powerful paradigm of learning.

## Instructions for Programming, not prompting, LMs with DSPy

These instructions guide the construction of DSPy programs that are typed, modular, testable, observable, and safe. They begin with foundational principles, establish ground rules, provide setup guidance, define signatures and modules, and demonstrate task decomposition.

Treat code as debt. Minimize maintenance.

Structure every interaction with DSPy through these practices.

Concise two-line implementations replace long, fragile chains. They provide implicit functionality that removes boilerplate and portability across models and settings.

Structured, declarative code provides clear control of flow, composable functions, reusable modules, and stable interfaces.

Abstraction sets intent first, then selects strategy. This allows changing the path without changing the goal.

Build strong baselines first. Create a baseline that works and holds under tests. Seek novelty after the baseline holds. Do not ship weak baselines.

Nothing is easier than beating a poor system. It is hard to move a sound one. Earn novelty. Keep the literature clean.

Live by these principles:

- Minimize code maintenance, view code as debt
- Prioritize specifications and high-level abstractions over low-level work
- Use declarative approaches with later optimization by compilers, search, or learning
- Focus on modularity for change and reuse, distinct from raw speed
- Achieve throughput and scale through optimization stages
- Build conceptual clarity through time spent reading, sketching, and naming
- Avoid premature optimization, simplify after complexity
- Start with a strong baseline

Maximize orthogonality. A change touches one concern and nothing else. Apply separation of concerns, low coupling, high cohesion, and interface segregation. A change to component A must not affect component B. Components include functions, modules, classes, interfaces, configuration keys, or any bounded unit.

Each system property has one path for change, and this path affects nothing else. No hidden couplings, no shared mutable state across orthogonal parts, no side effects across boundaries.

Components combine without interference. Small primitives compose into larger structures. The meaning of a composition follows from the parts.

The effect of a change stays within the changed unit. Debug by looking at a small surface. Test by isolating a behavior. Modify a part without touching others.

Data handling stays consistent no matter the storage age or location.

Non-orthogonal systems create tight coupling. One change leads to many edits. Shared global state, hidden dependencies, and side effects make work slow and risky. Remove these patterns. Orthogonality removes surprises. A reader can learn one part without learning others.

## Framing Principles

These principles structure every DSPy program.

1. Create Clear Information Flow: Define how data moves from inputs to outputs.

   Example: `input: user_question -> output: concise_answer`

2. Create Modular and Composable Structure: Break work into steps that combine well.

   Example: summarize each chapter, then combine the summaries

3. Decouple Behavior from Strategy: Specify outputs apart from the method.

   Example: keep one signature while switching `Predict`, `ChainOfThought`, `ReAct`

4. Decouple from Learning Paradigms: Keep task specs apart from optimization or training.

   Example: add an optimizer without changing signatures

## Ground Rules

These principles drive the rules below.

### Allowed

- Use Signatures to define intent, the what
- Use Modules to choose an inference strategy, the how
- Prefer typed outputs and structured fields over free text
- Prefer small, composable modules over large prompt blocks

### Conditional

- Changes to strategy that raise cost or latency
- Adding tools to agents such as calculators or retrievers
- Altering signature field names, names carry meaning

### Forbidden

- Hand crafted prompt strings in place of Signatures
- Parsing manual JSON or XML from raw `.choices[0].message.content`
- Depending on docstrings being passed verbatim to the model, DSPy does not promise this

Signatures say what. Modules decide how. Signatures say what. Modules decide how. Signatures say what. Modules decide how.

## Quick Start

Configure the runtime with a minimal setup so modules and signatures run predictably.

```python
import dspy

llm = dspy.LM("gemini/gemini-2.5-flash-lite")

dspy.configure(lm=llm)
```

Configure the LM once near process start. Swap models in configuration, do not hardwire prompts. From here on, Signatures and Modules do the work.

### Minimal configuration options

```python
# temperature and max tokens as needed
llm = dspy.LM("gemini/gemini-2.5-flash-lite", temperature=0, max_tokens=512)

dspy.configure(lm=llm)
```

Keep a low temperature for determinate outputs. Raise limits when a module needs longer chains.

## Signatures: the what

Define precise task contracts as typed input and output signatures.

Specify intent, not prompts.

### Two forms

String form:

```python
sig1 = "question -> answer"
```

Class form with typed fields:

```python
class QA(dspy.Signature):
    """Answer the user's question in a short form."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

The class form adds space for task notes, but DSPy may adapt that text, so do not rely on exact pass through.

### Typed outputs

```python
money_sig = "invoice: str -> total_amount: float"
```

DSPy parses model output into the declared schema. Use scalars, lists, and Pydantic models when needed. Do not write custom parsers.

### Multi field example

```python
sig = "title: str, body: str -> tags: list[str], summary: str"
```

Choose field names that clearly indicate meaning, preferring single-word nouns when possible and concrete domain-specific terms for inputs (e.g., `invoice` instead of `invoice_text`) and explicit result nouns for outputs (e.g., `total_amount`). Avoid underscores unless absolutely necessary. Rename ambiguous fields like `val` to `price`, `txt` to `description`, or `num` to `quantity`.

## Modules: the how

Select the inference strategy that executes a given Signature.

Swap the module, keep the signature. Change the path, keep the goal.

### Predict, the minimal path

```python
qa = dspy.Predict(sig1)
qa(question="What is the capital of france?")
# → Prediction(answer='Paris.')
```

Predict builds a prompt from the signature, calls the model, returns a Prediction with structured fields.

### Chain of Thought, add steps

```python
mult_sig = "number_a: int, number_b: int -> sum: int"
mult_cot = dspy.ChainOfThought(mult_sig)
mult_cot(number_a=5, number_b=13)
# → Prediction(reasoning='…', sum=18)
```

Switch from Predict to ChainOfThought to add steps that lead to the answer.

### ReAct with tools, add actions

```python
def calculate_sum(a: int, b: int) -> int:
    """calculate the sum of two numbers"""
    return a + b

mult_agent = dspy.ReAct(mult_sig, tools=[calculate_sum])
mult_agent(number_a=123122, number_b=3122312)
# → Prediction(trajectory={...}, reasoning='…', sum=3245434)
```

A tool removes guesswork. The signature remains the same. The module changes.

Signature unchanged. Module swapped. Signature unchanged. Module swapped.

## Task Decomposition with Custom Modules

Compose smaller modules to process segments and then aggregate results into a final prediction.

```python
class DocumentSummarizer(dspy.Module):
    def __init__(self):
        self.summarize_chapter = dspy.ChainOfThought("chapter -> summary")
        self.create_document_summary = dspy.ChainOfThought(
            "chapter_summaries: list[str] -> document_summary"
        )

    def forward(self, document: str):
        chapters = document.split("##")
        chapter_summaries = []
        for chapter in chapters:
            s = self.summarize_chapter(chapter=chapter).summary
            chapter_summaries.append(s)
        doc_sum = self.create_document_summary(
            chapter_summaries=chapter_summaries
        ).document_summary
        return dspy.Prediction(summary=doc_sum)
```

From the outside, it acts like a single module. Inside, it composes smaller modules. Encapsulate. Abstract. Compose.

Composition moves:

- Map chapters to chapter summaries
- Reduce chapter summaries into a document summary
- Validate shape and types at the boundary

## Interface Contracts and Types

Enforce data contracts at the boundary of each module and signature.

- Every signature defines a contract for inputs and outputs
- Use concrete types for numbers, strings, lists, and objects
- Attach validators for ranges, enums, and formats
- Treat parsing errors as contract breaks, fix either the signature or the module

### Example with a model

```python
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    sku: str
    qty: int = Field(gt=0)
    price: float = Field(ge=0)

class Invoice(BaseModel):
    items: list[LineItem]
    total: float
```

Use the model as the output field. Keep the parse strict. Fix names that cause drift.

## Tools and Environment

Design tools with clear contracts and safe side effects.

### Tool Contract

- Pure function signatures with explicit input and output types
- Strict input validation
- Deterministic behavior for the same inputs
- Timeout and retry policy
- Defined error types
- Structured logging fields (cost, duration, error labels)

```python
class ToolError(Exception):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def calculate_sum(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ToolError("Inputs must be integers", code="InvalidInput")
    return a + b
```

### Common tools

- Calculators for math
- Retrievers for text segments
- Search for web or local indexes
- Formatters for units and dates

## Evaluation and Testing

Measure correctness with unit, regression, and contract tests that guard types and behavior.

- Write unit tests for modules with fixed seeds
- Use golden files for stable outputs
- Track invariants such as type, length, and key field presence
- Run regression tests when a signature or module changes
- Seed modules for repeatable outputs
- Generate cases that hit boundary values and empty inputs
- Snapshot stable outputs and fail on drift
- Assert type and shape invariants on every Prediction

### Simple test sketch

```python
def test_addition():
    mult_sig = "number_a: int, number_b: int -> sum: int"
    add = dspy.ReAct(mult_sig, tools=[calculate_sum])
    out = add(number_a=2, number_b=3)
    assert out.sum == 5
```

## Performance & Telemetry

Unify observability and cost control into a single performance discipline.

Propose a common log schema with fields: `timestamp`, `correlation_id`, `signature`, `module`, `tokens_in`, `tokens_out`, `latency_ms`, `tool_calls[]`, `error_code`.

Record every call as a JSON object for consistency.

Example log record:

```json
{
  "timestamp": "2025-08-30T12:00:00Z",
  "correlation_id": "abc123",
  "signature": "number_a: int, number_b: int -> sum: int",
  "module": "ReAct",
  "tokens_in": 15,
  "tokens_out": 3,
  "latency_ms": 42,
  "tool_calls": [
    {"tool": "calculate_sum", "args": {"a": 2, "b": 3}, "result": 5}
  ],
  "error_code": null
}
```

### Budgets

Set budgets for tokens, time, and money by module and request. Reference the metrics in logs to monitor and enforce budgets. Apply circuit breakers when budgets are exceeded. Provide safe fallbacks when budgets block completion.

Instrument every run so you can debug and improve programs.

- Log one record per call with timestamp, signature name, module name, input ids, output ids, and truncated content
- Attach a correlation id to link logs across calls
- Emit counters for calls, successes, failures, retries
- Emit histograms for latency, tokens, and tool cost
- Capture traces for tool calls and LM calls with parent child links

### Cost and Latency Control

Set budgets for tokens, time, and money and choose the lowest cost strategy that meets accuracy.

- Prefer Predict when a direct answer suffices
- Use ChainOfThought when a chain yields better accuracy
- Add tools when the model lacks a skill
- Cache inputs and outputs for hot paths
- Set token, time, and money budgets per request and per module
- Prefer narrow context and short chains when they meet accuracy
- Apply timeouts and circuit breakers to tool and LM calls
- Use caching with a clear time to live and stable keys
- Provide a safe fallback for partial answers when a budget is hit

## Developer Ergonomics

- No manual chat arrays that hide structure
- No manual parsers for nested text
- Prefer named fields over long strings
- Iterate by swapping modules, not by rewriting prompt text

## DSPy Design Patterns

1. Signature first

```python
plan = "input -> output"
```

2. Start simple

```python
solve = dspy.Predict(plan)
```

3. Add thought when needed

```python
solve = dspy.ChainOfThought(plan)
```

4. Add tools when needed

```python
solve = dspy.ReAct(plan, tools=[calculate_sum])
```

5. Decompose when scale demands

```python
class SolveBig(dspy.Module):
    pass
```

Signature to Predict to CoT to ReAct to Custom Module. Climb this ladder as scale grows.

## Style Guidelines

- Docstrings guide DSPy, not the raw model input
- Field names encode intent, pick names that signal purpose
- Use typed outputs to stabilize calling code
- When a rule blocks progress or conflicts, ask for review

## Checklists

### Before writing code

- Outline the workflow and data flow
- Ensure the signature matches the goal
- Define inputs and outputs with types
- Pick the simplest module that can work
- Name fields to match meaning

### When results are weak

- Check that inputs and outputs match the task
- Add a worked example or a short schema note
- Swap to ChainOfThought for a chain of steps
- Add a tool with ReAct for a missing skill
- Split into smaller modules

### Before merging

- Confirm modules and signatures match the principles above
- Check similar tasks for naming and shape alignment
- Verify outputs pass type checks
- Remove manual parsing
- Record strategy changes and why

## Failure Handling

Handle errors as data so callers can recover or escalate.

- If output does not fit the declared type, retry once, then escalate
- If fields seem wrong, propose clearer names in a pull request
- If a docstring seems ignored, assume DSPy adapted it, refine fields instead
- Raise typed errors for validation, tool failure, and budget exceeded
- Include a code, a short message, and a remediation hint
- Escalate to human review when retries fail or data risk is detected
- Log the error and the inputs that led to it without leaking secrets

## Anti patterns

- One giant prompt that mixes intent and method
- Hidden state that alters behavior across calls
- Outputs that shift shape across runs
- Names that do not match the goal

## Examples Library

### Classification

```python
label_sig = "text: str -> label: str"
label = dspy.Predict(label_sig)
out = label(text="A short review that notes great battery life")
# out.label
```

### Extraction

```python
extract_sig = "record: str -> fields: list[str]"
extract = dspy.ChainOfThought(extract_sig)
res = extract(record="Order ABC, 3 units, total 120.00")
# res.fields
```

### RAG sketch

```python
def retrieve(query: str) -> list[str]:
    return []

rag_sig = "question: str, passages: list[str] -> answer: str"
rag = dspy.ReAct(rag_sig, tools=[retrieve])
ans = rag(question="When does the warranty start?", passages=["..."])
# ans.answer
```

# Tutorial: Saving and Loading your DSPy program

This guide demonstrates how to save and load your DSPy program. At a high level, there are two ways to save your DSPy program:

1. Save the state of the program only, similar to weights-only saving in PyTorch.
2. Save the whole program, including both the architecture and the state, which is supported by `dspy>=2.6.0`.

## State-only Saving

State represents the DSPy program's internal state, including the signature, demos (few-shot examples), and other information like
the `lm` to use for each `dspy.Predict` in the program. It also includes configurable attributes of other DSPy modules like
`k` for `dspy.retrievers.Retriever`. To save the state of a program, use the `save` method and set `save_program=False`. You can
choose to save the state to a JSON file or a pickle file. We recommend saving the state to a JSON file because it is safer and readable.
But sometimes your program contains non-serializable objects like `dspy.Image` or `datetime.datetime`, in which case you should save
the state to a pickle file.

Let's say we have compiled a program with some data, and we want to save the program for future usage:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

To save the state of your program to json file:

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

To save the state of your program to a pickle file:

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

To load your saved state, you need to **recreate the same program**, then load the state using the `load` method.

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

Or load the state from a pickle file:

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.pkl")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## Whole Program Saving

Starting from `dspy>=2.6.0`, DSPy supports saving the whole program, including the architecture and the state. This feature
is powered by `cloudpickle`, which is a library for serializing and deserializing Python objects.

To save the whole program, use the `save` method and set `save_program=True`, and specify a directory path to save the program
instead of a file name. We require a directory path because we also save some metadata, e.g., the dependency versions along
with the program itself.

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

To load the saved program, directly use `dspy.load` method:

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

With whole program saving, you don't need to recreate the program, but can directly load the architecture along with the state.
You can pick the suitable saving approach based on your needs.

### Serializing Imported Modules

When saving a program with `save_program=True`, you might need to include custom modules that your program depends on. This is
necessary if your program depends on these modules, but at loading time these modules are not imported before calling `dspy.load`.

You can specify which custom modules should be serialized with your program by passing them to the `modules_to_serialize`
parameter when calling `save`. This ensures that any dependencies your program relies on are included during serialization and
available when loading the program later.

Under the hood this uses cloudpickle's `cloudpickle.register_pickle_by_value` function to register a module as picklable by value.
When a module is registered this way, cloudpickle will serialize the module by value rather than by reference, ensuring that the
module contents are preserved with the saved program.

For example, if your program uses custom modules:

```python
import dspy
import my_custom_module

compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

# Save the program with the custom module
compiled_dspy_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

This ensures that the required modules are properly serialized and available when loading the program later. Any number of
modules can be passed to `modules_to_serialize`. If you don't specify `modules_to_serialize`, no additional modules will be
registered for serialization.

## Backward Compatibility

As of `dspy<3.0.0`, we don't guarantee the backward compatibility of the saved program. For example, if you save the program with `dspy==2.5.35`,
at loading time please make sure to use the same version of DSPy to load the program, otherwise the program may not work as expected. Chances
are that loading a saved file in a different version of DSPy will not raise an error, but the performance could be different from when
the program was saved.

Starting from `dspy>=3.0.0`, we will guarantee the backward compatibility of the saved program in major releases, i.e., programs saved in `dspy==3.0.0`
should be loadable in `dspy==3.7.10`.

# Use and Customize DSPy Cache

In this tutorial, we will explore the design of DSPy's caching mechanism and demonstrate how to effectively use and customize it.

## DSPy Cache Structure

DSPy's caching system is architected in three distinct layers:

1.  **In-memory cache**: Implemented using `cachetools.LRUCache`, this layer provides fast access to frequently used data.
2.  **On-disk cache**: Leveraging `diskcache.FanoutCache`, this layer offers persistent storage for cached items.
3.  **Prompt cache (Server-side cache)**: This layer is managed by the LLM service provider (e.g., OpenAI, Anthropic).

While DSPy does not directly control the server-side prompt cache, it offers users the flexibility to enable, disable, and customize the in-memory and on-disk caches to suit their specific requirements.

## Using DSPy Cache

By default, both in-memory and on-disk caching are automatically enabled in DSPy. No specific action is required to start using the cache. When a cache hit occurs, you will observe a significant reduction in the module call's execution time. Furthermore, if usage tracking is enabled, the usage metrics for a cached call will be `None`.

Consider the following example:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")
```

A sample output looks like:

```
Time elapse:  4.384113
Total usage: {'openai/gpt-4o-mini': {'completion_tokens': 97, 'prompt_tokens': 144, 'total_tokens': 241, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0, 'text_tokens': None, 'image_tokens': None}}}

Time elapse:  0.000529
Total usage: {}
```

## Disabling/Enabling DSPy Cache

There are scenarios where you might need to disable caching, either entirely or selectively for in-memory or on-disk caches. For instance:

- You require different responses for identical LM requests.
- You lack disk write permissions and need to disable the on-disk cache.
- You have limited memory resources and wish to disable the in-memory cache.

DSPy provides the `dspy.configure_cache()` utility function for this purpose. You can use the corresponding flags to control the enabled/disabled state of each cache type:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

In additions, you can manage the capacity of the in-memory and on-disk caches:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

Please note that `disk_size_limit_bytes` defines the maximum size in bytes for the on-disk cache, while `memory_max_entries` specifies the maximum number of entries for the in-memory cache.

## Understanding and Customizing the Cache

In specific situations, you might want to implement a custom cache, for example, to gain finer control over how cache keys are generated. By default, the cache key is derived from a hash of all request arguments sent to `litellm`, excluding credentials like `api_key`.

To create a custom cache, you need to subclass `dspy.clients.Cache` and override the relevant methods:

```python
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {write your own constructor}

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {write your logic of computing cache key}

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {write your cache read logic}

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {write your cache write logic}
```

To ensure seamless integration with the rest of DSPy, it is recommended to implement your custom cache using the same method signatures as the base class, or at a minimum, include `**kwargs` in your method definitions to prevent runtime errors during cache read/write operations.

Once your custom cache class is defined, you can instruct DSPy to use it:

```python
dspy.cache = CustomCache()
```

Let's illustrate this with a practical example. Suppose we want the cache key computation to depend solely on the request message content, ignoring other parameters like the specific LM being called. We can create a custom cache as follows:

```python
class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)
```

For comparison, consider executing the code below without the custom cache:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")
```

The time elapsed will indicate that the cache is not hit on the second call. However, when using the custom cache:

```python
import dspy
import os
import time
from typing import Dict, Any, Optional
import ujson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")
```

You will observe that the cache is hit on the second call, demonstrating the effect of the custom cache key logic.

# Tutorial: Deploying your DSPy program

This guide demonstrates two potential ways to deploy your DSPy program in production: FastAPI for lightweight deployments and MLflow for more production-grade deployments with program versioning and management.

Below, we'll assume you have the following simple DSPy program that you want to deploy. You can replace this with something more sophisticated.

```python
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

## Deploying with FastAPI

FastAPI offers a straightforward way to serve your DSPy program as a REST API. This is ideal when you have direct access to your program code and need a lightweight deployment solution.

```bash
> pip install fastapi uvicorn
> export OPENAI_API_KEY="your-openai-api-key"
```

Let's create a FastAPI application to serve your `dspy_program` defined above.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and 'asyncify' your DSPy program.
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm, async_max_workers=4) # default is 8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

In the code above, we call `dspy.asyncify` to convert the dspy program to run in async mode for high-throughput FastAPI
deployments. Currently, this runs the dspy program in a separate thread and awaits its result.

By default, the limit of spawned threads is 8. Think of this like a worker pool.
If you have 8 in-flight programs and call it once more, the 9th call will wait until one of the 8 returns.
You can configure the async capacity using the new `async_max_workers` setting.

??? "Streaming, in DSPy 2.6.0+"

    Streaming is also supported in DSPy 2.6.0+, which can be installed via `pip install -U dspy`.

    We can use `dspy.streamify` to convert the dspy program to a streaming mode. This is useful when you want to stream
    the intermediate outputs (i.e. O1-style reasoning) to the client before the final prediction is ready. This uses
    asyncify under the hood and inherits the execution semantics.

    ```python
    dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
    streaming_dspy_program = dspy.streamify(dspy_program)

    @app.post("/predict/stream")
    async def stream(question: Question):
        async def generate():
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    data = {"prediction": value.labels().toDict()}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {ujson.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Since you're often going to want to stream the result of a DSPy program as server-sent events,
    # we've included a helper function for that, which is equivalent to the code above.

    from dspy.utils.streaming import streaming_response

    @app.post("/predict/stream")
    async def stream(question: Question):
        stream = streaming_dspy_program(question=question.text)
        return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
    ```

Write your code to a file, e.g., `fastapi_dspy.py`. Then you can serve the app with:

```bash
> uvicorn fastapi_dspy:app --reload
```

It will start a local server at `http://127.0.0.1:8000/`. You can test it with the python code below:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "What is the capital of France?"}
)
print(response.json())
```

You should see the response like below:

```json
{
  "status": "success",
  "data": {
    "reasoning": "The capital of France is a well-known fact, commonly taught in geography classes and referenced in various contexts. Paris is recognized globally as the capital city, serving as the political, cultural, and economic center of the country.",
    "answer": "The capital of France is Paris."
  }
}
```

## Deploying with MLflow

We recommend deploying with MLflow if you are looking to package your DSPy program and deploy in an isolated environment.
MLflow is a popular platform for managing machine learning workflows, including versioning, tracking, and deployment.

```bash
> pip install mlflow>=2.18.0
```

Let's spin up the MLflow tracking server, where we will store our DSPy program. The command below will start a local server at
`http://127.0.0.1:5000/`.

```bash
> mlflow ui
```

Then we can define the DSPy program and log it to the MLflow server. "log" is an overloaded term in MLflow, basically it means
we store the program information along with environment requirements in the MLflow server. This is done via the `mlflow.dspy.log_model()`
function, please see the code below:

> [!NOTE]
> As of MLflow 2.22.0, there is a caveat that you must wrap your DSPy program in a custom DSPy Module class when deploying with MLflow.
> This is because MLflow requires positional arguments while DSPy pre-built modules disallow positional arguments, e.g., `dspy.Predict`
> or `dspy.ChainOfThought`. To work around this, create a wrapper class that inherits from `dspy.Module` and implement your program's
> logic in the `forward()` method, as shown in the example below.

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, messages):
        return self.cot(question=messages[0]["content"])

dspy_program = MyProgram()

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

We recommend you to set `task="llm/v1/chat"` so that the deployed program automatically takes input and generate output in
the same format as the OpenAI chat API, which is a common interface for LM applications. Write the code above into
a file, e.g. `mlflow_dspy.py`, and run it.

After you logged the program, you can view the saved information in MLflow UI. Open `http://127.0.0.1:5000/` and select
the `deploy_dspy_program` experiment, then select the run your just created, under the `Artifacts` tab, you should see the
logged program information, similar to the following screenshot:

![MLflow UI](./dspy_mlflow_ui.png)

Grab your run id from UI (or the console print when you execute `mlflow_dspy.py`), now you can deploy the logged program
with the following command:

```bash
> mlflow models serve -m runs:/{run_id}/model -p 6000
```

After the program is deployed, you can test it with the following command:

```bash
> curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json"  --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

You should see the response like below:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"reasoning\": \"The question asks for the sum of 2 and 2. To find the answer, we simply add the two numbers together: 2 + 2 = 4.\", \"answer\": \"4\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

For complete guide on how to deploy a DSPy program with MLflow, and how to customize the deployment, please refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

### Best Practices for MLflow Deployment

1. **Environment Management**: Always specify your Python dependencies in a `conda.yaml` or `requirements.txt` file.
2. **Versioning**: Use meaningful tags and descriptions for your model versions.
3. **Input Validation**: Define clear input schemas and examples.
4. **Monitoring**: Set up proper logging and monitoring for production deployments.

For production deployments, consider using MLflow with containerization:

```bash
> mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
> docker run -p 6000:8080 dspy-program
```

For a complete guide on production deployment options and best practices, refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

# Streaming

In this guide, we will walk you through how to enable streaming in your DSPy program. DSPy Streaming
consists of two parts:

- **Output Token Streaming**: Stream individual tokens as they're generated, rather than waiting for the complete response.
- **Intermediate Status Streaming**: Provide real-time updates about the program's execution state (e.g., "Calling web search...", "Processing results...").

## Output Token Streaming

DSPy's token streaming feature works with any module in your pipeline, not just the final output. The only requirement is that the streamed field must be of type `str`. To enable token streaming:

1. Wrap your program with `dspy.streamify`
2. Create one or more `dspy.streaming.StreamListener` objects to specify which fields to stream

Here's a basic example:

```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
```

To consume the streamed output:

```python
import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```

This will produce output like:

```
StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' other')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' side of the frying pan!')
Prediction(
    answer='To get to the other side of the frying pan!'
)
```

Note: Since `dspy.streamify` returns an async generator, you must use it within an async context. If you're using an environment like Jupyter or Google Colab that already has an event loop (async context), you can use the generator directly.

You may have noticed that the above streaming contains two different entities: `StreamResponse`
and `Prediction.` `StreamResponse` is the wrapper over streaming tokens on the field being listened to, and in
this example it is the `answer` field. `Prediction` is the program's final output. In DSPy, streaming is
implemented in a sidecar fashion: we enable streaming on the LM so that LM outputs a stream of tokens. We send these
tokens to a side channel, which is being continuously read by the user-defined listeners. Listeners keep interpreting
the stream, and decides if the `signature_field_name` it is listening to has started to appear and has finalized.
Once it decides that the field appears, the listener begins outputting tokens to the async generator users can
read. Listeners' internal mechanism changes according to the adapter behind the scene, and because usually
we cannot decide if a field has finalized until seeing the next field, the listener buffers the output tokens
before sending to the final generator, which is why you will usually see the last chunk of type `StreamResponse`
has more than one token. The program's output is also written to the stream, which is the chunk of `Prediction`
as in the sample output above.

To handle these different types and implement custom logic:

```python
import asyncio

async def read_output_stream():
  output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

  async for chunk in output_stream:
    return_value = None
    if isinstance(chunk, dspy.streaming.StreamResponse):
      print(f"Output token of field {chunk.signature_field_name}: {chunk.chunk}")
    elif isinstance(chunk, dspy.Prediction):
      return_value = chunk


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

### Understand `StreamResponse`

`StreamResponse` (`dspy.streaming.StreamResponse`) is the wrapper class of streaming tokens. It comes with 3
fields:

- `predict_name`: the name of the predict that holds the `signature_field_name`. The name is the
  same name of keys as you run `your_program.named_predictors()`. In the code above because `answer` is from
  the `predict` itself, so the `predict_name` shows up as `self`, which is the only key as your run
  `predict.named_predictors()`.
- `signature_field_name`: the output field that these tokens map to. `predict_name` and `signature_field_name`
  together form the unique identifier of the field. We will demonstrate how to handle multiple fields streaming
  and duplicated field name later in this guide.
- `chunk`: the value of the stream chunk.

### Streaming with Cache

When a cached result is found, the stream will skip individual tokens and only yield the final `Prediction`. For example:

```
Prediction(
    answer='To get to the other side of the dinner plate!'
)
```

### Streaming Multiple Fields

You can monitor multiple fields by creating a `StreamListener` for each one. Here's an example with a multi-module program:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)

async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value

program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

The output will look like:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk='To')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' reach')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' the')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' other side of the recipe!')
Final output:  Prediction(
    simplified_answer='To reach the other side of the recipe!'
)
```

### Streaming the Same Field Multiple Times (as in dspy.ReAct)

By default, a `StreamListener` automatically closes itself after completing a single streaming session.
This design helps prevent performance issues, since every token is broadcast to all configured stream listeners,
and having too many active listeners can introduce significant overhead.

However, in scenarios where a DSPy module is used repeatedly in a loop—such as with `dspy.ReAct` — you may want to stream
the same field from each prediction, every time it is used. To enable this behavior, set allow_reuse=True when creating
your `StreamListener`. See the example below:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


def fetch_user_info(user_name: str):
    """Get user information like name, birthday, etc."""
    return {
        "name": user_name,
        "birthday": "2009-05-16",
    }


def get_sports_news(year: int):
    """Get sports news for a given year."""
    if year == 2009:
        return "Usane Bolt broke the world record in the 100m race."
    return None


react = dspy.ReAct("question->answer", tools=[fetch_user_info, get_sports_news])

stream_listeners = [
    # dspy.ReAct has a built-in output field called "next_thought".
    dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
]
stream_react = dspy.streamify(react, stream_listeners=stream_listeners)


async def read_output_stream():
    output = stream_react(question="What sports news happened in the year Adam was born?")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


print(asyncio.run(read_output_stream()))
```

In this example, by setting `allow_reuse=True` in the StreamListener, you ensure that streaming for "next_thought" is
available for every iteration, not just the first. When you run this code, you will see the streaming tokens for `next_thought`
output each time the field is produced.

#### Handling Duplicate Field Names

When streaming fields with the same name from different modules, specify both the `predict` and `predict_name` in the `StreamListener`:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("question, answer->answer, score")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict1,
        predict_name="predict1"
    ),
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict2,
        predict_name="predict2"
    ),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)


async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

The output will be like:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk="I'm")
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' ready')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' assist')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' you')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk='! Please provide a question.')
Final output:  Prediction(
    answer="I'm ready to assist you! Please provide a question.",
    score='N/A'
)
```

## Intermediate Status Streaming

Status streaming keeps users informed about the program's progress, especially useful for long-running operations like tool calls or complex AI pipelines. To implement status streaming:

1. Create a custom status message provider by subclassing `dspy.streaming.StatusMessageProvider`
2. Override the desired hook methods to provide custom status messages
3. Pass your provider to `dspy.streamify`

Example:

```python
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def lm_end_status_message(self, outputs):
        return f"Tool finished with output: {outputs}!"
```

Available hooks:

- lm_start_status_message: status message at the start of calling dspy.LM.
- lm_end_status_message: status message at the end of calling dspy.LM.
- module_start_status_message: status message at the start of calling a dspy.Module.
- module_end_status_message: status message at the start of calling a dspy.Module.
- tool_start_status_message: status message at the start of calling dspy.Tool.
- tool_end_status_message: status message at the end of calling dspy.Tool.

Each hook should return a string containing the status message.

After creating the message provider, just pass it to `dspy.streamify`, and you can enable both
status message streaming and output token streaming. Please see the example below. The intermediate
status message is represented in the class `dspy.streaming.StatusMessage`, so we need to have
another condition check to capture it.

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.tool = dspy.Tool(lambda x: 2 * x, name="double_the_number")
        self.predict = dspy.ChainOfThought("num1, num2->sum")

    def forward(self, num, **kwargs):
        num2 = self.tool(x=num)
        return self.predict(num1=num, num2=num2)


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def tool_start_status_message(self, instance, inputs):
        return f"Calling Tool {instance.name} with inputs {inputs}..."

    def tool_end_status_message(self, outputs):
        return f"Tool finished with output: {outputs}!"


predict = MyModule()
stream_listeners = [
    # dspy.ChainOfThought has a built-in output field called "reasoning".
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
    status_message_provider=MyStatusMessageProvider(),
)


async def read_output_stream():
    output = stream_predict(num=3)

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            print(chunk)
    return return_value


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

Sample output:

```
StatusMessage(message='Calling tool double_the_number...')
StatusMessage(message='Tool calling finished! Querying the LLM with tool calling results...')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='To')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' find')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' sum')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' of')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' two')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' numbers')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' we')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' simply')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' add')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' them')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' together')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='.')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' Here')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' ')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='3')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' plus')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 6 equals 9.')
Final output:  Prediction(
    reasoning='To find the sum of the two numbers, we simply add them together. Here, 3 plus 6 equals 9.',
    sum='9'
)
```

## Synchronous Streaming

By default calling a streamified DSPy program produces an async generator. In order to get back
a sync generator, you can set the flag `async_streaming=False`:


```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,
)

output = stream_predict(question="why did a chicken cross the kitchen?")

program_output = None
for chunk in output:
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk)
    elif isinstance(chunk, dspy.Prediction):
        program_output = chunk
print(f"Program output: {program_output}")
```

# Async DSPy Programming

DSPy provides native support for asynchronous programming, allowing you to build more efficient and
scalable applications. This guide will walk you through how to leverage async capabilities in DSPy,
covering both built-in modules and custom implementations.

## Why Use Async in DSPy?

Asynchronous programming in DSPy offers several benefits:
- Improved performance through concurrent operations
- Better resource utilization
- Reduced waiting time for I/O-bound operations
- Enhanced scalability for handling multiple requests

## When Should I use Sync or Async?

Choosing between synchronous and asynchronous programming in DSPy depends on your specific use case.
Here's a guide to help you make the right choice:

Use Synchronous Programming When

- You're exploring or prototyping new ideas
- You're conducting research or experiments
- You're building small to medium-sized applications
- You need simpler, more straightforward code
- You want easier debugging and error tracking

Use Asynchronous Programming When:

- You're building a high-throughput service (high QPS)
- You're working with tools that only support async operations
- You need to handle multiple concurrent requests efficiently
- You're building a production service that requires high scalability

### Important Considerations

While async programming offers performance benefits, it comes with some trade-offs:

- More complex error handling and debugging
- Potential for subtle, hard-to-track bugs
- More complex code structure
- Different code between ipython (Colab, Jupyter lab, Databricks notebooks, ...) and normal python runtime.

We recommend starting with synchronous programming for most development scenarios and switching to async
only when you have a clear need for its benefits. This approach allows you to focus on the core logic of
your application before dealing with the additional complexity of async programming.

## Using Built-in Modules Asynchronously

Most DSPy built-in modules support asynchronous operations through the `acall()` method. This method
maintains the same interface as the synchronous `__call__` method but operates asynchronously.

Here's a basic example using `dspy.Predict`:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    # Use acall() for async execution
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)


asyncio.run(main())
```

### Working with Async Tools

DSPy's `Tool` class seamlessly integrates with async functions. When you provide an async
function to `dspy.Tool`, you can execute it using `acall()`. This is particularly useful
for I/O-bound operations or when working with external services.

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    # Simulate an async operation
    await asyncio.sleep(0.1)
    print(f"I get: {x}")

# Create a tool from the async function
tool = dspy.Tool(foo)

async def main():
    # Execute the tool asynchronously
    await tool.acall(x=2)

asyncio.run(main())
```

Note: When using `dspy.ReAct` with tools, calling `acall()` on the ReAct instance will automatically
execute all tools asynchronously using their `acall()` methods.

## Creating Custom Async DSPy Modules

To create your own async DSPy module, implement the `aforward()` method instead of `forward()`. This method
should contain your module's async logic. Here's an example of a custom module that chains two async operations:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

    async def aforward(self, question, **kwargs):
        # Execute predictions sequentially but asynchronously
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())
```

# Tutorial: Debugging and Observability in DSPy

This guide demonstrates how to debug problems and improve observability in DSPy. Modern AI programs often involve multiple components, such as language models, retrievers, and tools. DSPy allows you to build and optimize such complex AI systems in a clean and modular way.

However, as systems grow more sophisticated, the ability to **understand what your system is doing** becomes critical. Without transparency, the prediction process can easily become a black box, making failures or quality issues difficult to diagnose and production maintenance challenging.

By the end of this tutorial, you'll understand how to debug an issue and improve observability using [MLflow Tracing](#tracing). You'll also explore how to build a custom logging solution using callbacks.



## Define a Program

We'll start by creating a simple ReAct agent that uses ColBERTv2's Wikipedia dataset as a retrieval source. You can replace this with a more sophisticated program.

```python
import dspy
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """Retrieve top 3 relevant information from ColBert"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
```

Now, let's ask the agent a simple question:

```python
prediction = agent(question="Which baseball team does Shohei Ohtani play for in June 2025?")
print(prediction.answer)
```

```
Shohei Ohtani is expected to play for the Hokkaido Nippon-Ham Fighters in June 2025, based on the available information.
```

Oh, this is incorrect. He no longer plays for the Hokkaido Nippon-Ham Fighters; he moved to the Dodgers and won the World Series in 2024! Let's debug the program and explore potential fixes.

## Using ``inspect_history``

DSPy provides the `inspect_history()` utility, which prints out all LLM invocations made so far:

```python
# Print out 5 LLM calls
dspy.inspect_history(n=5)
```

```
[2024-12-01T10:23:29.144257]

System message:

Your input fields are:
1. `question` (str)

...

Response:

Response:

[[ ## reasoning ## ]]
The search for information regarding Shohei Ohtani's team in June 2025 did not yield any specific results. The retrieved data consistently mentioned that he plays for the Hokkaido Nippon-Ham Fighters, but there was no indication of any changes or updates regarding his team for the specified date. Given the lack of information, it is reasonable to conclude that he may still be with the Hokkaido Nippon-Ham Fighters unless there are future developments that are not captured in the current data.

[[ ## answer ## ]]
Shohei Ohtani is expected to play for the Hokkaido Nippon-Ham Fighters in June 2025, based on the available information.

[[ ## completed ## ]]

```
The log reveals that the agent could not retrieve helpful information from the search tool. However, what exactly did the retriever return? While useful, `inspect_history` has some limitations:

* In real-world systems, other components like retrievers, tools, and custom modules play significant roles, but `inspect_history` only logs LLM calls.
* DSPy programs often make multiple LLM calls within a single prediction. Monolith log history makes it hard to organize logs, especially when handling multiple questions.
* Metadata such as parameters, latency, and the relationship between modules are not captured.

**Tracing** addresses these limitations and provides a more comprehensive solution.

## Tracing

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) is an end-to-end machine learning platform that is integrated seamlessly with DSPy to support best practices in LLMOps. Using MLflow's automatic tracing capability with DSPy is straightforward; **No sign up for services or an API key is required**. You just need to install MLflow and call `mlflow.dspy.autolog()` in your notebook or script.

```bash
pip install -U mlflow>=2.18.0
```

After installation, spin up your server via the command below.

```
# It is highly recommended to use SQL store when using MLflow tracing
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

If you don't specify a different port via `--port` flag, you MLflow server will be hosted at port 5000.

Now let's change our code snippet to enable MLflow tracing. We need to:

- Tell MLflow where the server is hosted.
- Apply `mlflow.autolog()` so that DSPy tracing is automatically captured.

The full code is as below, now let's run it again!

```python
import dspy
import os
import mlflow

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# Tell MLflow about the server URI.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DSPy")

lm = dspy.LM("openai/gpt-4o-mini")
colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
dspy.configure(lm=lm)


def retrieve(query: str):
    """Retrieve top 3 relevant information from ColBert"""
    results = colbert(query, k=3)
    return [x["text"] for x in results]


agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)
print(agent(question="Which baseball team does Shohei Ohtani play for?"))
```


MLflow automatically generates a **trace** for each prediction and records it within your experiment. To explore these traces visually, open `http://127.0.0.1:5000/`
in your browser, then select the experiment you just created and navigate to the Traces tab:

![MLflow Trace UI](./mlflow_trace_ui.png)

Click on the most recent trace to view its detailed breakdown:

![MLflow Trace View](./mlflow_trace_view.png)

Here, you can examine the input and output of every step in your workflow. For example, the screenshot above shows the `retrieve` function's input and output. By inspecting the retriever's output, you can see that it returned outdated information, which is not sufficient to determine which team Shohei Ohtani plays for in June 2025. You can also inspect
other steps, e.g, language model's input, output, and configuration.

To address the issue of outdated information, you can replace the `retrieve` function with a web search tool powered by [Tavily search](https://www.tavily.com/).

```python
from tavily import TavilyClient
import dspy
import mlflow

# Tell MLflow about the server URI.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("DSPy")

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """Run a web search and return the content from the top 5 search results"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[web_search])

prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(agent.answer)
```

```
Los Angeles Dodgers
```

Below is a GIF demonstrating how to navigate through the MLflow UI:

![MLflow Trace UI Navigation](./mlflow_trace_ui_navigation.gif)


For a complete guide on how to use MLflow tracing, please refer to
the [MLflow Tracing Guide](https://mlflow.org/docs/3.0.0rc0/tracing).



!!! info Learn more about MLflow

    MLflow is an end-to-end LLMOps platform that offers extensive features like experiment tracking, evaluation, and deployment. To learn more about DSPy and MLflow integration, visit [this tutorial](../deployment/index.md#deploying-with-mlflow).


## Building a Custom Logging Solution

Sometimes, you may want to implement a custom logging solution. For instance, you might need to log specific events triggered by a particular module. DSPy's **callback** mechanism supports such use cases. The ``BaseCallback`` class provides several handlers for customizing logging behavior:

|Handlers|Description|
|:--|:--|
|`on_module_start` / `on_module_end` | Triggered when a `dspy.Module` subclass is invoked. |
|`on_lm_start` / `on_lm_end` | Triggered when a `dspy.LM` subclass is invoked. |
|`on_adapter_format_start` / `on_adapter_format_end`| Triggered when a `dspy.Adapter` subclass formats the input prompt. |
|`on_adapter_parse_start` / `on_adapter_parse_end`| Triggered when a `dspy.Adapter` subclass postprocess the output text from an LM. |
|`on_tool_start` / `on_tool_end` | Triggered when a `dspy.Tool` subclass is invoked. |
|`on_evaluate_start` / `on_evaluate_end` | Triggered when a `dspy.Evaluate` instance is invoked. |

Here's an example of custom callback that logs the intermediate steps of a ReAct agent:

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])
```


```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Ohtani ..."]

...
```

!!! info Handling Inputs and Outputs in Callbacks

    Be cautious when working with input or output data in callbacks. Mutating them in-place can modify the original data passed to the program, potentially leading to unexpected behavior. To avoid this, it's strongly recommended to create a copy of the data before performing any operations that may alter it.

# Building AI Applications by Customizing DSPy Modules

In this guide, we will walk you through how to build a GenAI application by customizing `dspy.Module`.

A [DSPy module](https://dspy.ai/learn/programming/modules/) is the building block for DSPy programs.

- Each built-in module abstracts a prompting technique (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.

- A DSPy module has learnable parameters (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.

Although you can build a DSPy program without implementing a custom module, we highly recommend putting your logic with a custom module so that you can use other DSPy features, like DSPy optimizer or MLflow DSPy tracing.

</div>

<div class="cell markdown" id="KBYjBQtv3Cn5">

Before getting started, make sure you have DSPy installed:

    !pip install dspy

</div>

<div class="cell markdown" id="reQSTM8a8qMf">

## Customize DSPy Module

You can implement custom prompting logic and integrate external tools or services by customizing a DSPy module. To achieve this, subclass from `dspy.Module` and implement the following two key methods:

- `__init__`: This is the constructor, where you define the attributes and sub-modules of your program.
- `forward`: This method contains the core logic of your DSPy program.

Within the `forward()` method, you are not limited to calling only other DSPy modules; you can also integrate any standard Python functions, such as those for interacting with Langchain/Agno agents, MCP tools, database handlers, and more.

The basic structure for a custom DSPy module looks like this:

``` python
class MyProgram(dspy.Module):
    
    def __init__(self, ...):
        # Define attributes and sub-modules here
        {constructor_code}

    def forward(self, input_name1, input_name2, ...):
        # Implement your program's logic here
        {custom_logic_code}
```

</div>

<div class="cell markdown" id="DziTWwT8_TrY">

Let's illustrate this with a practical code example. We will build a simple Retrieval-Augmented Generation (RAG) application with multiple stages:

1.  **Query Generation:** Generate a suitable query based on the user's question to retrieve relevant context.
2.  **Context Retrieval:** Fetch context using the generated query.
3.  **Answer Generation:** Produce a final answer based on the retrieved context and the original question.

The code implementation for this multi-stage program is shown below.

</div>

<div class="cell code" execution_count="3" id="lAoV5_v7YlvN">

``` python
import dspy

class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return [x["text"] for x in results]

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("question,context->answer")

    def forward(self, question, **kwargs):
        query = self.query_generator(question=question).query
        context = search_wikipedia(query)[0]
        return self.answer_generator(question=question, context=context).answer
```

</div>

<div class="cell markdown">

Let's take a look at the `forward` method. We first send the question to `self.query_generator`, which is a `dspy.Predict`, to get the query for context retrieving. Then we use the query to call ColBERT and keep the first context retrieved. Finally, we send the question and context into `self.answer_generator`, which is a `dspy.ChainOfThought` to generate the final answer.

</div>

<div class="cell markdown" id="FBq_4e8NamwY">

Next, we'll create an instance of our `RAG` module to run the program.

**Important:** When invoking a custom DSPy module, you should use the module instance directly (which calls the `__call__` method internally), rather than calling the `forward()` method explicitly. The `__call__` method handles necessary internal processing before executing the `forward` logic.

</div>

<div class="cell code" execution_count="7" colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ZR7xcFSTa596" outputId="f3427754-8a16-48fe-c540-8c9f31d9a30d">

``` python
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
rag = RAG()
print(rag(question="Is Lebron James the basketball GOAT?"))
```

<div class="output stream stdout">

    The question of whether LeBron James is the basketball GOAT is subjective and depends on personal opinions. Many consider him one of the greatest due to his achievements and impact on the game, but others may argue for different players like Michael Jordan.

</div>

</div>

<div class="cell markdown">

That's it! In summary, to build your GenAI applications, we just put the custom logic into the `forward()` method, then create a module instance and call the instance itself.

</div>

<div class="cell markdown" id="aYAYc-Hg39ri">

## Why Customizing Module?

DSPy is a lightweight authoring and optimization framework, and our focus is to resolve the mess of prompt engineering by transforming prompting (string in, string out) LLM into programming LLM (structured inputs in, structured outputs out) for robust AI system.

While we provide pre-built modules which have custom prompting logic like `dspy.ChainOfThought` for reasoning, `dspy.ReAct` for tool calling agent to facilitate building your AI applications, we don't aim at standardizing how you build agents.

In DSPy, your application logic simply goes to the `forward` method of your custom Module, which doesn't have any constraint as long as you are writing python code. With this layout, DSPy is easy to migrate to from other frameworks or vanilla SDK usage, and easy to migrate off because essentially it's just python code.

</div>

# `dspy.Type`

Use DSPy types by importing `from dspy import Type`. To help you understand why you should use DSPy types, take a look at the `dspy.Type` source code:

```
import json
import re
from typing import Any, get_args, get_origin

import json_repair
import pydantic

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Example:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type"""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom types from the annotation.

        This is used to extract all custom types from the annotation of a field, while the annotation can
        have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
        """
        # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
        # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result = []
        # Recurse into all type args
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        formatted = self.format()
        if isinstance(formatted, list):
            return f"{CUSTOM_TYPE_START_IDENTIFIER}{formatted}{CUSTOM_TYPE_END_IDENTIFIER}"
        return formatted


def split_message_content_for_custom_types(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split user message content into a list of content blocks.

    This method splits each user message's content in the `messages` list to be a list of content block, so that
    the custom types like `dspy.Image` can be properly formatted for better quality. For example, the split content
    may look like below if the user message has a `dspy.Image` object:

    ```
    [
        {"type": "text", "text": "{text_before_image}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
        {"type": "text", "text": "{text_after_image}"},
    ]
    ```

    This is implemented by finding the `<<CUSTOM-TYPE-START-IDENTIFIER>>` and `<<CUSTOM-TYPE-END-IDENTIFIER>>`
    in the user message content and splitting the content around them. The `<<CUSTOM-TYPE-START-IDENTIFIER>>`
    and `<<CUSTOM-TYPE-END-IDENTIFIER>>` are the reserved identifiers for the custom types as in `dspy.Type`.

    Args:
        messages: a list of messages sent to the LM. The format is the same as [OpenAI API's messages
            format](https://platform.openai.com/docs/guides/chat-completions/response-format).

    Returns:
        A list of messages with the content split into a list of content blocks around custom types content.
    """
    for message in messages:
        if message["role"] != "user":
            # Custom type messages are only in user messages
            continue

        pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        result = []
        last_end = 0
        # DSPy adapter always formats user input into a string content before custom type splitting
        content: str = message["content"]

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            # Add text before the current block
            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            # Parse the JSON inside the block
            custom_type_content = match.group(1).strip()
            try:
                try:
                    # Replace single quotes with double quotes to make it valid JSON
                    parsed = json.loads(custom_type_content.replace("'", '"'))
                except json.JSONDecodeError:
                    parsed = json_repair.loads(custom_type_content)
                for custom_type_content in parsed:
                    result.append(custom_type_content)

            except json.JSONDecodeError:
                # fallback to raw string if it's not valid JSON
                parsed = {"type": "text", "text": custom_type_content}
                result.append(parsed)

            last_end = end

        if last_end == 0:
            # No custom type found, return the original message
            continue

        # Add any remaining text after the last match
        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages
```

To help you better understand `dspy.Type`, here are some of DSPy's _built-in_ subtypes of `dspy.Type`:

## `from dspy import Code`:

```
import re
from typing import Any, ClassVar

import pydantic
from pydantic import create_model

from dspy.adapters.types.base_type import Type


class Code(Type):
    """Code type in DSPy.

    This type is useful for code generation and code analysis.

    Example 1: dspy.Code as output type in code generation:

    ```python
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


    class CodeGeneration(dspy.Signature):
        '''Generate python code to answer the question.'''

        question: str = dspy.InputField(description="The question to answer")
        code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")


    predict = dspy.Predict(CodeGeneration)

    result = predict(question="Given an array, find if any of the two numbers sum up to 10")
    print(result.code)
    ```

    Example 2: dspy.Code as input type in code analysis:

    ```python
    import dspy
    import inspect

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class CodeAnalysis(dspy.Signature):
        '''Analyze the time complexity of the function.'''

        code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
        result: str = dspy.OutputField(description="The time complexity of the function")


    predict = dspy.Predict(CodeAnalysis)


    def sleepsort(x):
        import time

        for i in x:
            time.sleep(i)
            print(i)

    result = predict(code=inspect.getsource(sleepsort))
    print(result.result)
    ```
    """

    code: str

    language: ClassVar[str] = "python"

    def format(self):
        return f"{self.code}"

    @pydantic.model_serializer()
    def serialize_model(self):
        """Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags."""
        return self.format()

    @classmethod
    def description(cls) -> str:
        return (
            "Code represented in a string, specified in the `code` field. If this is an output field, the code "
            "field should follow the markdown code block format, e.g. \n```python\n{code}\n``` or \n```cpp\n{code}\n```"
            f"\nProgramming language: {cls.language}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"code": _filter_code(data)}

        if isinstance(data, dict):
            if "code" not in data:
                raise ValueError("`code` field is required for `dspy.Code`")
            if not isinstance(data["code"], str):
                raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
            return {"code": _filter_code(data["code"])}

        raise ValueError(f"Received invalid value for `dspy.Code`: {data}")


def _filter_code(code: str) -> str:
    """Extract code from markdown code blocks, stripping any language identifier."""
    # Case 1: format like:
    # ```python
    # {code_block}
    # ```
    regex_pattern = r"```(?:[^\n]*)\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: ```<code>``` (no language, single-line)
    regex_pattern_simple = r"```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback case
    return code


# Patch __class_getitem__ directly on the class to support dspy.Code["python"] syntax
def _code_class_getitem(cls, language):
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)
```

## `from dspy import History`:

```
from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Example:
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
```

Analyze the `dspy.Type` implementation and the `dspy.Code` and `dspy.History` subtypes of `dspy.Type` I used as examples so deeply that you're able to use `dspy.Type` fluently.

# Notes

Use signatures and Modules to maximize fast iteration and declarative specification in systems. You are programming, not prompting, LMs. Signatures define what. Modules define how. Iterate by swapping strategy, not rewriting intent.

DSPy is a declarative programming model. It's sort of like relational databases / SQL.

You write what you actually want to say (and want others to read) in English, knowing that it may or may not be the best "prompt" for your LLM.

DSPy then gives you the tools to translate that clean English + code/structure + evals into the form that your LLM needs to perform best.

This can take three forms:

1. Optimized prompts, often very specific to your LLM and your evals (e.g., tricks like specific few-shot examples that work best or very specific ways of repeating certain instructions, or ALL CAPS or whatever nonsense you don't want to maintain in your code).

2. RL updates to your LLM weights. You certainly would do this by hand, but DSPy can automate RL for your pipeline *exactly* with the same API as prompt optimization.

3. Inference scaling, like applying techniques that make more extensive use of your LLM(s) at inference time.

Let's focus on #1 here. When you iterate with evals by hand to "prompt engineer", you're doing two things.

A) You're clarifying your own specs and correcting important details. This is amazing and crucial. It's NOT handled by DSPy in the general case; you're still supposed to iterate on programming your system.

B) You're overfitting to a specific LLM's nuances and failure modes, and making a lot of low-level choices to appease that model until it works. You're moving instructions to different parts of the prompt, asking for XML instead of JSON, repeating yourself for a few key things, asking the model to pretend it's Einstein, etc. 

These "tricks" aren't bad; sometimes they work. Sometimes they're even essential. But you shouldn't be the one hard-coding them into your application. Tricks are best handled by a compiler that has global view of your system and that builds a quality model of your LLM, so it can iterate on your behalf.

When you end up switching models or sharing your code with others, this all proves invaluable. When you decide you want to RL, this all proves invaluable. When the LLM(s) get so good at your task such that hacks are not necessary, this all proves invaluable. When you want to look at pleasant clean code, not weird formatting/parsing tricks, a few months later, this all proves invaluable.