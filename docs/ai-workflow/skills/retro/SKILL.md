---
name: retro
description: Review a completed coding session or task, identify problems and reusable lessons, and propose improvements to project instructions, architecture, tests, documentation, or development workflow.
---

# Retro

Run a structured retrospective after meaningful development work.

The purpose is not merely to summarize what happened. The purpose is to make future development better by identifying:

- friction
- mistakes
- repeated debugging
- incorrect assumptions
- missing tests
- weak architecture decisions
- unnecessary complexity
- reusable discoveries
- project rules worth preserving

## When to use

Use this skill when:

- a feature has been completed
- a debugging session has finished
- several implementation attempts failed
- the user asks what was learned
- the user asks to improve the development workflow
- important project knowledge should be preserved
- a recurring mistake has appeared
- development instructions may need updating

Typical requests:

- "retro"
- "review what we just did"
- "what did we learn?"
- "what went wrong?"
- "update our development rules"
- "capture this learning"

## Core principle

Do not change project rules or permanent documentation silently.

First identify the learning.

Then explain:

1. what happened
2. why it happened
3. whether it is reusable
4. where it should be preserved
5. what exact change is recommended

Only make permanent changes when appropriate.

## Retrospective process

### 1. Reconstruct the task

Determine:

- original objective
- intended architecture
- files/components involved
- tools used
- major implementation decisions
- tests or checks performed

Do not rely only on the final state.

Consider the path taken to reach it.

### 2. Detect friction

Look for events such as:

- repeated errors
- repeated edits to the same area
- misunderstood requirements
- wrong assumptions
- unnecessary API calls
- excessive token usage
- incorrect tool selection
- fragile implementation
- missing environment variables
- dependency problems
- tests that should have existed earlier
- debugging that could have been avoided

For each important item record:

```text
Problem:
Root cause:
Impact:
Better approach:
```

### 3. Detect reusable learnings

Look for knowledge that would help future work.

Examples:

- project architecture facts
- API behaviour
- naming conventions
- database constraints
- useful commands
- working debugging techniques
- environment requirements
- framework-specific behaviour
- test patterns
- integration limitations

A learning should be stored only if it is likely to matter again.

Avoid preserving temporary noise.

### 4. Classify each finding

Choose the most appropriate destination.

Possible destinations:

#### Project rule

Use when future development should consistently follow the lesson.

Examples:

- always use structured JSON responses
- all API endpoints must use Pydantic schemas
- never call the LLM directly from frontend code

#### Architecture documentation

Use when the finding explains how the system works.

Examples:

- authentication flow
- database ownership
- service boundaries
- RAG pipeline

#### Test

Use when the learning represents behaviour that should be automatically verified.

Examples:

- regression discovered during debugging
- security boundary
- edge case
- expected API response

#### Developer documentation

Use for commands, setup steps, configuration, or operational knowledge.

#### Code improvement

Use when the finding should be solved directly in implementation.

#### Skill/instruction improvement

Use when the AI development workflow itself caused the problem.

#### No action

Use when the observation is temporary or unlikely to recur.

### 5. Prioritize

Do not produce a giant list of trivial observations.

Prioritize findings that:

- prevent future bugs
- reduce repeated debugging
- improve reliability
- improve security
- reduce development time
- reduce LLM/token cost
- clarify architecture

Prefer approximately 3–10 meaningful findings.

### 6. Propose concrete improvements

Each proposed improvement should contain:

```text
Finding:
Evidence:
Why it matters:
Recommended destination:
Recommended change:
```

### 7. Verify before declaring completion

Before saying the retrospective is complete, check:

- major friction was identified
- reusable learnings were considered
- obvious regression tests were identified
- documentation implications were considered
- architecture implications were considered
- no speculative learning was presented as fact

## Output format

Return:

# Retrospective

## What worked

Short list of successful decisions.

## Friction

For each important issue:

### Finding

**Problem:**  
...

**Root cause:**  
...

**Better approach:**  
...

## Reusable learnings

For each learning:

**Learning:**  
...

**Where it belongs:**  
Project rules / architecture / test / docs / code / skill

**Recommended action:**  
...

## Recommended changes

Use a concise table:

| Priority | Change | Destination | Reason |
|---|---|---|---|
| High | ... | ... | ... |

## Next-session improvements

List the most important behaviours that should change during the next development session.

## Rules

Never:

- invent problems that did not occur
- preserve temporary debugging noise as permanent rules
- rewrite major architecture because of one minor issue
- claim something was tested when it was not
- hide unresolved problems
- silently create permanent rules without explaining why

Always prefer evidence from the actual development session.
