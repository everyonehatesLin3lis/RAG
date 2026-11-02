---
description: Implement specification with full traceability and test-driven development
argument-hint: Specification ID (e.g., AUTH-001)
---

# Implement Specification

You are helping a developer implement a specification using test-driven development with full traceability from requirements to code.

## Usage
/implement-spec AUTH-001

## Process
1. Read specification file: @specs/$1.md
2. Extract acceptance criteria from spec
3. Generate failing tests for each criterion
4. Implement minimal code to pass tests
5. Add traceability comments: // Implements: specs/$1.md#AC-001
6. Run tests to verify 100% spec coverage
7. Generate traceability report linking code to requirements
