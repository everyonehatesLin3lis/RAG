---
name: implement-spec
description: Implement a feature from a written specification using acceptance criteria, test-driven development, minimal implementation, verification, and requirement-to-code traceability.
---

# Implement Spec

Implement software directly from a written specification.

The goal is to ensure every requirement has:

1. an explicit acceptance criterion
2. at least one verification mechanism
3. an implementation
4. a final status

The specification is the source of truth.

## When to use

Use when:

- a specification already exists
- a feature contains explicit acceptance criteria
- implementation must closely follow requirements
- traceability between requirements, tests, and code is useful
- the user asks to implement a numbered specification

Examples:

```text
implement AUTH-001
implement SPEC-004
build the feature described in specs/search.md
```

## Core principle

Do not begin coding before understanding the acceptance criteria.

The workflow is:

```text
SPEC
 ↓
ACCEPTANCE CRITERIA
 ↓
TESTS
 ↓
IMPLEMENTATION
 ↓
VERIFICATION
 ↓
TRACEABILITY
```

## Process

### 1. Read the specification

Locate the requested specification.

Example structure:

```text
specs/
  AUTH-001.md
  SEARCH-001.md
  RAG-001.md
```

Read the complete relevant specification.

Identify:

- objective
- functional requirements
- non-functional requirements
- acceptance criteria
- constraints
- dependencies
- explicit exclusions

Do not implement requirements that are not present unless they are technically necessary.

### 2. Extract acceptance criteria

Convert requirements into individually verifiable criteria.

Example:

```text
AC-001
Users can log in with valid credentials.

AC-002
Invalid credentials return an authentication error.

AC-003
Five consecutive failed attempts lock the account.
```

If the specification already provides IDs, preserve them.

If criteria are unnumbered, assign internal IDs.

### 3. Create an implementation map

Before editing code, identify likely components.

Example:

```text
AUTH-001

AC-001
Tests:
- test_login_success

Implementation:
- src/auth/service.py
- src/api/routes/auth.py

AC-002
Tests:
- test_login_invalid_password

Implementation:
- src/auth/service.py
```

This prevents random coding without requirement coverage.

### 4. Inspect the existing system

Before implementing:

- inspect existing architecture
- inspect related code
- inspect current tests
- inspect project conventions
- identify reusable components
- identify existing APIs and schemas

Prefer extending established patterns over creating parallel architecture.

### 5. Write failing tests

For each acceptance criterion, create a test when practical.

Tests should represent behaviour, not implementation details.

Example:

```python
def test_auth001_ac001_valid_login():
    ...
```

Good tests answer:

> Does the system satisfy this requirement?

Avoid tests that simply duplicate internal code structure.

### 6. Confirm tests fail for the expected reason

A new test should initially demonstrate the missing behaviour.

If it passes immediately:

- the feature may already exist
- the test may be weak
- the acceptance criterion may already be satisfied

Investigate before continuing.

### 7. Implement the minimum required behaviour

Write the smallest clean implementation that satisfies the acceptance criteria.

Prefer:

- existing abstractions
- simple control flow
- established project patterns
- typed interfaces
- clear error handling

Avoid unrelated refactoring unless required.

### 8. Maintain traceability

Where useful, associate implementation and tests with the specification ID.

Examples:

```python
# Implements AUTH-001 / AC-003
```

or:

```python
def test_auth001_ac003_account_lockout():
    ...
```

Do not fill production code with unnecessary traceability comments.

Use identifiers where they provide meaningful value.

### 9. Run verification

Run:

- relevant unit tests
- integration tests
- type checking
- linting
- security checks when relevant
- broader regression tests when practical

Never claim a check passed unless it was actually executed.

### 10. Check every acceptance criterion

Create a traceability matrix.

Example:

| AC | Requirement | Test | Implementation | Status |
|---|---|---|---|---|
| AC-001 | Valid login | `test_login_success` | `auth.service.login` | PASS |
| AC-002 | Reject invalid password | `test_invalid_password` | `auth.service.login` | PASS |
| AC-003 | Lock after 5 failures | `test_account_lockout` | `auth.lockout` | PASS |

Allowed statuses:

```text
PASS
FAIL
PARTIAL
BLOCKED
NOT TESTED
```

Do not mark PARTIAL as PASS.

### 11. Check scope

Before finishing, inspect the diff.

Ask:

- Did implementation stay within the specification?
- Did unrelated functionality change?
- Did new dependencies become necessary?
- Were undocumented assumptions introduced?
- Was unnecessary complexity added?

Remove accidental scope creep.

### 12. Report unresolved issues

If a requirement could not be implemented, explicitly explain:

```text
AC-004 — BLOCKED

Reason:
External API credentials are unavailable.

Implemented:
Request interface and mocks.

Still required:
Live integration verification.
```

Never hide incomplete criteria.

## Implementation rules

### Follow existing architecture

Prefer existing:

- services
- repositories
- schemas
- dependency injection
- API conventions
- error models
- test fixtures

### Keep changes minimal

Do not use a small specification as permission for a large rewrite.

### Preserve backward compatibility

Unless the specification explicitly introduces a breaking change.

### Structured data

When interacting between application layers or services, prefer structured schemas.

Example:

```json
{
  "status": "success",
  "data": {},
  "error": null
}
```

### Security

For authentication, authorization, file access, secrets, permissions, or user-owned data, include negative tests.

Examples:

```text
User A can access User A resource.
User B cannot access User A resource.
Unauthenticated user cannot access protected resource.
Malformed identifier does not bypass authorization.
```

### AI features

For LLM/RAG features, separate deterministic requirements from probabilistic behaviour.

Deterministic checks may include:

- correct document retrieval
- metadata filtering
- JSON schema validity
- source attribution
- tool invocation
- permission boundaries

Do not create tests that expect exact natural-language responses unless the specification explicitly requires exact text.

## Completion gate

Do not declare the specification complete until:

- [ ] specification was fully read
- [ ] every acceptance criterion was identified
- [ ] every criterion has implementation status
- [ ] applicable criteria have tests
- [ ] relevant tests were executed
- [ ] implementation follows project architecture
- [ ] regressions were checked
- [ ] unresolved criteria are explicitly documented
- [ ] traceability report was produced

## Final output

Return:

# Implementation Report — `<SPEC-ID>`

## Summary

Short explanation of what was implemented.

## Files changed

```text
path/to/file
path/to/test
```

## Acceptance criteria

| AC | Requirement | Verification | Implementation | Status |
|---|---|---|---|---|
| AC-001 | ... | ... | ... | PASS |

## Tests

```text
Command:
...

Result:
...
```

## Design decisions

List only important architectural decisions.

## Remaining issues

State:

```text
None
```

or describe each unresolved requirement.

## Coverage

Example:

```text
Acceptance criteria satisfied: 5 / 5
Acceptance criteria tested: 5 / 5
Blocked: 0
```

Never describe the specification as complete when required criteria remain failed, partial, blocked, or untested.
