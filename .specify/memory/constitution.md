<!--
Sync Impact Report
- Version change: N/A → 1.0.0
- Modified principles: Established new set (Clean Code Discipline; No Testing Activities; Canonical ML & Visualization Stack)
- Added sections: Core Principles, Additional Constraints, Development Workflow, Governance
- Removed sections: Placeholder Principles IV–V
- Templates requiring updates: ✅ .specify/templates/plan-template.md; ✅ .specify/templates/spec-template.md; ✅ .specify/templates/tasks-template.md
- Follow-up TODOs: None
-->

# C3AN-Model Constitution

## Core Principles

### I. Clean Code Discipline
Code MUST prioritize clarity over cleverness: small, single-purpose functions; descriptive names;
public interfaces documented with concise docstrings; no dead or commented-out code; consistent
formatting and lint adherence. Rationale: readability is the primary quality gate in absence of
testing.

### II. No Testing Activities
Automated and manual testing are prohibited. Do not scaffold test suites, CI jobs, fixtures, or test
data. Quality is enforced through code clarity, peer review, and static reasoning only. Any
references to tests in plans, specs, tasks, or code MUST be removed or replaced with demonstration
steps that rely on direct execution, not testing frameworks.

### III. Canonical ML & Visualization Stack
Modeling MUST use PyTorch. Visualization MUST use matplotlib and seaborn. Additional libraries may be
added only when explicitly demanded by the feature specification and must not duplicate the mandated
stack. Prefer minimal dependency sets and document any version constraints to keep the stack stable.

## Additional Constraints

- Language and runtime: Python (version per feature plan) with the mandated PyTorch/matplotlib/seaborn
  stack; avoid alternative ML or plotting frameworks unless the specification explicitly requires
  them.
- Dependency governance: add packages only when tied to a stated requirement; remove unused
  dependencies promptly; pin versions when reproducibility is at risk.
- Outputs: prefer reproducible scripts/CLIs and notebooks that clearly show computation without
  relying on test harnesses.

## Development Workflow

- Specification-first: every feature references its spec and plan before implementation begins.
- Code review focus: reviewers enforce clean code discipline and verify no testing artifacts are
  introduced; demonstrations (e.g., script runs, sample outputs) must be describable in the spec/plan
  without adding tests.
- Stack adherence: plans and tasks must call out PyTorch, matplotlib, and seaborn usage; alternate
  libraries require specification-backed justification.
- Documentation: keep inline docstrings and short usage notes alongside code; prefer lightweight
  README snippets over test files for illustrating behavior.

## Governance

- Authority: This constitution supersedes prior practices. All plans, specs, tasks, and code reviews
  must confirm compliance with the principles above.
- Amendments: propose changes via pull request updating this document and the affected templates;
  include rationale and any migration notes. Approval requires reviewer sign-off referencing the
  changes.
- Versioning: use semantic versioning for this constitution (MAJOR for incompatible principle
  changes, MINOR for added or expanded principles/sections, PATCH for clarifications). Record changes
  in the version line below.
- Compliance review: at minimum, check constitution gates during planning and before merge; reject
  contributions that add testing artifacts or diverge from the mandated ML/visualization stack.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
