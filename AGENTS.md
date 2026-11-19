# AGENTS.md

Instructions for AI agents working in this repository.

## Before changing anything

- Read [docs/design.md](docs/design.md). Its rules, including the
  documentation rules, apply to everything you write.
- Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md#checks) before you
  report a change as done.

## Working agreement

- The maintainer drives the design. Implement what was asked. Features,
  dependencies, configuration, CI and files that were not requested are
  proposed, not added.
- Ask before anything hard to undo or outward-facing: rewriting git history,
  force-pushing, publishing, deleting files you did not create.
- Report what happened. If a check fails or something was not verified, say
  so and show the output. Never weaken a test to make it pass.

## Writing

Your output is held to the same standard as the maintainer's.

- Do not write filler: no promotional adjectives, no summaries of what you
  just wrote, no sentence that tells the reader nothing new.
- Do not assign importance. Emphasis, ordering and length must follow from
  what is actually at stake, and you must be able to justify each.
- Do not state a fact you have not checked. Verify code behaviour by running
  it, figures by looking at them, and citations against their publisher
  record.
