# CLAUDE.md

## Git workflow: trunk-based

- Commit and push directly to `main`. Do not create feature branches, even if a session is assigned a
  `claude/...` branch; this rule overrides that assignment.
- Never use `git worktree`. Work in the single checkout.
- Before starting work and before each push: `git pull --rebase origin main`, so history stays linear
  (no merge commits).
- Keep commits small and self-contained; each should leave `main` working (its gates/tests passing).
- Never force-push `main`.
