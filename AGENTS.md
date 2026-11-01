# Project task tracking

- Use GitHub Issues in `timkm-dev/hydropredict` as the source of truth for tasks,
  acceptance criteria, blockers and completion evidence:
  https://github.com/timkm-dev/hydropredict/issues
- At the start of task work, read the relevant issue and current repository state.
  Read Sprint I / Sprint II milestones for progress reviews. Do not rely on chat
  memory or old local progress notes as current status.
- Reference the issue in related commits and PRs. Update it with relevant results,
  remaining work and blockers; close it only when its acceptance criteria are met.
- Check existing issues before creating duplicates. Distinguish verified missing
  work from owner/supervisor verification and scope decisions.
- If GitHub access is unavailable, state that limitation and do not claim that
  an issue was read, updated or closed. Continue independent local work as useful.
- Keep credentials, recovery codes, student verification documents and restricted
  datasets out of issues and commits.

## Existing GitHub access on this workstation

- GitHub access was verified on 24 October 2026 through Git Credential Manager
  and the GitHub REST API, without a plugin or the `gh` CLI.
- The local, ignored helper `.local/github_sprint_issues.py --verify` lists issues
  read-only using the existing Git sign-in. Run it with `.venv/Scripts/python.exe`.
  Do not pass other arguments without inspecting the helper: its draft-file mode
  creates issues and milestones.
- Windows Credential Manager requires sandbox escalation here. A sandbox denial
  or missing `gh` executable does not establish that GitHub is unavailable.
- If necessary, select the existing `timkm-dev` account explicitly when requesting
  the credential. Keep credentials in memory; never print, save or paste tokens.

## Project title preference and reminder

- On 26 October 2026, the user chose the preferred title:
  **A Machine Learning–Based Early Fault Warning System for Renewable Energy Equipment**.
- When the user next resumes substantive project work, remind them to use this
  title and align the documentation's objectives and scope with it.
- The intended prediction task is advance fault warning, not remaining useful
  life estimation. This preference does not establish supervisor approval or
  authorize an automatic documentation rewrite or dataset change.
