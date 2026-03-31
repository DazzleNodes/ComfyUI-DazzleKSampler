# git-repokit-common

Shared scripts, hooks, and developer tools for DazzleTools projects. Consumed as a git submodule in `scripts/`.

## Quick Start

Add to your project:

```bash
git submodule add https://github.com/DazzleTools/git-repokit-common.git scripts/repokit-common
bash scripts/repokit-common/install-hooks.sh
```

Update to latest:

```bash
cd scripts/repokit-common && git pull origin main
```

## What's Included

### Git Hooks (`hooks/`)
- **pre-commit** -- Version sync (`sync-versions.py --auto`), private content protection, large file blocking
- **post-commit** -- Refreshes version hash after commit
- **pre-push** -- Python syntax check, pytest, debug statement detection

### Version Management
- **sync-versions.py** -- Single source of truth for version bumping with git metadata (branch, build count, date, hash)
- **update-version.sh** -- Legacy bash version updater (deprecated; use sync-versions.py)

### GitHub Tools
- **gh_issue_full.py** -- Display complete issue context: timeline, cross-refs, sub-issues, comments
- **gh_sub_issues.py** -- Manage GitHub sub-issue relationships

### Claude Code Session Tools
- **search_sesslog.py** -- Search Claude Code JSONL session transcripts
- **extract_tool_result.py** -- Find and extract tool results from session data

### CLI Demo Recording
- **build_demo.py** -- Build CLI demo recordings
- **demo_render.py** -- Render demo recordings
- **vhs/** -- VHS tape templates for CLI demo recording

### Utilities
- **install-hooks.sh** -- Install git hooks from this submodule into `.git/hooks/`
- **paths.sh** -- Common path constants for scripts
- **safe_move.sh** -- Hash-verified file move with timestamp preservation

## Configuration

Projects configure repokit-common via `[tool.repokit-common]` in `pyproject.toml`:

```toml
[tool.repokit-common]
version-source = "mypackage/_version.py"
changelog = "CHANGELOG.md"
repo-url = "https://github.com/DazzleTools/my-project"
tag-prefix = "v"
private-patterns = ["private/", "local/", ".env"]
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.
