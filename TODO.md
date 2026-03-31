# Post-Setup TODO

After adding `git-repokit-common` as a submodule or worktree, complete these steps:

## Required

- [ ] **Configure pyproject.toml**: Add `[tool.repokit-common]` section (hooks depend on this):
  ```toml
  [tool.repokit-common]
  version-source = "your_package/_version.py"
  changelog = "CHANGELOG.md"
  repo-url = "https://github.com/YourOrg/your-project"
  tag-prefix = "v"
  private-patterns = ["private/", "local/", ".env"]
  ```
- [ ] **Create `_version.py`**: Copy the version module template into your package directory and edit the initial version values
- [ ] **Install hooks**: Run `bash scripts/repokit-common/install-hooks.sh`

## Customize (as needed)

- [ ] **VHS demo tapes** (`vhs/*.tape`): Replace `repokit-common` with your actual CLI command name
- [ ] **demo_render.py**: Rewrite with your project's output format (the existing content is a template/example)
- [ ] **Pre-push hook**: Verify the auto-detected package directory is correct for your project

## Optional

- [ ] **search_sesslog.py**: Useful if you use Claude Code -- searches session transcripts
- [ ] **extract_tool_result.py**: Useful if you use Claude Code -- extracts tool results from sessions
- [ ] **build_demo.py**: Run after customizing VHS tapes to generate demo GIFs

## Notes

- The hooks auto-detect your Python package directory (looks for `__init__.py` at the project root)
- `sync-versions.py` reads config from `pyproject.toml [tool.repokit-common]` automatically
- Update this submodule anytime: `cd scripts/repokit-common && git pull origin main`
