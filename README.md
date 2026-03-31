# $PROJECT_NAME

$DESCRIPTION

## Installation

```bash
pip install $PACKAGE_NAME
```

### From Source

```bash
git clone https://github.com/$GITHUB_ORG/$PROJECT_NAME.git
cd $PROJECT_NAME
pip install -e ".[dev]"
```

## Usage

```bash
$CLI_COMMAND --help
```

## Development

```bash
# Clone and install
git clone https://github.com/$GITHUB_ORG/$PROJECT_NAME.git
cd $PROJECT_NAME
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Install git hooks (if using repokit-common submodule)
bash scripts/repokit-common/install-hooks.sh
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

---

## Template Variables

When creating a project from this template, replace these placeholders:

| Variable | Description | Example |
|----------|-------------|---------|
| `$PROJECT_NAME` | Repository/project name | `my-cool-tool` |
| `$PACKAGE_NAME` | Python package name (underscores) | `my_cool_tool` |
| `$DESCRIPTION` | One-line project description | `A tool that does cool things` |
| `$GITHUB_ORG` | GitHub organization or user | `DazzleTools` |
| `$GITHUB_USER` | GitHub username | `djdarcy` |
| `$AUTHOR_EMAIL` | Author email | `user@example.com` |
| `$CLI_COMMAND` | CLI entry point command | `mytool` |

Quick replacement (after cloning from template):
```bash
# Linux/Mac
find . -type f -not -path "./.git/*" -exec sed -i 's/\$PROJECT_NAME/my-cool-tool/g' {} +
find . -type f -not -path "./.git/*" -exec sed -i 's/\$PACKAGE_NAME/my_cool_tool/g' {} +
# ... etc for each variable

# Or use git-repokit:
repokit adopt . --name my-cool-tool
```
