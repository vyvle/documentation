#!/usr/bin/env python3
"""
doc_sync.py — Generate documentation from source PRs and open a doc PR.

Reads open PRs from os-migrate source repositories, filters out release/version
bumps, asks Claude to generate AsciiDoc content for user-visible changes, then
creates a pull request in the documentation repository.

Required env vars:
  ANTHROPIC_API_KEY  — Anthropic API key
  GITHUB_TOKEN       — GitHub token (write access to DOCS_REPO, read to source repos)
  SOURCE_ORG         — GitHub org that owns the source repos (default: os-migrate)
  TARGET_REPOS       — Comma-separated list of source repos (default: vmware-migration-kit,os-migrate)
  DAYS_BACK          — How many days back to scan (default: 1)
  DOCS_REPO          — Full repo name for the documentation repo (e.g. matbu/documentation)
  DRY_RUN            — Set to "true" to skip PR creation (default: false)
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tempfile
import anthropic
from github import Github, GithubException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_ORG = os.getenv("SOURCE_ORG", "os-migrate")
TARGET_REPOS = os.getenv("TARGET_REPOS", "vmware-migration-kit,os-migrate").split(",")
DAYS_BACK = int(os.getenv("DAYS_BACK", "1"))
DOCS_REPO = os.getenv("DOCS_REPO", "os-migrate/documentation")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# Files whose presence (exclusively) marks a PR as a release bump.
RELEASE_ONLY_FILES = frozenset(
    {
        "CHANGELOG.md",
        "CHANGELOG.rst",
        "CHANGES.rst",
        "galaxy.yml",
        "go.mod",
        "go.sum",
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "VERSION",
        "version.py",
        "Cargo.toml",
        "Cargo.lock",
    }
)

# Regex patterns against PR *title* that indicate a release bump.
RELEASE_TITLE_PATTERNS = [
    re.compile(r"(?i)^release\s+v?\d+\.\d+"),
    re.compile(r"(?i)^bump\s+(version|v)\s*\d"),
    re.compile(r"(?i)^chore[\s:]+release"),
    re.compile(r"(?i)^prepare\s+v?\d+\.\d+"),
    re.compile(r"(?i)^v\d+\.\d+\.\d+(\s|$)"),
    re.compile(r"(?i)^\[release\]"),
    re.compile(r"(?i)^version\s+bump"),
    re.compile(r"(?i)^(bump|update)\s+.*version"),
]

CLAUDE_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a technical documentation writer for the OS Migrate project — an open source Ansible toolbox for parallel cloud migration (VMware to OpenStack, and OpenStack to OpenStack).

The documentation repository uses AsciiDoc format organised as:
- source/operator-*.adoc   End-user / operator guides (installation, usage, configuration, troubleshooting)
- source/developer-*.adoc  Developer / contributor documentation
- source/reference-module-*.adoc  Ansible module reference
- source/reference-role-*.adoc    Ansible role reference

AsciiDoc style guidelines used in this project:
- Sections use = (H1), == (H2), === (H3)
- Code blocks: [source,yaml] / [source,bash] / [source,json] followed by ---- delimiters
- Tables use |=== with a header row
- Each new section should have an [id="..."] anchor in kebab-case
- File-level header example:
    [id="section-anchor_context"]
    = Title Here

When a PR introduces user-visible changes (new features, new variables, new playbooks, changed behaviour, new requirements), generate a concise AsciiDoc section that operators or developers can read to understand what changed and how to use it.

Skip documentation for: pure CI changes, internal refactoring with no user-visible impact, test-only changes, dependency version bumps with no API changes, typo fixes in existing source-repo docs."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_release_bump(pr) -> bool:
    """Return True if *pr* is a release / version-bump PR that needs no doc."""
    # Title pattern check
    for pattern in RELEASE_TITLE_PATTERNS:
        if pattern.search(pr.title):
            return True

    # Label check
    label_names = {label.name.lower() for label in pr.labels}
    if label_names & {"release", "version-bump", "chore", "no-doc", "skip-docs"}:
        return True

    # All-files check — only trigger when every changed file is release-related
    try:
        changed = [Path(f.filename).name for f in pr.get_files()]
        if changed and all(
            name in RELEASE_ONLY_FILES or f.filename.endswith(".lock") or f.filename.startswith(".github/")
            for name, f in zip(changed, pr.get_files())
        ):
            return True
    except Exception:
        pass

    return False


def get_pr_diff_summary(pr, max_chars: int = 10_000) -> str:
    """Return a text summary of changed files + truncated patches."""
    parts = []
    total = 0
    files = list(pr.get_files())
    for f in files:
        if total >= max_chars:
            parts.append(f"... (truncated; {len(files)} files total)")
            break
        patch = (f.patch or "")[:3_000]
        entry = f"### {f.filename} ({f.status}, +{f.additions}/-{f.deletions})\n{patch}"
        parts.append(entry)
        total += len(entry)
    return "\n\n".join(parts)


def doc_pr_exists(gh: Github, doc_repo_name: str, source_repo: str, pr_number: int) -> bool:
    """Return True if a documentation PR already references source_repo#pr_number."""
    query = f"repo:{doc_repo_name} is:pr {source_repo}#{pr_number} in:title"
    try:
        results = list(gh.search_issues(query))
        return len(results) > 0
    except Exception:
        return False


def _make_claude_client() -> anthropic.AnthropicVertex:
    """Return an AnthropicVertex client using the same env vars as claude-code-security-review."""
    vertex_project = (os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or "").strip()
    if not vertex_project:
        raise RuntimeError("ANTHROPIC_VERTEX_PROJECT_ID env var is required.")

    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write(creds_json)
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

    return anthropic.AnthropicVertex(project_id=vertex_project, region="us-east5")


def call_claude(client: anthropic.AnthropicVertex, pr, repo_name: str) -> dict:
    """Ask Claude to analyse the PR and return structured documentation data."""
    diff_summary = get_pr_diff_summary(pr)

    user_message = f"""Analyse this Pull Request from `{SOURCE_ORG}/{repo_name}` and decide whether it requires new or updated documentation in the `{DOCS_REPO}` documentation repository.

**PR #{pr.number}: {pr.title}**
URL: {pr.html_url}

**Description:**
{pr.body or "(no description provided)"}

**Changed files and diffs:**
{diff_summary}

Return ONLY a valid JSON object (no markdown fences) with exactly these fields:
{{
  "needs_docs": <boolean>,
  "skip_reason": "<string, required when needs_docs is false>",
  "doc_file": "<relative path in the docs repo, e.g. source/operator-vmware-guide.adoc>",
  "section_id": "<AsciiDoc anchor ID in kebab-case, e.g. new-cbt-variable>",
  "section_content": "<complete AsciiDoc block for the new / updated section>",
  "pr_title": "<documentation PR title, e.g. '[Doc Sync] Add CBT variable docs ({repo_name}#{pr.number})'>",
  "pr_body": "<markdown body for the documentation PR, referencing the source PR and explaining what was added>"
}}
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4_096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def create_doc_pr(gh_client: Github, doc_repo, source_repo_name: str, pr_number: int, data: dict):
    """Create a branch with the generated content and open a PR."""
    branch = f"doc-sync/{source_repo_name}-pr-{pr_number}"
    base = doc_repo.default_branch
    base_sha = doc_repo.get_branch(base).commit.sha

    # Create branch (idempotent)
    try:
        doc_repo.create_git_ref(f"refs/heads/{branch}", base_sha)
    except GithubException as exc:
        if exc.status == 422:
            print(f"    Branch {branch!r} already exists — skipping")
            return None
        raise

    doc_file = data["doc_file"]
    section = data["section_content"].strip()
    commit_msg = f"docs: add documentation for {source_repo_name}#{pr_number}\n\nAuto-generated by doc-sync workflow."

    try:
        existing = doc_repo.get_contents(doc_file, ref=branch)
        current = existing.decoded_content.decode("utf-8")
        updated = current.rstrip() + "\n\n" + section + "\n"
        doc_repo.update_file(doc_file, commit_msg, updated, existing.sha, branch=branch)
    except GithubException as exc:
        if exc.status != 404:
            raise
        # File doesn't exist yet — create it
        doc_repo.create_file(doc_file, commit_msg, section + "\n", branch=branch)

    pr = doc_repo.create_pull(
        title=data["pr_title"],
        body=data["pr_body"],
        head=branch,
        base=base,
    )
    return pr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def main():
    gh_client = Github(os.environ["OSM_GITHUB_TOKEN"])
    claude_client = _make_claude_client()

    doc_repo = gh_client.get_repo(DOCS_REPO)
    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

    report = []

    for repo_name in TARGET_REPOS:
        repo_name = repo_name.strip()
        full_name = f"{SOURCE_ORG}/{repo_name}"
        print(f"\n{'='*60}")
        print(f"Scanning PRs in {full_name} (past {DAYS_BACK} day(s))…")
        print(f"{'='*60}")

        try:
            source_repo = gh_client.get_repo(full_name)
        except GithubException as exc:
            print(f"  Cannot access {full_name}: {exc} — skipping")
            continue

        pulls = source_repo.get_pulls(state="open", sort="updated", direction="desc")

        for pr in pulls:
            if pr.updated_at < cutoff:
                break

            entry = {
                "repo": repo_name,
                "pr_number": pr.number,
                "pr_title": pr.title,
                "pr_url": pr.html_url,
                "action": None,
                "detail": None,
            }

            print(f"\n  PR #{pr.number}: {pr.title}")

            if is_release_bump(pr):
                print("    → skipped: release bump")
                entry["action"] = "skipped"
                entry["detail"] = "release bump"
                report.append(entry)
                continue

            if doc_pr_exists(gh_client, DOCS_REPO, repo_name, pr.number):
                print("    → skipped: doc PR already exists")
                entry["action"] = "skipped"
                entry["detail"] = "doc PR already exists"
                report.append(entry)
                continue

            try:
                print("    → calling Claude to analyse…")
                data = call_claude(claude_client, pr, repo_name)
            except Exception as exc:
                print(f"    → error during Claude analysis: {exc}")
                entry["action"] = "error"
                entry["detail"] = str(exc)
                report.append(entry)
                continue

            if not data.get("needs_docs"):
                reason = data.get("skip_reason", "no documentation needed")
                print(f"    → skipped by Claude: {reason}")
                entry["action"] = "skipped"
                entry["detail"] = reason
                report.append(entry)
                continue

            if DRY_RUN:
                print(f"    → DRY RUN: would create PR — {data['pr_title']}")
                print(f"       doc_file : {data['doc_file']}")
                entry["action"] = "dry_run"
                entry["detail"] = data["pr_title"]
                report.append(entry)
                continue

            try:
                doc_pr = create_doc_pr(gh_client, doc_repo, repo_name, pr.number, data)
                if doc_pr:
                    print(f"    → created doc PR: {doc_pr.html_url}")
                    entry["action"] = "created"
                    entry["detail"] = doc_pr.html_url
                else:
                    entry["action"] = "skipped"
                    entry["detail"] = "branch already exists"
            except Exception as exc:
                print(f"    → error creating doc PR: {exc}")
                entry["action"] = "error"
                entry["detail"] = str(exc)

            report.append(entry)

    # Write a report artefact for the GH Actions upload step
    report_path = Path("doc-sync-report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {report_path}")

    # Surface any errors as a non-zero exit code so the job is marked failed
    errors = [e for e in report if e["action"] == "error"]
    if errors:
        print(f"\n{len(errors)} error(s) occurred — see report for details", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
