import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from knowledge_pull_requests import (
    extract_pull_request_id_from_subject,
    extract_pull_request_ids_from_knowledge,
    filter_commit_shas_by_pull_requests,
    parse_pull_request_ids,
    resolve_pull_request_filter,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class KnowledgePullRequestTests(unittest.TestCase):
    def test_parse_pull_request_ids_splits_commas(self) -> None:
        self.assertEqual(parse_pull_request_ids(["99", "107,100"]), {99, 107, 100})

    def test_extract_pull_request_ids_from_run_summary(self) -> None:
        text = """\
## Run Summary

| Field | Value |
|---|---|
| Analyzed pull requests | 99, 107 |
"""
        self.assertEqual(extract_pull_request_ids_from_knowledge(text), {99, 107})

    def test_resolve_prefers_explicit_cli_ids(self) -> None:
        kb = "## Analyzed Pull Requests\n\n100\n"
        resolved, source = resolve_pull_request_filter({99}, kb)
        self.assertEqual(resolved, {99})
        self.assertEqual(source, "cli")

    def test_resolve_falls_back_to_knowledge_base(self) -> None:
        kb = "## Analyzed Pull Requests\n\n100, 101\n"
        resolved, source = resolve_pull_request_filter(set(), kb)
        self.assertEqual(resolved, {100, 101})
        self.assertEqual(source, "knowledge-base")

    def test_filter_commit_shas_by_pull_request(self) -> None:
        commit_to_pr = {
            "85171374766b": 99,
            "82dcaed82119": 100,
        }
        kept = filter_commit_shas_by_pull_requests(
            ["85171374766b", "82dcaed82119"],
            pull_request_filter={99},
            commit_to_pr=commit_to_pr,
        )
        self.assertEqual(kept, ["85171374766b"])

    def test_extract_pull_request_id_from_github_merge_subject(self) -> None:
        subject = "Merge pull request #352 from shengzhaotian/hw/950-perf"
        self.assertEqual(extract_pull_request_id_from_subject(subject), 352)

    def test_extract_pull_request_id_from_gitcode_merge_subject(self) -> None:
        subject = "Merge pull request !99 from feature/foo into main"
        self.assertEqual(extract_pull_request_id_from_subject(subject), 99)

    def test_extract_pull_request_id_from_gitcode_shorthand(self) -> None:
        self.assertEqual(extract_pull_request_id_from_subject("!107 optimize chunk_o"), 107)

    def test_extract_pull_request_id_ignores_unrelated_hash_reference(self) -> None:
        self.assertIsNone(extract_pull_request_id_from_subject("fix regression, see issue #352"))


if __name__ == "__main__":
    unittest.main()
