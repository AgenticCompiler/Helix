import pytest
from triton_agent_upload_server.naming import (
    slugify_workspace_name,
    build_archive_name,
    build_receipt_name,
    validate_upload_headers,
)


class TestNaming:
    def test_slugify_preserves_safe(self):
        assert slugify_workspace_name("matmul_case-01") == "matmul_case-01"

    def test_slugify_replaces_unsafe(self):
        assert slugify_workspace_name("matmul case/01") == "matmul_case_01"

    def test_slugify_collapses_underscores(self):
        assert slugify_workspace_name("a___b") == "a_b"

    def test_slugify_fallback(self):
        assert slugify_workspace_name("////") == "workspace"

    def test_build_archive_name(self):
        name = build_archive_name("20260526T141530Z", "matmul_case_01", "abc123")
        assert name == "20260526T141530Z-matmul_case_01-abc123.tar.gz"

    def test_build_receipt_name(self):
        name = build_receipt_name("20260526T141530Z", "matmul_case_01", "abc123")
        assert name == "20260526T141530Z-matmul_case_01-abc123.receipt.json"

    def test_validate_headers_valid(self):
        headers = {
            "x-triton-agent-upload-uid": "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            "x-triton-agent-upload-timestamp": "20260526T141530Z",
            "x-triton-agent-workspace-name": "test_workspace",
            "x-triton-agent-workspace-slug": "test_workspace",
            "x-triton-agent-manifest-version": "1",
        }
        result = validate_upload_headers(headers)
        assert result["upload_uid"] == "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12"
        assert result["workspace_name"] == "test_workspace"

    def test_validate_headers_missing(self):
        with pytest.raises(ValueError, match="Missing required header"):
            validate_upload_headers({})

    def test_validate_headers_bad_uid(self):
        headers = {
            "x-triton-agent-upload-uid": "short",
            "x-triton-agent-upload-timestamp": "20260526T141530Z",
            "x-triton-agent-workspace-name": "test",
            "x-triton-agent-workspace-slug": "test",
            "x-triton-agent-manifest-version": "1",
        }
        with pytest.raises(ValueError, match="upload_uid"):
            validate_upload_headers(headers)
