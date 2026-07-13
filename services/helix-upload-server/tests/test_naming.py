import pytest
from helix_upload_server.naming import (
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
            "x-helix-upload-uid": "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            "x-helix-upload-timestamp": "20260526T141530Z",
            "x-helix-workspace-name": "test_workspace",
            "x-helix-workspace-slug": "test_workspace",
            "x-helix-manifest-version": "1",
        }
        result = validate_upload_headers(headers)
        assert result["upload_uid"] == "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12"
        assert result["workspace_name"] == "test_workspace"

    def test_validate_headers_missing(self):
        with pytest.raises(ValueError, match="Missing required header"):
            validate_upload_headers({})

    def test_validate_headers_bad_uid(self):
        headers = {
            "x-helix-upload-uid": "short",
            "x-helix-upload-timestamp": "20260526T141530Z",
            "x-helix-workspace-name": "test",
            "x-helix-workspace-slug": "test",
            "x-helix-manifest-version": "1",
        }
        with pytest.raises(ValueError, match="upload_uid"):
            validate_upload_headers(headers)
