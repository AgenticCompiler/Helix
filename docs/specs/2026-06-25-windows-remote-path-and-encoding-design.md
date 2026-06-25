## Summary

Windows control hosts can fail remote eval workflows in two places: `scp` sees local absolute paths like `D:/Project/result.pt` as remote-style `host:path` arguments, and buffered `ssh/scp` subprocess output is decoded with the host locale instead of the remote session's UTF-8 output.

## Intended Behavior

- Remote artifact uploads and downloads must avoid passing Windows drive-letter absolute paths directly to `scp`.
- Buffered `ssh/scp` execution and SSH preflight errors must decode UTF-8 remote output reliably on Windows GBK consoles.

## Implementation Notes

- For `scp` transfers, run the subprocess from the local file's parent directory and pass only the local basename to `scp`.
- For buffered subprocess handling, read raw bytes and decode with UTF-8 first, falling back to the local preferred encoding with replacement when needed.
- Apply the same decoding rule to SSH preflight so authentication and connection failures stay readable.
