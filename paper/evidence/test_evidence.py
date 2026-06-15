from run_evidence import capture  # noqa: E402


def test_capture_writes_log(tmp_path):
    log = tmp_path / "echo.log"
    rc, out = capture(["python", "-c", "print('hello-evidence')"], log)
    assert rc == 0
    assert "hello-evidence" in out
    assert log.read_text().strip() == "hello-evidence"
