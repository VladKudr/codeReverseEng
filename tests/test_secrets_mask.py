"""Подсаженные секреты: строка подключения, токен, приватный ключ (DoD 5)."""
from checks.secrets_mask import MASK, mask_file, mask_text

PLANTED = """\
rules:
  - evidence: |
      DB_URL = "postgresql://billing:S3cr3tPass@db.internal:5432/billing"
    source: billing.py:3
  - evidence: |
      API_TOKEN = "ghp_abcdefghij1234567890KLMNOPQRSTuvwx"
    source: billing.py:4
  - evidence: |
      KEY = \"\"\"-----BEGIN RSA PRIVATE KEY-----
      MIIEowIBAAKCAQEA7ZKQ
      -----END RSA PRIVATE KEY-----\"\"\"
    source: crypto.py:10
"""


def test_planted_connection_string_token_and_key_are_masked():
    masked, events = mask_text(PLANTED)
    kinds = {e["kind"] for e in events}
    assert "строка подключения" in kinds
    assert "токен GitHub" in kinds
    assert "приватный ключ" in kinds
    assert "S3cr3tPass" not in masked
    assert "ghp_abcdefghij" not in masked
    assert "BEGIN RSA PRIVATE KEY" not in masked
    assert MASK in masked
    # контекст сохраняется: маскируется значение, а не вся строка
    assert 'postgresql://billing:' + MASK + '@db.internal:5432/billing' in masked


def test_masking_is_logged_with_line_numbers():
    _, events = mask_text(PLANTED)
    assert all(e["line"] > 0 for e in events)
    # превью не раскрывает секрет
    assert all(len(e["preview"]) <= 5 or e["preview"].endswith("…") for e in events)


def test_placeholders_and_code_expressions_are_not_masked():
    text = (
        'password = "<your-password>"\n'
        'token = os.environ["TOKEN"]\n'
        'secret = get_secret()\n'
        'password = "***MASKED***"\n'
    )
    masked, events = mask_text(text)
    assert events == []
    assert masked == text


def test_password_assignment_is_masked():
    masked, events = mask_text("password = 'hunter2xx'")
    assert masked == f"password = '{MASK}'"
    assert len(events) == 1


def test_mask_file_in_place(tmp_path):
    p = tmp_path / "artifact.yaml"
    p.write_text(PLANTED, encoding="utf-8")
    events = mask_file(p)
    assert events
    assert "S3cr3tPass" not in p.read_text(encoding="utf-8")
    # повторный вызов идемпотентен: маски не перезамаскируются
    assert mask_file(p) == []
