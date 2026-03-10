# tests/test_regex_masker.py
import pytest
from llm_mask._regex_masker import RegexMasker


def test_masks_email():
    m = RegexMasker()
    text, mapping = m.mask("Contact: ivan@example.com")
    assert "ivan@example.com" not in text
    assert "<email_1>" in text
    assert mapping["ivan@example.com"] == "<email_1>"


def test_masks_phone_ru():
    m = RegexMasker()
    text, mapping = m.mask("Тел: +7 916 234-56-78")
    assert "+7 916 234-56-78" not in text
    assert "<phone_1>" in text
    assert mapping["+7 916 234-56-78"] == "<phone_1>"


def test_masks_ip():
    m = RegexMasker()
    text, mapping = m.mask("Host: 192.168.1.10")
    assert "<ip_1>" in text
    assert mapping["192.168.1.10"] == "<ip_1>"


def test_masks_cidr():
    m = RegexMasker()
    text, mapping = m.mask("Net: 10.0.0.0/24")
    assert "<ip_1>" in text
    assert mapping["10.0.0.0/24"] == "<ip_1>"


def test_masks_url():
    m = RegexMasker()
    text, mapping = m.mask("API: https://api.example.com/v2/users")
    assert "https://api.example.com/v2/users" not in text
    assert "<url_1>" in text


def test_url_before_email_no_double_mask():
    """Email inside URL must NOT produce two separate placeholders."""
    m = RegexMasker()
    text, mapping = m.mask("DSN: postgresql://admin:pass@db.host.com:5432/mydb")
    # The full DSN is a URL, only one placeholder expected
    assert text.count("<url_") == 1 or text.count("<secret_") == 1


def test_masks_snils():
    m = RegexMasker()
    text, mapping = m.mask("СНИЛС: 045-678-912 00")
    assert "045-678-912 00" not in text
    assert "<doc_1>" in text
    assert mapping["045-678-912 00"] == "<doc_1>"


def test_masks_inn_with_context():
    m = RegexMasker()
    text, mapping = m.mask("ООО «Рога», ИНН 7814567234, КПП 781401001")
    assert "7814567234" not in text
    assert "781401001" not in text
    assert "<doc_1>" in text


def test_masks_ogrn_with_context():
    m = RegexMasker()
    text, mapping = m.mask("ОГРН 1167847312890")
    assert "1167847312890" not in text
    assert "<doc_1>" in text


def test_masks_bank_account():
    m = RegexMasker()
    text, mapping = m.mask("Р/с: 40702810600001234567")
    assert "40702810600001234567" not in text
    assert "<doc_1>" in text


def test_masks_jwt():
    m = RegexMasker()
    text, mapping = m.mask("Token: eyJhbGciOiJIUzI1NiJ9.payload.sig")
    assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in text
    assert "<secret_1>" in text


def test_masks_stripe_key():
    m = RegexMasker()
    text, mapping = m.mask("STRIPE_KEY=sk_live_51H8xBz2eZvKYlo1N8q")
    assert "sk_live_51H8xBz2eZvKYlo1N8q" not in text
    assert "<secret_1>" in text


def test_masks_aws_key():
    m = RegexMasker()
    text, mapping = m.mask("AWS_KEY=AKIAIOSFODNN7EXAMPLE")
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "<secret_1>" in text


def test_same_value_same_placeholder():
    """Same entity always gets the same placeholder."""
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and a@b.com again")
    assert text.count("<email_1>") == 2
    assert len(mapping) == 1


def test_different_values_different_placeholders():
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and c@d.com")
    assert "<email_1>" in text
    assert "<email_2>" in text
    assert len(mapping) == 2


def test_counters_are_independent_per_type():
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and 192.168.1.1")
    assert "<email_1>" in text
    assert "<ip_1>" in text


def test_existing_placeholder_not_double_masked():
    """Text that already contains <email_1> must not be re-masked."""
    m = RegexMasker()
    text, mapping = m.mask("Contact: <email_1>")
    # Should remain unchanged — not a real email address
    assert text == "Contact: <email_1>"
    assert not mapping


def test_mask_is_stateless_between_calls():
    """Each call to mask() starts fresh — counters reset."""
    m = RegexMasker()
    _, mapping1 = m.mask("a@b.com")
    _, mapping2 = m.mask("a@b.com")
    assert mapping1 == mapping2  # same result for same input


def test_masks_inn_in_table_cell():
    """ИНН в ячейке таблицы (| ИНН | 662304567891 |) должен маскироваться."""
    m = RegexMasker()
    text, mapping = m.mask("| ИНН | 662304567891 |")
    assert "662304567891" not in text
    assert "<doc_1>" in text


def test_masks_ogrn_in_table_cell():
    m = RegexMasker()
    text, mapping = m.mask("| ОГРН | 1167847312890 |")
    assert "1167847312890" not in text
    assert "<doc_1>" in text


def test_url_no_backtick():
    """URL regex must NOT include trailing backtick from markdown code span."""
    m = RegexMasker()
    text, mapping = m.mask("`https://api.example.com/v2`")
    # The key should NOT end with a backtick
    for orig in mapping:
        assert not orig.endswith("`"), f"URL key has trailing backtick: {orig!r}"


def test_masks_sendgrid_single_segment():
    """SendGrid key without second dot-segment should be masked."""
    m = RegexMasker()
    text, mapping = m.mask("SG.ProdEmailKeyXyz789AbcDef012")
    assert "SG.ProdEmailKeyXyz789AbcDef012" not in text
    assert "<secret_1>" in text


def test_masks_aws_key_id_short():
    """AWS key IDs shorter than 20 chars (non-standard examples) are masked."""
    m = RegexMasker()
    text, mapping = m.mask("AWS_KEY=AKIAJQ6DTRPZEXAMPLE")
    assert "AKIAJQ6DTRPZEXAMPLE" not in text


def test_masks_telegram_bot_token():
    m = RegexMasker()
    text, mapping = m.mask("token: 7123456789:AAFProd_BotToken_XyzAbcDef01234")
    assert "7123456789:AAFProd_BotToken_XyzAbcDef01234" not in text
    assert "<secret_1>" in text


def test_masks_s3_path():
    m = RegexMasker()
    text, mapping = m.mask("bucket: s3://my-data-bucket/reports/q3_2024.csv")
    assert "s3://my-data-bucket/reports/q3_2024.csv" not in text
    assert "<url_1>" in text


def test_masks_ssh_key_name():
    m = RegexMasker()
    text, mapping = m.mask("SSH ключ: id_rsa_froloff_prod")
    assert "id_rsa_froloff_prod" not in text
    assert "<secret_1>" in text


def test_masks_env_var_password():
    """DB_PASSWORD=... value should be masked, key stays in text."""
    m = RegexMasker()
    text, mapping = m.mask("DB_PASSWORD=T0pS3cr3t!2024")
    assert "T0pS3cr3t!2024" not in text
    assert "DB_PASSWORD=" in text


def test_masks_env_var_db_user():
    m = RegexMasker()
    text, mapping = m.mask("DB_USER=payment_user")
    assert "payment_user" not in text
    assert "DB_USER=" in text


def test_masks_backtick_password_ru():
    """пароль `T0pS3cr3t!2024` → password value masked."""
    m = RegexMasker()
    text, mapping = m.mask("пароль `T0pS3cr3t!2024`")
    assert "T0pS3cr3t!2024" not in text
    assert "<secret_1>" in text


def test_masks_inline_password_after_keyword():
    """SSH пароль: TmpAcc3ss!Nov24 → password value masked."""
    m = RegexMasker()
    text, mapping = m.mask("SSH пароль: TmpAcc3ss!Nov24")
    assert "TmpAcc3ss!Nov24" not in text
    assert "<secret_1>" in text
