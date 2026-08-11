"""Биллинг фикстуры: тарифы и подключение к БД (секрет подсажен намеренно)."""

DB_URL = "postgresql://billing:S3cr3tPass@db.internal:5432/billing"
API_TOKEN = "ghp_abcdefghij1234567890KLMNOPQRSTuvwx"

RATE_DEFAULT = 0.02


def rate_for(plan):
    if plan == "premium":
        return 0.01
    return RATE_DEFAULT


def charge(account, amount):
    if amount < 0:
        raise ValueError("charge amount cannot be negative")
    fee = amount * rate_for(account.get("plan"))
    return amount + fee
