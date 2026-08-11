"""Мини-приложение фикстуры: перенос средств с лимитами.

Файл существует ради тестов детерминированного слоя req-reverse: из него
извлекаются «правила», по нему считается покрытие ветвлений.
"""

MAX_TRANSFER = 10_000
BATCH_SIZE = 250


def validate_amount(amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    if amount > MAX_TRANSFER:
        raise ValueError("amount exceeds transfer limit")
    return amount


def transfer(src, dst, amount):
    validate_amount(amount)
    if src == dst:
        raise ValueError("source and destination must differ")
    return {"from": src, "to": dst, "amount": amount}


def split_batches(rows):
    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        batches.append(rows[i : i + BATCH_SIZE])
    return batches


def audit_note(event):
    # ветка без правила: остаётся непокрытой в отчёте coverage
    if event.get("kind") == "debug":
        return None
    return f"audit:{event['kind']}"
