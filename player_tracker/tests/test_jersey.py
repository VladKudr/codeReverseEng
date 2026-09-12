from tracker.jersey import NumberRead, NumberVotes


def test_votes_need_weight_and_dominance():
    v = NumberVotes(min_weight=1.0, dominance=2.0)
    assert v.best() is None
    v.add(NumberRead("10", 0.6))
    assert v.best() is None
    v.add(NumberRead("10", 0.6))
    assert v.best()[0] == "10"
    v.add(NumberRead("18", 0.9))  # одно ошибочное прочтение не переопределяет, но снимает уверенность
    assert v.best() is None
    v.add(NumberRead("10", 0.9))
    assert v.best()[0] == "10"


def test_relation_and_known():
    a, b = NumberVotes(), NumberVotes()
    a.set_known("7")
    assert a.relation(b) is None
    b.add(NumberRead("7", 0.9))
    b.add(NumberRead("7", 0.9))
    assert a.relation(b) is True
    c = NumberVotes()
    c.add(NumberRead("9", 0.9))
    c.add(NumberRead("9", 0.9))
    assert a.relation(c) is False


def test_garbage_ignored():
    v = NumberVotes()
    v.add(NumberRead("abc", 0.9))
    v.add(NumberRead("123", 0.9))
    v.add(None)
    assert v.best() is None and v.reads == 0
