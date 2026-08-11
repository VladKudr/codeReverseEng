# тестовый файл фикстуры: обязан отфильтроваться из плана этапов 1/4
import unittest


class T(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
