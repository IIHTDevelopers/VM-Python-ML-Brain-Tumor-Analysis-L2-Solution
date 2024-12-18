import unittest
from test.TestUtils import TestUtils
class ExceptionalTest(unittest.TestCase):
    def test_exception(self):
        test_obj = TestUtils()
        test_obj.yakshaAssert("TestException",True,"exception")
        print("TestException = Passed")
