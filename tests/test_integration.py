import unittest2
from process import process_data

TEST_DATA_PATH = 'tests/test_data.csv'


class TestCase(unittest2.TestCase):

    def setUp(self):
        pass

    def test_integration(self):
        print process_data(training_data_path=TEST_DATA_PATH)

if __name__ == '__main__':
    unittest2.main()
