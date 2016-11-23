#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest2
from app.process import process_data
from server.django_server.utils import get_trending_links

TEST_DATA_PATH = 'tests/test_data.csv'


class TestCase(unittest2.TestCase):

    def setUp(self):
        pass

    def test_integration(self):
        data_frame = process_data(training_data_path=TEST_DATA_PATH)
        self.check_basic_github_repo(data_frame.loc[0, :])
        self.check_homepage_github_repo(data_frame.loc[1, :])

    def check_basic_github_repo(self, data_frame):
        self.assertEqual(data_frame['repository'], 'hpi-swt2/event-und-raumplanung')
        self.assertEqual(data_frame['owner'], 'hpi-swt2')
        self.assertEqual(data_frame['name'], 'event-und-raumplanung')
        self.assertEqual(data_frame['label'], 'DEV')
        self.assertEqual(data_frame['description'], u'Ein Tool das die interne Planung von Events verbessern soll und dabei besonderen Fokus auf die Zuteilung von Räumen und Ausstattung legt.')
        self.assertEqual('event-und-raumplanung' in data_frame['readme'], True)
        self.assertEqual(data_frame['open_issues'], 56)
        self.assertEqual(data_frame['closed_issues'], 167)
        self.assertEqual(data_frame['open_pull_requests'], 0)
        self.assertEqual(data_frame['closed_pull_requests'], 13)
        self.assertEqual(data_frame['merged_pull_requests'], 65)
        self.assertEqual(data_frame['projects'], 0)
        self.assertEqual(data_frame['watchers'], 31)
        self.assertEqual(data_frame['stargazers'], 7)
        self.assertEqual(data_frame['forks'], 5)
        self.assertEqual(data_frame['mentionableUsers'], 48)
        self.assertEqual(data_frame['size'], 119701)
        self.assertEqual(data_frame['hasHomepage'], False)
        self.assertEqual(data_frame['isOwnerHomepage'], False)
        self.assertEqual(data_frame['commitsCount'], 1752)
        self.assertEqual(data_frame['branchesCount'], 116)

    def check_homepage_github_repo(self, data_frame):
        self.assertEqual(data_frame['repository'], 'WGierke/wgierke.github.io')
        self.assertEqual(data_frame['owner'], 'WGierke')
        self.assertEqual(data_frame['name'], 'wgierke.github.io')
        self.assertEqual(data_frame['label'], 'WEB')
        self.assertEqual(data_frame['description'], '')
        self.assertEqual(data_frame['readme'], '')
        self.assertEqual(data_frame['open_issues'], 0)
        self.assertEqual(data_frame['closed_issues'], 0)
        self.assertEqual(data_frame['open_pull_requests'], 0)
        self.assertEqual(data_frame['closed_pull_requests'], 0)
        self.assertEqual(data_frame['merged_pull_requests'], 0)
        self.assertEqual(data_frame['projects'], 0)
        self.assertEqual(data_frame['watchers'], 2)
        self.assertEqual(data_frame['stargazers'], 1)
        self.assertEqual(data_frame['forks'], 0)
        self.assertEqual(data_frame['mentionableUsers'], 1)
        self.assertEqual(data_frame['size'], 0)
        self.assertEqual(data_frame['hasHomepage'], True)
        self.assertEqual(data_frame['isOwnerHomepage'], True)
        self.assertEqual(data_frame['commitsCount'], 1)
        self.assertEqual(data_frame['branchesCount'], 1)

    def test_trending_repo_fetching(self):
        self.assertEqual(len(get_trending_links()), 25)


if __name__ == '__main__':
    unittest2.main()
