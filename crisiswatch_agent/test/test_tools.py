import unittest
from unittest.mock import patch, MagicMock
from crisiswatch_agent.tools.fetch import fetch_crisiswatch_data
from crisiswatch_agent.tools.search import search_reports_rag
from crisiswatch_agent.tools.summarize import summarize_reports
import os
import sqlite3

TEST_DB_PATH = "test_crisiswatch.db"


class TestCrisisWatchTools(unittest.TestCase):

    def setUp(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

        # Create required tables for all tests
        conn = sqlite3.connect(TEST_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE reports (
                id TEXT PRIMARY KEY,
                date TEXT,
                title TEXT,
                summary TEXT,
                region TEXT
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE embeddings (
                report_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        """
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    @patch("crisiswatch_agent.tools.fetch.requests.get")
    def test_fetch_and_cache_crisis_data(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """<html><body><div class='views-row'>
        <span class='date-display-single'>July 2023</span>
        <a href='/report/123'>Conflict in Region A</a>
        <div class='field-content'>Summary of Region A</div>
        </div></body></html>"""

        fetch_crisiswatch_data(db_path=TEST_DB_PATH)

        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM reports")
        rows = cursor.fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "July 2023")
        conn.close()

    def test_query_reports(self):
        # Setup sample database
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        # cursor.execute("CREATE TABLE reports (id TEXT PRIMARY KEY, date TEXT, title TEXT, summary TEXT)")
        cursor.execute(
            "INSERT INTO reports (id, date, title, summary, region) VALUES (?, ?, ?, ?, ?)",
            ("1", "July 2023", "Conflict", "Some summary", "Region A"),
        )
        conn.commit()
        conn.close()

        results = search_reports_rag("Conflict", db_path=TEST_DB_PATH)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Conflict")

    @patch("crisiswatch_agent.tools.search.embed_text")
    def test_search_reports_rag(self, mock_embed):
        # Mock embeddings
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        # cursor.execute("CREATE TABLE reports (id TEXT PRIMARY KEY, date TEXT, title TEXT, summary TEXT)")
        # cursor.execute("INSERT INTO reports VALUES (?, ?, ?, ?)", ("1", "July 2023", "Region A", "A summary"))
        cursor.execute(
            "INSERT INTO reports (id, date, title, summary, region) VALUES (?, ?, ?, ?, ?)",
            ("1", "July 2023", "Conflict", "Some summary", "Region A"),
        )
        conn.commit()
        conn.close()

        with patch("crisiswatch_agent.tools.search.embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [[0.1, 0.2, 0.3]]
            result = search_reports_rag("conflict", top_k=1, db_path=TEST_DB_PATH)
            self.assertIn("1", result)

    def test_summarize_reports(self):
        summaries = [
            {"title": "Conflict in X", "summary": "Escalating violence in X."},
            {"title": "Conflict in Y", "summary": "Political unrest in Y."},
        ]
        result = summarize_reports(summaries, db_path=TEST_DB_PATH)
        self.assertIn("X", result)
        self.assertIn("Y", result)


if __name__ == "__main__":
    unittest.main()
