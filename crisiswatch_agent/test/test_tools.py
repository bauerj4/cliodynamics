import unittest
from unittest.mock import patch, MagicMock
from crisiswatch_agent.tools.fetch import prepopulate_from_urls
from crisiswatch_agent.tools.search import search_reports_rag
from crisiswatch_agent.tools.summarize import summarize_reports
import os
import tempfile
import sqlite3
import fitz  # PyMuPDF


class TestCrisisWatchTools(unittest.TestCase):
    def setUp(self):
        self.test_db_fd, self.test_db_path = tempfile.mkstemp(suffix=".db")
        conn = sqlite3.connect(self.test_db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                title TEXT,
                url TEXT,
                text TEXT,
                region TEXT,
                summary TEXT
            );
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                report_id INTEGER PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY(report_id) REFERENCES reports(id)
            );
            """
        )

        conn.commit()
        conn.close()

    def tearDown(self):
        os.close(self.test_db_fd)
        os.remove(self.test_db_path)

    def generate_sample_pdf_bytes(self) -> bytes:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test CrisisWatch report.")
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes

    def test_prepopulate_from_urls(self):

        sample_pdf = self.generate_sample_pdf_bytes()
        test_url = "https://www.crisisgroup.org/sites/default/files/crisiswatch/crisiswatch-june-2025-global-overview.pdf"

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.content = sample_pdf
            result = prepopulate_from_urls([test_url], db_path=self.test_db_path)

            self.assertIn("Added 1 reports", result)
            self.assertIn("skipped 0", result)

            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT title, date, url, text FROM reports")
            rows = cursor.fetchall()
            conn.close()

            self.assertEqual(len(rows), 1)
            title, date, url, text = rows[0]
            self.assertIn("June 2025", title)
            self.assertEqual("2025-06-01", date)
            self.assertEqual(test_url, url)
            self.assertIn("This is a test CrisisWatch report.", text)

    @patch("crisiswatch_agent.tools.search.embed_text")
    def test_search_reports_rag(self, mock_embed):
        mock_embed.return_value = [[0.1] * 384]

        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO reports (date, title, url, text) VALUES (?, ?, ?, ?)",
            ("2023-07-01", "Conflict in A", "url", "text"),
        )
        conn.commit()
        conn.close()

        result = search_reports_rag("Conflict", top_k=1, db_path=self.test_db_path)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["title"], "Conflict in A")

    def test_summarize_reports(self):
        sample_reports = [
            {"title": "Conflict in X", "summary": "Escalating violence in X."},
            {"title": "Conflict in Y", "summary": "Political unrest in Y."},
        ]
        result = summarize_reports(sample_reports, db_path=self.test_db_path)
        self.assertIn("X", result)
        self.assertIn("Y", result)


if __name__ == "__main__":
    unittest.main()
