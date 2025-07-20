import unittest
from unittest.mock import patch, MagicMock
from crisiswatch_agent.tools.fetch import prepopulate_from_urls
from crisiswatch_agent.tools.search import search_reports_rag
from crisiswatch_agent.tools.summarize import summarize_reports
import os
import tempfile
import sqlite3
import fitz  # PyMuPDF


CORRECT_SUMMARY = """Here is a summary of the CrisisWatch reports:\n\n**Conflict in X:**\n\n* Escalating violence in X, with reports of increased fighting and casualties.\n* The conflict has been ongoing for several months, with multiple factions vying for control of the region.\n* The United Nations has deployed troops to X to support the local authorities and provide humanitarian aid.\n* The situation remains volatile, with reports of rocket attacks and ambushes.\n\n**Conflict in Y:**\n\n* Political unrest in Y, with protests and demonstrations erupting in response to economic sanctions and political repression.\n* The government has been accused of human rights abuses and corruption, with many citizens feeling disillusioned with the ruling party.\n* The international community has been criticized for its response to the crisis, with some countries imposing economic sanctions and others providing military aid.\n* The situation remains tense, with reports of clashes between protesters and security forces.\n\n**Crisis in Z:**\n\n* A series of natural disasters have struck the region, including a devastating earthquake in Z, which has killed hundreds of people and destroyed entire communities.\n* The government has been accused of mismanaging the disaster response, with many areas still recovering from the initial impact.\n* The international community has been criticized for its response to the crisis, with some countries imposing economic sanctions and others providing humanitarian aid.\n* The situation remains unstable, with reports of looting and violence in some areas"""


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
        self.assertIsInstance(result, dict)
        self.assertEqual(result["titles"][0], "Conflict in A")

    def test_summarize_reports(self):
        sample_reports = [
            {"title": "Conflict in X", "summary": "Escalating violence in X."},
            {"title": "Conflict in Y", "summary": "Political unrest in Y."},
        ]
        result = summarize_reports(
            sample_reports, db_path=self.test_db_path, do_sample=False
        )
        self.assertIsInstance(result, str)
        self.assertAlmostEqual(result, CORRECT_SUMMARY)


if __name__ == "__main__":
    unittest.main()
