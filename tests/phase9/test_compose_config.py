import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ComposeRuntimeConfigTests(unittest.TestCase):
    def test_frontend_binds_all_interfaces_for_its_container_healthcheck(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        frontend = compose.split("  frontend:\n", 1)[1].split("\nvolumes:", 1)[0]

        self.assertIn("      HOSTNAME: 0.0.0.0", frontend)
        self.assertIn("fetch('http://127.0.0.1:3000')", frontend)

    def test_database_is_loopback_published_and_container_backend_uses_service_dns(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        database = compose.split("  db:\n", 1)[1].split("\n  backend:", 1)[0]
        backend = compose.split("  backend:\n", 1)[1].split("\n  frontend:", 1)[0]

        self.assertIn('      - "127.0.0.1:5432:5432"', database)
        self.assertIn("@db:5432/", backend)


if __name__ == "__main__":
    unittest.main()
