-- Fake "internal company data" for the agent to query via the Postgres MCP server.
CREATE TABLE IF NOT EXISTS services (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    owner_team TEXT NOT NULL,
    tier TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id SERIAL PRIMARY KEY,
    service TEXT NOT NULL REFERENCES services(name),
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at DATE NOT NULL
);

INSERT INTO services (name, owner_team, tier) VALUES
  ('auth-api', 'platform', 'tier-1'),
  ('payments-api', 'payments', 'tier-1'),
  ('search-service', 'discovery', 'tier-2'),
  ('notification-worker', 'platform', 'tier-3')
ON CONFLICT DO NOTHING;

INSERT INTO incidents (service, severity, summary, created_at) VALUES
  ('auth-api', 'high', 'Login latency spiked to 4s after connection pool exhaustion', '2026-09-02'),
  ('auth-api', 'medium', 'Token refresh failing intermittently under load', '2026-09-14'),
  ('payments-api', 'high', 'Duplicate charges caused by retry without idempotency keys', '2026-09-08'),
  ('search-service', 'low', 'Slow queries on large result sets', '2026-09-19'),
  ('notification-worker', 'medium', 'Queue backlog grew during peak hours', '2026-09-25');
