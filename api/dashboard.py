from __future__ import annotations


def dashboard_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SentinelPay Live Dashboard</title>
  <style>
    :root {
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --surface: #f7f8fb;
      --panel: #ffffff;
      --accent: #0f766e;
      --danger: #b91c1c;
      --warn: #b45309;
      --ok: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--surface);
    }
    header {
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    header span { color: var(--muted); font-size: 14px; }
    main {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--line);
      min-height: calc(100vh - 68px);
    }
    section {
      background: var(--panel);
      padding: 18px;
      min-width: 0;
    }
    h2 { margin: 0 0 14px; font-size: 16px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 700; }
    .risk { font-weight: 800; }
    .CRITICAL, .HIGH, .SUSPICIOUS { color: var(--danger); }
    .ELEVATED { color: var(--warn); }
    .LOW, .LEGITIMATE { color: var(--ok); }
    .empty {
      color: var(--muted);
      padding: 16px 0;
    }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SentinelPay Live Dashboard</h1>
      <span>Polling recent scores, fraud queue, and Maya's triage output.</span>
    </div>
    <span id="updated">Waiting for data</span>
  </header>
  <main>
    <section>
      <h2>Live Scores</h2>
      <div id="events" class="empty">No scores yet.</div>
    </section>
    <section>
      <h2>Fraud Queue</h2>
      <div id="cases" class="empty">No open cases yet.</div>
    </section>
  </main>
  <script>
    const updated = document.querySelector("#updated");
    const eventsRoot = document.querySelector("#events");
    const casesRoot = document.querySelector("#cases");

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function renderEvents(events) {
      if (!events.length) {
        eventsRoot.className = "empty";
        eventsRoot.textContent = "No scores yet.";
        return;
      }
      eventsRoot.className = "";
      eventsRoot.innerHTML = `
        <table>
          <thead><tr><th>Time</th><th>Transaction</th><th>Risk</th><th>Label</th><th>Maya</th></tr></thead>
          <tbody>
            ${events.map((event) => `
              <tr>
                <td>${escapeHtml(new Date(event.timestamp).toLocaleTimeString())}</td>
                <td>${escapeHtml(event.transaction_id)}</td>
                <td class="risk">${Number(event.risk_score).toFixed(4)}</td>
                <td class="${escapeHtml(event.label)}">${escapeHtml(event.label)}</td>
                <td>${escapeHtml(event.analyst_summary || event.model_name)}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    }

    function renderCases(cases) {
      if (!cases.length) {
        casesRoot.className = "empty";
        casesRoot.textContent = "No open cases yet.";
        return;
      }
      casesRoot.className = "";
      casesRoot.innerHTML = `
        <table>
          <thead><tr><th>Case</th><th>Severity</th><th>Queue</th><th>Summary</th></tr></thead>
          <tbody>
            ${cases.map((item) => `
              <tr>
                <td>${escapeHtml(item.case_id)}</td>
                <td class="${escapeHtml(item.severity)}">${escapeHtml(item.severity)}</td>
                <td>${escapeHtml(item.queue)}</td>
                <td>${escapeHtml(item.score?.analyst?.summary || "")}</td>
              </tr>`).join("")}
          </tbody>
        </table>`;
    }

    async function refresh() {
      const [eventsResponse, casesResponse] = await Promise.all([
        fetch("/events/recent?limit=25"),
        fetch("/cases?status=OPEN&limit=25")
      ]);
      renderEvents(await eventsResponse.json());
      renderCases(await casesResponse.json());
      updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    }

    refresh();
    setInterval(refresh, 2500);
  </script>
</body>
</html>
"""
