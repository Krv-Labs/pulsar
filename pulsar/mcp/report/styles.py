"""Standalone report CSS."""

from __future__ import annotations


def _render_report_styles() -> str:
    """Return the standalone report CSS."""
    return """
    :root {
      --bg: #ffffff;
      --surface: #f8f9fa;
      --surface-strong: #f1f3f4;
      --ink: #1f1f1f;
      --muted: #5f6368;
      --border: rgba(60, 64, 67, 0.14);
      --hairline: rgba(60, 64, 67, 0.10);
      --accent: #1a73e8;
      --accent-soft: rgba(26, 115, 232, 0.08);
      --positive: #147d64;
      --negative: #c2482b;
      --neutral: #6b7280;
      --radius-lg: 18px;
      --radius-md: 12px;
      --radius-sm: 999px;
      --font-heading: "Google Sans", "Google Sans Text", "Product Sans", Roboto, Arial, sans-serif;
      --font-sans: Roboto, Arial, sans-serif;
      --font-mono: "SFMono-Regular", "JetBrains Mono", "Cascadia Code", "Menlo", monospace;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: var(--font-sans);
      line-height: 1.62;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }

    a {
      color: var(--accent);
      text-decoration: none;
    }

    .report-shell {
      width: min(100%, 1320px);
      margin: 0 auto;
      padding: 28px 44px 112px;
      display: grid;
      grid-template-columns: minmax(0, 940px) 260px;
      gap: 68px;
      align-items: start;
    }

    .report-main {
      min-width: 0;
      max-width: 100%;
    }

    .report-nav {
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow-y: auto;
      padding-right: 8px;
    }

    .report-nav__brand {
      margin-bottom: 18px;
    }

    .report-nav p,
    .section-subtitle,
    .hero-copy p,
    .figure-caption,
    .finding-card p,
    .cluster-card__summary,
    .cluster-card__meta,
    .panel-note,
    .instance-card,
    .cluster-section__summary,
    .hero-list li,
    .cluster-lede,
    .overview-note {
      color: var(--muted);
    }

    .report-nav nav {
      display: grid;
      gap: 2px;
    }

    .report-nav__eyebrow,
    .section-eyebrow {
      display: inline-block;
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
    }

    .brand-mark {
      margin-bottom: 10px;
    }

    .report-nav h1,
    .hero-copy h1,
    .section-header h2,
    .cluster-section__heading h2 {
      font-family: var(--font-heading);
    }

    .report-nav h1 {
      margin: 0 0 6px;
      font-size: 1.04rem;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--ink);
    }

    .nav-group {
      display: grid;
      gap: 2px;
      margin-bottom: 18px;
    }

    .nav-label {
      margin-bottom: 8px;
    }

    .nav-link {
      display: block;
      padding: 6px 10px;
      border-radius: 10px;
      color: #6f7275;
      font-size: 0.88rem;
      font-weight: 500;
      line-height: 1.35;
      transition: color 0.18s ease, background-color 0.18s ease;
    }

    .nav-link:hover,
    .nav-link:focus-visible {
      outline: none;
      color: var(--accent);
      background: var(--accent-soft);
    }

    .nav-link.is-active {
      color: var(--accent);
      background: var(--accent-soft);
    }

    .report-nav__footer {
      display: none;
    }

    .report-stack {
      display: grid;
      gap: 64px;
      min-width: 0;
      max-width: 100%;
    }

    .report-section {
      scroll-margin-top: 28px;
      min-width: 0;
      max-width: 100%;
    }

    .hero-grid {
      display: grid;
      gap: 22px;
    }

    .hero-copy h2 {
      font-family: var(--font-heading);
      margin: 10px 0 14px;
      font-size: clamp(2.7rem, 8vw, 4.8rem);
      line-height: 0.98;
      letter-spacing: -0.045em;
      font-weight: 750;
    }

    .hero-copy p {
      font-size: 1.08rem;
      margin: 0;
    }

    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 18px;
      margin-top: 12px;
    }

    .metric-card {
      padding: 14px 0 16px;
      border-bottom: 1px solid var(--hairline);
    }

    .metric-value {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .metric-label {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 600;
    }

    .findings-grid {
      margin-top: 8px;
    }

    .hero-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
    }

    .hero-list strong {
      color: var(--ink);
    }

    .section-header {
      display: grid;
      gap: 10px;
      margin-bottom: 22px;
    }

    .section-header h2,
    .cluster-section__heading h2 {
      margin: 0;
      font-size: clamp(1.65rem, 4vw, 2.35rem);
      line-height: 1.04;
      letter-spacing: -0.035em;
      font-weight: 720;
    }

    .figure-frame {
      margin: 0;
      padding: 18px 18px 14px;
      background: var(--surface);
      border-radius: var(--radius-lg);
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
    }

    .figure-toolbar {
      display: grid;
      gap: 12px;
      justify-content: start;
      margin-bottom: 14px;
    }

    .figure-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--ink);
    }

    .figure-subtitle {
      font-size: 0.9rem;
      color: var(--muted);
      margin: 0;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .legend-chip,
    .trait-chip,
    .detail-chip,
    .signal-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      border-radius: var(--radius-sm);
      font-size: 0.78rem;
      font-weight: 600;
    }

    .legend-chip {
      padding: 0;
      background: transparent;
      color: var(--muted);
    }

    .legend-chip::before,
    .detail-dot {
      content: "";
      width: 0.65rem;
      height: 0.65rem;
      border-radius: 999px;
      background: var(--cluster-accent, var(--accent));
      flex: 0 0 auto;
    }

    .figure-caption {
      margin: 14px auto 0;
      max-width: 640px;
      font-size: 0.9rem;
      text-align: center;
    }

    .figure-caption strong {
      color: var(--ink);
    }

    .graph-wrapper svg {
      display: block;
      width: 100%;
      height: auto;
      background: transparent;
    }

    .search-shell {
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
    }

    .cluster-search {
      width: 100%;
      padding: 12px 14px;
      border-radius: var(--radius-md);
      border: none;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }

    .cluster-search:focus-visible {
      outline: 2px solid rgba(37, 99, 235, 0.16);
    }

    .cluster-grid {
      display: grid;
      gap: 0;
    }

    .cluster-card {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      padding: 18px 0;
      color: inherit;
      border-bottom: 1px solid var(--hairline);
      transition: color 0.18s ease, background-color 0.18s ease;
    }

    .cluster-card:hover,
    .cluster-card:focus-visible {
      outline: none;
      color: var(--accent);
      background: linear-gradient(90deg, rgba(248, 249, 250, 0.95), rgba(248, 249, 250, 0));
    }

    .cluster-card__topline,
    .cluster-section__topline {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }

    .cluster-id {
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 600;
    }

    .cluster-share {
      font: 600 0.95rem/1 var(--font-mono);
      color: var(--ink);
    }

    .cluster-card h3,
    .cluster-section__heading h2 {
      margin: 0;
    }

    .cluster-card h3 {
      font-family: var(--font-heading);
      font-size: 1.22rem;
      line-height: 1.12;
      letter-spacing: -0.02em;
      font-weight: 700;
      color: var(--ink);
    }

    .cluster-card__summary {
      margin: 8px 0 12px;
      font-size: 0.95rem;
    }

    .bar-track {
      width: 100%;
      height: 4px;
      border-radius: 999px;
      background: rgba(95, 99, 104, 0.12);
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: var(--cluster-accent, var(--accent));
    }

    .cluster-card__traits,
    .cluster-section__traits,
    .instance-details {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .trait-chip,
    .detail-chip {
      padding: 0;
      background: transparent;
      color: var(--ink);
      font-size: 0.84rem;
    }

    .detail-chip strong {
      font-weight: 700;
    }

    .heatmap-shell {
      overflow-x: auto;
      border-radius: var(--radius-lg);
      background: var(--surface);
      padding: 4px 0;
      min-width: 0;
      max-width: 100%;
    }

    .heatmap-table,
    .data-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 42rem;
    }

    .heatmap-table th,
    .heatmap-table td,
    .data-table th,
    .data-table td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--hairline);
      text-align: left;
      vertical-align: top;
    }

    .heatmap-table th,
    .data-table th {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 600;
    }

    .heatmap-table tbody tr:last-child td,
    .data-table tbody tr:last-child td {
      border-bottom: none;
    }

    .heatmap-table td:first-child,
    .heatmap-table th:first-child {
      min-width: 12.5rem;
    }

    .heat-cell {
      text-align: center;
      white-space: nowrap;
    }

    .heat-value {
      display: block;
      font-family: var(--font-sans);
      font-size: 0.8rem;
      color: var(--ink);
      font-weight: 500;
      line-height: 1.15;
    }

    .heat-subvalue {
      display: block;
      margin-top: 2px;
      font-size: 0.64rem;
      color: var(--muted);
      letter-spacing: 0.01em;
      line-height: 1.15;
    }

    .heat-positive {
      background: rgba(20, 125, 100, var(--heat-alpha, 0));
    }

    .heat-negative {
      background: rgba(242, 153, 0, var(--heat-alpha, 0));
    }

    .heatmap-subsection {
      margin-top: 18px;
    }

    .heatmap-subsection h3 {
      margin: 0 0 8px;
      font-size: 0.98rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .heatmap-subsection p {
      margin: 0 0 12px;
      font-size: 0.9rem;
      color: var(--muted);
    }

    .cluster-section {
      padding-top: 6px;
      border-top: 1px solid var(--hairline);
    }

    .cluster-section__heading {
      display: grid;
      gap: 12px;
      margin-bottom: 22px;
    }

    .cluster-section__summary {
      margin: 0;
      font-size: 1rem;
      max-width: 700px;
    }

    .cluster-lede {
      font-size: 1rem;
      margin: 0;
    }

    .overview-note {
      margin: 0;
      font-size: 1rem;
    }

    .cluster-detail-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 26px;
      align-items: start;
    }

    .cluster-secondary-stack {
      display: grid;
      gap: 26px;
    }

    .panel {
      padding: 0;
    }

    .panel h3 {
      margin: 0 0 10px;
      font-size: 1rem;
      line-height: 1.25;
      letter-spacing: -0.01em;
      font-weight: 700;
    }

    .panel-note {
      margin: 0 0 12px;
      font-size: 0.9rem;
    }

    .data-table {
      min-width: 100%;
    }

    .sparkline {
      font-family: var(--font-mono);
      color: var(--accent);
      font-size: 1.1rem;
      letter-spacing: -0.08em;
    }

    code {
      font-family: var(--font-mono);
      font-size: 0.86em;
      color: var(--ink);
      background: var(--surface);
      padding: 0.16rem 0.38rem;
      border-radius: 0.45rem;
    }

    .signal-badge {
      padding: 0.34rem 0.62rem;
      background: transparent;
    }

    .signal-pos {
      color: var(--positive);
    }

    .signal-neg {
      color: var(--negative);
    }

    .signal-neu {
      color: var(--neutral);
    }

    .instances-grid {
      display: grid;
      gap: 12px;
    }

    .instance-card {
      padding: 14px 16px;
      background: var(--surface);
      border-radius: var(--radius-md);
    }

    .instance-label {
      margin-bottom: 10px;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      font-weight: 700;
    }

    .empty-state {
      padding: 14px 16px;
      border-radius: var(--radius-md);
      background: var(--surface);
      color: var(--muted);
      font-size: 0.92rem;
    }

    .config-appendix {
      display: grid;
      gap: 14px;
      padding-top: 8px;
      border-top: 1px solid var(--hairline);
      min-width: 0;
      max-width: 100%;
    }

    .config-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    .config-toolbar p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .copy-button {
      appearance: none;
      border: none;
      background: var(--surface);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 10px;
      font: 500 0.9rem/1 var(--font-sans);
      cursor: pointer;
      transition: background-color 0.18s ease, color 0.18s ease;
    }

    .copy-button:hover,
    .copy-button:focus-visible {
      outline: none;
      background: var(--accent-soft);
      color: var(--accent);
    }

    .config-textarea {
      width: 100%;
      min-height: 320px;
      max-width: 100%;
      resize: vertical;
      border: none;
      border-radius: var(--radius-md);
      background: var(--surface);
      color: var(--ink);
      padding: 18px;
      font: 0.88rem/1.55 "Roboto Mono", var(--font-mono);
      white-space: pre;
      overflow: auto;
    }

    .config-textarea:focus-visible {
      outline: 2px solid rgba(26, 115, 232, 0.16);
    }

    @media (max-width: 1180px) {
      .report-shell {
        grid-template-columns: minmax(0, 1fr);
        gap: 36px;
      }

      .report-nav {
        position: static;
        max-height: none;
        overflow: visible;
        order: -1;
      }

      .report-nav nav {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      }

      .hero-metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
      .report-shell {
        padding: 20px 20px 72px;
      }

      .report-stack {
        gap: 44px;
      }

      .hero-copy h2 {
        font-size: 2.55rem;
      }

      .hero-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .cluster-card {
        grid-template-columns: 1fr;
        gap: 12px;
      }
    }
    """
