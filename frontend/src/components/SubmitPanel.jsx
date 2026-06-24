import { useState } from "react";
import { submitAnalysis } from "../api";

const STAGES = ["download", "extract", "detect", "analyze", "render"];
const STAGE_LABELS = {
  download: "Downloading",
  extract: "Extracting frames",
  detect: "Pose detection",
  analyze: "Contact timing",
  render: "Rendering",
};

export default function SubmitPanel({ onJobStarted }) {
  const [url, setUrl] = useState("");
  const [expected, setExpected] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    try {
      const job = await submitAnalysis(url.trim(), expected || null);
      onJobStarted(job);
      setUrl("");
      setExpected("");
    } catch (err) {
      setError(err.message || "Submission failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={styles.logo}>⚾</span>
        <div>
          <h1 style={styles.title}>VAR Tracker</h1>
          <p style={styles.sub}>Paste a YouTube URL of a first-base close play</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          style={styles.input}
          type="url"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
        <div style={styles.row}>
          <select
            style={styles.select}
            value={expected}
            onChange={(e) => setExpected(e.target.value)}
            disabled={loading}
          >
            <option value="">Expected outcome (optional)</option>
            <option value="safe">Safe</option>
            <option value="out">Out</option>
          </select>
          <button style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }} disabled={loading}>
            {loading ? "Submitting…" : "Analyze"}
          </button>
        </div>
        {error && <p style={styles.error}>{error}</p>}
      </form>

      <div style={styles.stageRow}>
        {STAGES.map((s, i) => (
          <div key={s} style={styles.stagePill}>
            <span style={styles.stageNum}>{i + 1}</span>
            <span style={styles.stageLabel}>{STAGE_LABELS[s]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  panel: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    padding: "24px 28px",
    marginBottom: 24,
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 14,
    marginBottom: 20,
  },
  logo: { fontSize: 36 },
  title: { fontSize: 22, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.3px" },
  sub: { color: "var(--muted)", fontSize: 13, marginTop: 2 },
  form: { display: "flex", flexDirection: "column", gap: 10 },
  input: {
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "10px 14px",
    color: "var(--text)",
    fontSize: 14,
    width: "100%",
  },
  row: { display: "flex", gap: 10 },
  select: {
    flex: 1,
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "10px 14px",
    color: "var(--text)",
  },
  btn: {
    background: "var(--accent)",
    color: "#fff",
    borderRadius: 8,
    padding: "10px 22px",
    fontWeight: 600,
    whiteSpace: "nowrap",
    transition: "opacity 0.15s",
  },
  error: { color: "var(--out)", fontSize: 13 },
  stageRow: {
    display: "flex",
    gap: 8,
    marginTop: 18,
    flexWrap: "wrap",
  },
  stagePill: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 20,
    padding: "4px 10px",
  },
  stageNum: {
    background: "var(--border)",
    color: "var(--muted)",
    borderRadius: "50%",
    width: 18,
    height: 18,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 10,
    fontWeight: 700,
  },
  stageLabel: { color: "var(--muted)", fontSize: 12 },
};
