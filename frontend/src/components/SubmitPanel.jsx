import { useState, useEffect } from "react";
import { submitAnalysis, getPresets } from "../api";

const DECISION_COLOR = {
  SAFE: "#22c55e",
  OUT: "#ef4444",
  TOO_CLOSE: "#f59e0b",
  unknown: "#6b7280",
  unsuitable: "#6b7280",
};

export default function SubmitPanel({ onJobStarted }) {
  const [url, setUrl] = useState("");
  const [expected, setExpected] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [presets, setPresets] = useState([]);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!url.trim()) return;
    await doSubmit(url.trim(), expected || null);
  }

  async function doSubmit(submitUrl, submitExpected) {
    setLoading(true);
    setError("");
    try {
      const job = await submitAnalysis(submitUrl, submitExpected);
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
    <div style={s.panel}>
      {/* Header */}
      <div style={s.header}>
        <span style={s.logo}>⚾</span>
        <div>
          <h1 style={s.title}>VAR Tracker</h1>
          <p style={s.sub}>AI-powered first base contact analysis</p>
        </div>
      </div>

      {/* URL form */}
      <form onSubmit={handleSubmit} style={s.form}>
        <input
          style={s.input}
          type="url"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
        <div style={s.row}>
          <select
            style={s.select}
            value={expected}
            onChange={(e) => setExpected(e.target.value)}
            disabled={loading}
          >
            <option value="">Expected outcome (optional)</option>
            <option value="safe">Safe</option>
            <option value="out">Out</option>
          </select>
          <button style={{ ...s.btn, opacity: loading ? 0.55 : 1 }} disabled={loading}>
            {loading ? "Submitting…" : "Analyze"}
          </button>
        </div>
        {error && <p style={s.error}>{error}</p>}
      </form>

      {/* Preloaded presets */}
      {presets.length > 0 && (
        <div style={s.presetsWrap}>
          <p style={s.presetsLabel}>Quick launch — already in catalog</p>
          <div style={s.presetGrid}>
            {presets.map((p) => (
              <PresetButton
                key={p.id}
                preset={p}
                onSelect={() => doSubmit(p.url, p.expected)}
                disabled={loading}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PresetButton({ preset, onSelect, disabled }) {
  const dc = DECISION_COLOR[preset.decision] || DECISION_COLOR.unknown;
  const hasDec = preset.decision && preset.decision !== "unknown" && preset.decision !== "unsuitable";

  return (
    <button
      style={{ ...s.preset, opacity: disabled ? 0.5 : 1 }}
      onClick={onSelect}
      disabled={disabled}
      title={preset.url}
    >
      <div style={s.presetTop}>
        <span style={s.presetId}>{preset.id.slice(0, 8)}…</span>
        {hasDec && (
          <span style={{ ...s.presetDec, color: dc, borderColor: dc + "55", background: dc + "18" }}>
            {preset.decision}
            {preset.margin_ms != null && (
              <span style={{ opacity: 0.7, fontWeight: 400 }}> {Math.abs(Math.round(preset.margin_ms))}ms</span>
            )}
          </span>
        )}
        {!hasDec && preset.expected && (
          <span style={s.presetExpected}>exp: {preset.expected}</span>
        )}
      </div>
      <p style={s.presetNote}>{preset.label || "—"}</p>
    </button>
  );
}

const s = {
  panel: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    padding: "22px 24px 20px",
    marginBottom: 20,
  },
  header: { display: "flex", alignItems: "center", gap: 12, marginBottom: 18 },
  logo: { fontSize: 32 },
  title: { fontSize: 20, fontWeight: 700, color: "var(--text)", letterSpacing: "-0.3px" },
  sub: { color: "var(--muted)", fontSize: 12, marginTop: 2 },
  form: { display: "flex", flexDirection: "column", gap: 8 },
  input: {
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "10px 14px",
    color: "var(--text)",
    fontSize: 14,
    width: "100%",
  },
  row: { display: "flex", gap: 8 },
  select: {
    flex: 1,
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "10px 12px",
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
  presetsWrap: {
    marginTop: 18,
    paddingTop: 16,
    borderTop: "1px solid var(--border)",
  },
  presetsLabel: {
    fontSize: 11,
    color: "var(--muted)",
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    marginBottom: 10,
  },
  presetGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
    gap: 8,
  },
  preset: {
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: "10px 12px",
    textAlign: "left",
    cursor: "pointer",
    transition: "border-color 0.15s",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  presetTop: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 },
  presetId: { fontSize: 11, fontFamily: "monospace", color: "var(--muted)" },
  presetDec: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: "0.05em",
    padding: "2px 6px",
    borderRadius: 4,
    border: "1px solid",
    whiteSpace: "nowrap",
  },
  presetExpected: {
    fontSize: 10,
    color: "var(--muted)",
    background: "var(--border)",
    borderRadius: 4,
    padding: "2px 5px",
  },
  presetNote: { fontSize: 12, color: "var(--text)", lineHeight: 1.35, margin: 0 },
};
