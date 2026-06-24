import { useEffect, useRef, useState } from "react";
import { openWebSocket, videoUrl, deleteJob } from "../api";
import ContactTimeline from "./ContactTimeline";

const STAGE_ORDER = ["download", "extract", "detect", "analyze", "render"];
const STAGE_LABELS = {
  download: "Download",
  extract: "Extract",
  detect: "Pose",
  analyze: "Analyze",
  render: "Render",
};

const DECISION_CONFIG = {
  SAFE:         { color: "var(--safe)",         label: "SAFE",         bg: "rgba(34,197,94,0.12)" },
  OUT:          { color: "var(--out)",           label: "OUT",          bg: "rgba(239,68,68,0.12)" },
  TOO_CLOSE:    { color: "var(--inconclusive)",  label: "TOO CLOSE",    bg: "rgba(245,158,11,0.12)" },
  unknown:      { color: "var(--muted)",         label: "UNKNOWN",      bg: "rgba(107,114,128,0.1)" },
  unsuitable:   { color: "var(--muted)",         label: "UNSUITABLE",   bg: "rgba(107,114,128,0.1)" },
};

export default function JobCard({ job: initialJob, onDelete }) {
  const [job, setJob] = useState(initialJob);
  const wsRef = useRef(null);

  useEffect(() => {
    setJob(initialJob);
  }, [initialJob]);

  useEffect(() => {
    if (["done", "error"].includes(job.status)) return;

    wsRef.current = openWebSocket(job.job_id, (event) => {
      setJob((prev) => ({ ...prev, ...event }));
    });

    return () => wsRef.current?.close();
  }, [job.job_id]);

  const isRunning = !["done", "error", "queued"].includes(job.status);
  const isDone    = job.status === "done";
  const isError   = job.status === "error";

  const stageIdx = STAGE_ORDER.indexOf(job.stage);
  const dc = DECISION_CONFIG[job.decision] || DECISION_CONFIG.unknown;

  const shortUrl = job.url.replace(/^https?:\/\/(www\.)?/, "").slice(0, 55);

  async function handleDelete() {
    await deleteJob(job.job_id);
    onDelete(job.job_id);
  }

  return (
    <div style={styles.card}>
      {/* Top row: URL + decision chip */}
      <div style={styles.topRow}>
        <div style={styles.urlWrap}>
          <a href={job.url} target="_blank" rel="noreferrer" style={styles.url}>
            {shortUrl}
          </a>
          {job.expected && (
            <span style={styles.expectedTag}>expected: {job.expected}</span>
          )}
        </div>
        <div style={styles.rightTop}>
          {isDone && (
            <span style={{ ...styles.decisionChip, color: dc.color, background: dc.bg }}>
              {dc.label}
              {job.margin_ms != null && (
                <span style={styles.margin}> {Math.abs(Math.round(job.margin_ms))}ms</span>
              )}
            </span>
          )}
          <button style={styles.deleteBtn} onClick={handleDelete} title="Remove">✕</button>
        </div>
      </div>

      {/* Stage progress bar */}
      {!isDone && !isError && (
        <div style={styles.stagesRow}>
          {STAGE_ORDER.map((s, i) => {
            const done    = i < stageIdx || (isDone);
            const active  = s === job.stage && isRunning;
            return (
              <div key={s} style={styles.stageItem}>
                <div style={{
                  ...styles.stageDot,
                  background: done ? "var(--safe)" : active ? "var(--accent)" : "var(--border)",
                  boxShadow: active ? "0 0 6px var(--accent)" : "none",
                }} />
                <span style={{ ...styles.stageText, color: active ? "var(--text)" : "var(--muted)" }}>
                  {STAGE_LABELS[s]}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Contact timeline when done */}
      {isDone && (job.foot_ms != null || job.glove_ms != null) && (
        <ContactTimeline
          footMs={job.foot_ms}
          gloveMs={job.glove_ms}
          decision={job.decision}
        />
      )}

      {/* Error */}
      {isError && (
        <p style={styles.errorMsg}>
          {(job.error || "").split("\n").pop()?.slice(0, 120)}
        </p>
      )}

      {/* Video player when done */}
      {isDone && job.video_path && (
        <video
          style={styles.video}
          src={videoUrl(job.job_id)}
          controls
          playsInline
          preload="metadata"
        />
      )}
    </div>
  );
}

const styles = {
  card: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 10,
    padding: "14px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  topRow: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
  },
  urlWrap: { display: "flex", flexDirection: "column", gap: 3, minWidth: 0 },
  url: {
    color: "var(--text)",
    fontSize: 13,
    fontWeight: 500,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  expectedTag: {
    fontSize: 11,
    color: "var(--muted)",
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 4,
    padding: "1px 6px",
    alignSelf: "flex-start",
  },
  rightTop: { display: "flex", alignItems: "center", gap: 8, flexShrink: 0 },
  decisionChip: {
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: "0.04em",
    padding: "4px 10px",
    borderRadius: 6,
    border: "1px solid currentColor",
    borderOpacity: 0.3,
    whiteSpace: "nowrap",
  },
  margin: { fontWeight: 400, fontSize: 11, opacity: 0.75 },
  deleteBtn: {
    background: "transparent",
    color: "var(--muted)",
    fontSize: 13,
    padding: "4px 6px",
    borderRadius: 4,
    transition: "color 0.15s",
  },
  stagesRow: {
    display: "flex",
    gap: 16,
    paddingTop: 4,
  },
  stageItem: { display: "flex", alignItems: "center", gap: 5 },
  stageDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    transition: "background 0.3s, box-shadow 0.3s",
  },
  stageText: { fontSize: 12, transition: "color 0.3s" },
  timingRow: { display: "flex", gap: 10, flexWrap: "wrap" },
  timingChip: {
    fontSize: 12,
    color: "var(--muted)",
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 5,
    padding: "3px 8px",
    display: "flex",
    alignItems: "center",
    gap: 5,
  },
  errorMsg: {
    fontSize: 12,
    color: "var(--out)",
    background: "rgba(239,68,68,0.08)",
    border: "1px solid rgba(239,68,68,0.2)",
    borderRadius: 6,
    padding: "6px 10px",
  },
  video: {
    width: "100%",
    borderRadius: 8,
    background: "#000",
    maxHeight: 360,
  },
};
