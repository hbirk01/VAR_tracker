import { useEffect, useState, useCallback } from "react";
import SubmitPanel from "./components/SubmitPanel";
import JobCard from "./components/JobCard";
import { getJobs } from "./api";

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getJobs()
      .then(setJobs)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleJobStarted = useCallback((job) => {
    setJobs((prev) => [job, ...prev]);
  }, []);

  const handleDelete = useCallback((jobId) => {
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
  }, []);

  // Poll running jobs to pick up final state if WS was missed
  useEffect(() => {
    const interval = setInterval(() => {
      if (jobs.some((j) => !["done", "error"].includes(j.status))) {
        getJobs().then(setJobs).catch(() => {});
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [jobs]);

  const done    = jobs.filter((j) => j.status === "done");
  const safe    = done.filter((j) => j.decision === "SAFE").length;
  const out     = done.filter((j) => j.decision === "OUT").length;
  const labeled = done.filter((j) => j.expected && j.decision && j.decision !== "unknown");
  const correct = labeled.filter((j) => j.decision === j.expected?.toUpperCase()).length;

  return (
    <div style={styles.layout}>
      <main style={styles.main}>
        <SubmitPanel onJobStarted={handleJobStarted} />

        {done.length > 0 && (
          <div style={styles.stats}>
            <Stat label="Analyzed" value={done.length} />
            <Stat label="Safe" value={safe} color="var(--safe)" />
            <Stat label="Out" value={out} color="var(--out)" />
            {labeled.length > 0 && (
              <Stat
                label="Accuracy"
                value={`${Math.round((correct / labeled.length) * 100)}%`}
                color="var(--accent)"
              />
            )}
          </div>
        )}

        <div style={styles.listHeader}>
          <h2 style={styles.listTitle}>Analysis Queue</h2>
          <span style={styles.count}>{jobs.length} job{jobs.length !== 1 ? "s" : ""}</span>
        </div>

        {loading && <p style={styles.empty}>Loading…</p>}
        {!loading && jobs.length === 0 && (
          <div style={styles.emptyState}>
            <p style={styles.emptyIcon}>🎬</p>
            <p style={styles.emptyText}>No analyses yet. Paste a YouTube URL above to get started.</p>
          </div>
        )}

        <div style={styles.list}>
          {jobs.map((job) => (
            <JobCard key={job.job_id} job={job} onDelete={handleDelete} />
          ))}
        </div>
      </main>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div style={styles.stat}>
      <span style={{ ...styles.statValue, color: color || "var(--text)" }}>{value}</span>
      <span style={styles.statLabel}>{label}</span>
    </div>
  );
}

const styles = {
  layout: {
    minHeight: "100vh",
    background: "var(--bg)",
    padding: "24px 16px 48px",
  },
  main: { maxWidth: 760, margin: "0 auto" },
  stats: {
    display: "flex",
    gap: 1,
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 10,
    overflow: "hidden",
    marginBottom: 24,
  },
  stat: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "12px 0",
    borderRight: "1px solid var(--border)",
    gap: 2,
  },
  statValue: { fontSize: 22, fontWeight: 700 },
  statLabel: { fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em" },
  listHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  listTitle: { fontSize: 15, fontWeight: 600, color: "var(--text)" },
  count: { fontSize: 12, color: "var(--muted)" },
  list: { display: "flex", flexDirection: "column", gap: 10 },
  empty: { color: "var(--muted)", textAlign: "center", padding: 32 },
  emptyState: { textAlign: "center", padding: "48px 0" },
  emptyIcon: { fontSize: 36, marginBottom: 12 },
  emptyText: { color: "var(--muted)", fontSize: 14 },
};
