const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function submitAnalysis(url, expected) {
  const res = await fetch(`${BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, expected: expected || null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJobs() {
  const res = await fetch(`${BASE}/api/jobs`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteJob(jobId) {
  await fetch(`${BASE}/api/jobs/${jobId}`, { method: "DELETE" });
}

export function videoUrl(jobId) {
  return `${BASE}/api/video/${jobId}`;
}

export async function getPresets() {
  const res = await fetch(`${BASE}/api/presets`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function openWebSocket(jobId, onMessage) {
  const wsBase = BASE.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/${jobId}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  return ws;
}
