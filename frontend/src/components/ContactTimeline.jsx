/**
 * Horizontal timeline bar showing foot vs glove contact timestamps.
 * Foot = green marker, Glove = red marker.
 * The bar spans the play window; markers are positioned proportionally.
 */
export default function ContactTimeline({ footMs, gloveMs, decision }) {
  if (footMs == null && gloveMs == null) return null;

  const pts = [footMs, gloveMs].filter((v) => v != null);
  const minT = Math.min(...pts) - 500;
  const maxT = Math.max(...pts) + 500;
  const span = maxT - minT || 1000;

  function pct(t) {
    return ((t - minT) / span) * 100;
  }

  const footPct  = footMs  != null ? pct(footMs)  : null;
  const glovePct = gloveMs != null ? pct(gloveMs) : null;

  // Safe zone = region where foot arrived before glove
  const isSafe  = footMs != null && (gloveMs == null || footMs < gloveMs);
  const safeL   = footPct  != null ? Math.min(footPct, glovePct ?? footPct)  : null;
  const safeW   = footPct != null && glovePct != null ? Math.abs(glovePct - footPct) : 0;

  return (
    <div style={s.wrap}>
      <p style={s.label}>Contact timeline</p>
      <div style={s.track}>
        {/* Margin fill between the two contacts */}
        {footPct != null && glovePct != null && (
          <div style={{
            ...s.fill,
            left: `${Math.min(footPct, glovePct)}%`,
            width: `${Math.abs(glovePct - footPct)}%`,
            background: isSafe ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.18)",
          }} />
        )}

        {/* Foot marker */}
        {footPct != null && (
          <Marker pct={footPct} color="var(--safe)" label="Foot" ms={footMs} above />
        )}

        {/* Glove marker */}
        {glovePct != null && (
          <Marker pct={glovePct} color="var(--out)" label="Glove" ms={gloveMs} above={false} />
        )}
      </div>

      {/* Legend */}
      <div style={s.legend}>
        {footMs != null  && <LegendItem color="var(--safe)" label={`Foot  ${Math.round(footMs)}ms`} />}
        {gloveMs != null && <LegendItem color="var(--out)"  label={`Glove ${Math.round(gloveMs)}ms`} />}
        {footMs != null && gloveMs != null && (
          <LegendItem
            color={isSafe ? "var(--safe)" : "var(--out)"}
            label={`Δ ${Math.abs(Math.round(gloveMs - footMs))}ms — ${isSafe ? "foot first" : "glove first"}`}
          />
        )}
      </div>
    </div>
  );
}

function Marker({ pct, color, label, ms, above }) {
  return (
    <div style={{ ...s.markerWrap, left: `${pct}%` }}>
      {above && <span style={{ ...s.markerLabel, bottom: "100%", marginBottom: 4, color }}>{label}</span>}
      <div style={{ ...s.markerLine, background: color }} />
      {!above && <span style={{ ...s.markerLabel, top: "100%", marginTop: 4, color }}>{label}</span>}
    </div>
  );
}

function LegendItem({ color, label }) {
  return (
    <span style={s.legendItem}>
      <span style={{ ...s.dot, background: color }} />
      {label}
    </span>
  );
}

const s = {
  wrap: { display: "flex", flexDirection: "column", gap: 6 },
  label: { fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.07em" },
  track: {
    position: "relative",
    height: 40,
    background: "var(--surface2)",
    border: "1px solid var(--border)",
    borderRadius: 6,
    overflow: "visible",
    margin: "12px 0 6px",
  },
  fill: {
    position: "absolute",
    top: 0,
    bottom: 0,
    borderRadius: 4,
    transition: "all 0.3s",
  },
  markerWrap: {
    position: "absolute",
    top: 0,
    bottom: 0,
    transform: "translateX(-50%)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  markerLine: {
    width: 2,
    flex: 1,
    borderRadius: 2,
  },
  markerLabel: {
    position: "absolute",
    fontSize: 10,
    fontWeight: 600,
    whiteSpace: "nowrap",
    letterSpacing: "0.03em",
  },
  legend: { display: "flex", gap: 14, flexWrap: "wrap", marginTop: 2 },
  legendItem: { display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--muted)" },
  dot: { width: 7, height: 7, borderRadius: "50%", flexShrink: 0 },
};
