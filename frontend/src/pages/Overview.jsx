import React, { useEffect, useState } from 'react';
import { Activity, BarChart3, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

// ---------------------------------------------------------------------
// Same color tokens as the rest of the app.
// ---------------------------------------------------------------------
const C = {
  ink: '#0B0E14',
  paper: '#F4F0E6',
  accent: '#FF5A36',
  pass: '#3DDC84',
  lineSoft: '#1A1F2B',
  muted: '#5C6478',
  mutedLight: '#A8A296',
  white: '#F4F0E6',
};

const fontDisplay = "'Oswald', sans-serif";
const fontMono = "'JetBrains Mono', monospace";

const CLEAR_LABELS = ['Clear', 'Good Fabric', 'No Defect', 'None'];

// ---------------------------------------------------------------------
// The 10 real defect categories the model targets, plus the no-anomaly
// label. This is the canonical list used everywhere else in the app
// (InspectionSelector.jsx mock data, Landing.jsx). Defect Distribution
// below always shows all 11 of these, with a real count where data
// exists and 0 where it doesn't yet — instead of only showing whatever
// happens to already be present in /all_reports.
// ---------------------------------------------------------------------
const DEFECT_CATEGORIES = [
  'Hole', 'Stain', 'Missing End', 'Broken Pick', 'Slubs',
  'Reed Marks', 'Thick and Thin Places', 'Float',
  'Bowing and Skewing', 'Filling Bar',
];
const ALL_LABELS = [...DEFECT_CATEGORIES, 'Good Fabric'];

// ---------------------------------------------------------------------
// SINGLE PLACE TO EDIT once the backend's final schema is confirmed.
// Right now this page may be pointed at a temporary stand-in backend
// (different field names, e.g. Defect_Type) while testing the frontend
// ahead of the real zero-shot backend being finished. This function
// checks both the stand-in's field names and the schema the real
// backend is expected to use, so the page works with either without
// needing edits later — only this function would need a new fallback
// key added if a third schema shows up.
// ---------------------------------------------------------------------
function normalizeRow(r) {
  const category = r.Category ?? r.Defect_Type ?? r.primary_class ?? 'Unknown';
  const confidence = r.Confidence ?? r.confidence_score ?? r.confidence ?? null;
  const timestamp = r.Timestamp ?? r.timestamp ?? null;
  const isClear = CLEAR_LABELS.includes(category);
  return { category, confidence, timestamp, isClear };
}

export default function Overview() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/all_reports', { cache: 'no-store' })
      .then((res) => res.json())
      .then((data) => {
        const normalized = (Array.isArray(data) ? data : []).map(normalizeRow);
        setRows(normalized);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Reports fetch failed', err);
        setLoading(false);
      });
  }, []);

  // ---------------------------------------------------------------------
  // Stats derived from normalized rows (see normalizeRow above), so this
  // works whether /all_reports is currently served by the temporary
  // stand-in backend or the eventual real one.
  // ---------------------------------------------------------------------
  const totalScanned = rows.length;
  const defectRows = rows.filter((r) => !r.isClear);
  const defectCount = defectRows.length;
  const qualityRate = totalScanned > 0
    ? (((totalScanned - defectCount) / totalScanned) * 100).toFixed(1)
    : '100.0';

  // Start every canonical category at 0, then add real counts from the
  // data on top. Any category name in the data that isn't part of the
  // canonical 10 + Good Fabric (e.g. leftover stand-in backend labels
  // like "Cut" or "Baekra") still gets counted and shown, just appended
  // after the canonical list rather than replacing it.
  const counts = {};
  for (const label of ALL_LABELS) counts[label] = 0;
  for (const r of rows) counts[r.category] = (counts[r.category] || 0) + 1;

  const distribution = Object.entries(counts)
    .map(([name, count]) => ({ name, count, isCanonical: ALL_LABELS.includes(name) }))
    .sort((a, b) => {
      if (a.isCanonical !== b.isCanonical) return a.isCanonical ? -1 : 1;
      return b.count - a.count;
    });

  const recent = [...rows]
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, 8);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center" style={{ height: 384, gap: 16 }}>
        <Loader2 className="animate-spin" size={32} style={{ color: C.muted }} />
        <div style={{ fontFamily: fontMono, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.2em', fontSize: 10 }}>
          Syncing Analytics...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 style={{ fontFamily: fontDisplay, fontSize: 30, textTransform: 'uppercase', letterSpacing: '-0.01em', color: C.ink, fontWeight: 600, margin: 0 }}>
          System Overview
        </h1>
        <p style={{ fontFamily: fontMono, fontSize: 11, color: C.muted, marginTop: 6, letterSpacing: '0.05em' }}>
          Real-time fabric quality metrics
        </p>
      </div>

      {/* STAT CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard title="Total Scanned" value={totalScanned} icon={<Activity size={18} strokeWidth={1.6} />} color={C.ink} />
        <StatCard title="Defects Detected" value={defectCount} icon={<AlertCircle size={18} strokeWidth={1.6} />} color={C.accent} />
        <StatCard title="Quality Rate" value={`${qualityRate}%`} icon={<CheckCircle2 size={18} strokeWidth={1.6} />} color={C.pass} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* RECENT LOGS */}
        <div style={{ background: '#fff', border: `1px solid ${C.ink}1A`, padding: 28 }}>
          <h3 className="flex items-center gap-2" style={{ fontFamily: fontDisplay, fontSize: 15, color: C.ink, textTransform: 'uppercase', marginBottom: 22, fontWeight: 600 }}>
            <BarChart3 size={16} strokeWidth={1.8} style={{ color: C.accent }} /> Recent Logs
          </h3>
          <div className="space-y-2">
            {recent.length > 0 ? recent.map((item, i) => (
              <div
                key={i}
                className="flex items-center justify-between"
                style={{ padding: '14px 16px', background: C.paper, border: `1px solid ${C.ink}0D` }}
              >
                <div className="flex items-center gap-3">
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: item.isClear ? C.pass : C.accent, display: 'inline-block' }} />
                  <div>
                    <p style={{ fontFamily: fontMono, fontSize: 11, color: C.ink, textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
                      {item.category}
                    </p>
                    <p style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, marginTop: 2 }}>{item.timestamp ?? '—'}</p>
                  </div>
                </div>
                <span style={{ fontFamily: fontMono, fontSize: 10, color: C.muted }}>
                  {item.confidence ? `${(parseFloat(item.confidence) * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
            )) : (
              <p style={{ fontFamily: fontMono, color: C.muted, textAlign: 'center', padding: '40px 0', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
                No data available
              </p>
            )}
          </div>
        </div>

        {/* DISTRIBUTION MAP (All 10 Categories + Control Group) */}
        <div style={{ background: C.ink, color: C.white, padding: 28 }}>
          <h3 style={{ fontFamily: fontDisplay, fontSize: 15, textTransform: 'uppercase', marginBottom: 22, fontWeight: 600 }}>
            Defect Distribution
          </h3>
          <div className="space-y-5 max-h-[460px] overflow-y-auto pr-1 custom-scrollbar">
            {distribution.length > 0 ? distribution.map((item) => {
              const isClear = CLEAR_LABELS.includes(item.name);
              const pct = totalScanned > 0 ? (item.count / totalScanned) * 100 : 0;
              return (
                <div key={item.name}>
                  <div className="flex justify-between" style={{ marginBottom: 8 }}>
                    <span style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: C.mutedLight }}>
                      {item.name}
                    </span>
                    <span style={{ fontFamily: fontMono, fontSize: 10, color: isClear ? C.pass : C.accent }}>
                      {item.count} {item.count === 1 ? 'unit' : 'units'}
                    </span>
                  </div>
                  <div style={{ height: 4, background: C.lineSoft, width: '100%' }}>
                    <div style={{ height: '100%', width: `${pct || 0}%`, background: isClear ? C.pass : C.accent, transition: 'width 0.4s ease' }} />
                  </div>
                </div>
              );
            }) : (
              <p style={{ fontFamily: fontMono, color: C.muted, textAlign: 'center', padding: '20px 0', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
                No data available
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon, color }) {
  return (
    <div
      className="flex items-center justify-between"
      style={{ background: '#fff', border: `1px solid ${C.ink}1A`, padding: 22 }}
    >
      <div>
        <p style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.2em', marginBottom: 8 }}>
          {title}
        </p>
        <p style={{ fontFamily: fontDisplay, fontSize: 28, color, fontWeight: 600, margin: 0 }}>{value}</p>
      </div>
      <div className="flex items-center justify-center" style={{ width: 40, height: 40, border: `1px solid ${color}33`, color }}>
        {icon}
      </div>
    </div>
  );
}