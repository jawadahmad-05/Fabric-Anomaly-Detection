// src/pages/Landing.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Zap,
  Database,
  Hexagon,
  Camera,
  LayoutDashboard,
  EyeOff,
  ArrowRight,
  Crosshair,
  Users,
} from 'lucide-react';

// ---------------------------------------------------------------------
// Same color tokens as InspectionSelector.jsx / MainLayout.jsx, applied
// via inline style rather than Tailwind arbitrary-value classes — keeps
// every page in the app rendering identically regardless of the
// project's Tailwind/PostCSS pipeline.
// ---------------------------------------------------------------------
const C = {
  ink: '#0B0E14',
  paper: '#F4F0E6',
  accent: '#FF5A36',
  accentHover: '#E8492A',
  pass: '#3DDC84',
  line: '#2A3142',
  lineSoft: '#1A1F2B',
  muted: '#5C6478',
  mutedLight: '#A8A296',
  white: '#F4F0E6',
};

const fontDisplay = "'Oswald', sans-serif";
const fontMono = "'JetBrains Mono', monospace";

const DEFECT_CATEGORIES = [
  'Hole', 'Stain', 'Missing End', 'Broken Pick', 'Slubs',
  'Reed Marks', 'Thick and Thin Places', 'Float',
  'Bowing and Skewing', 'Filling Bar',
];

const teamMembers = [
  { name: 'JAWAD AHMAD', id: '221980065' },
  { name: 'MOMINA MASHYAM', id: '221980040' },
  { name: 'EMAN SHAHID', id: '221980036' },
  { name: 'TOOBA SARWAR', id: '221980070' },
];

export default function Landing() {
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) element.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div style={{ minHeight: '100vh', background: C.paper, fontFamily: 'Inter, sans-serif' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        @keyframes sweep { 0% { top: -8%; opacity: 0; } 8% { opacity: 1; } 92% { opacity: 1; } 100% { top: 104%; opacity: 0; } }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
        .scan-sweep { animation: sweep 2.4s cubic-bezier(0.4,0,0.2,1) infinite; }
        .blink-dot { animation: blink 1.4s ease-in-out infinite; }
        .nav-link:hover { color: #FF5A36 !important; }
        .feature-card:hover { border-color: #FF5A36 !important; }
        .category-pill:hover { border-color: #FF5A36 !important; }
        .category-pill:hover span { color: #F4F0E6 !important; }
        .team-card:hover { border-color: #FF5A36 !important; }
      `}</style>

      {/* --- HEADER --- */}
      <header
        className="sticky top-0 z-50"
        style={{ background: C.ink, color: C.white, borderBottom: `2px solid ${C.accent}` }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between" style={{ height: 76 }}>
            <div className="flex items-center gap-3">
              <div
                className="flex items-center justify-center"
                style={{ width: 36, height: 36, border: `1px solid ${C.accent}`, color: C.accent }}
              >
                <Hexagon size={18} strokeWidth={1.6} />
              </div>
              <span style={{ fontFamily: fontDisplay, fontSize: 18, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                ZSAD<span style={{ color: C.accent }}> Inspector</span>
              </span>
            </div>

            <nav className="hidden md:flex items-center gap-10">
              <button
                onClick={() => scrollToSection('overview')}
                className="nav-link"
                style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.muted, background: 'none', border: 'none', cursor: 'pointer' }}
              >
                Overview
              </button>
              <Link
                to="/app/new"
                className="nav-link"
                style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.muted, textDecoration: 'none' }}
              >
                New Inspection
              </Link>
              <Link
                to="/app/reports"
                className="nav-link"
                style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.muted, textDecoration: 'none' }}
              >
                Reports
              </Link>
            </nav>

            <Link
              to="/app/overview"
              className="flex items-center gap-2"
              style={{
                background: C.accent, color: C.ink, padding: '10px 22px', fontFamily: fontMono,
                fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 600, textDecoration: 'none',
              }}
            >
              <LayoutDashboard size={14} /> Dashboard
            </Link>
          </div>
        </div>
      </header>

      {/* --- HERO --- */}
      <section
        id="overview"
        className="relative flex items-center overflow-hidden"
        style={{ minHeight: 'calc(100vh - 76px)', background: C.ink, color: C.white }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 relative z-10 w-full" style={{ padding: '4rem 0' }}>
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center">
            <div className="space-y-10">
              <div>
                <div
                  className="inline-flex items-center gap-2"
                  style={{ padding: '6px 14px', border: `1px solid ${C.accent}4D`, marginBottom: 24 }}
                >
                  <Zap size={12} style={{ color: C.accent }} />
                  <span style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.accent }}>
                    Zero-Shot · Pixel-Level
                  </span>
                </div>
                <h1
                  style={{
                    fontFamily: fontDisplay, fontSize: 'clamp(2.8rem, 6vw, 5.5rem)', lineHeight: 0.95,
                    letterSpacing: '-0.01em', textTransform: 'uppercase', marginBottom: 28, fontWeight: 600,
                  }}
                >
                  Automated <br />
                  <span style={{ color: C.accent }}>Anomaly</span> <br />
                  Detection
                </h1>
                <p style={{ fontSize: 17, color: C.mutedLight, lineHeight: 1.7, maxWidth: 540 }}>
                  This system scans fabric at the pixel level flagging any patch that
                  departs from the surrounding weave, with no fixed defect list and no
                  retraining per fabric type. A separate classifier then names what it finds
                  across 10 universal textile defect categories.
                </p>
              </div>

              <div className="flex flex-wrap gap-4">
                <Link
                  to="/app/new"
                  className="flex items-center gap-3"
                  style={{
                    background: C.accent, color: C.ink, padding: '17px 34px', fontFamily: fontDisplay,
                    fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, textDecoration: 'none',
                  }}
                >
                  <Camera size={18} /> Start Inspection <ArrowRight size={16} />
                </Link>
                <button
                  onClick={() => scrollToSection('dataset')}
                  className="flex items-center gap-3"
                  style={{
                    background: 'transparent', color: C.white, padding: '17px 34px', fontFamily: fontDisplay,
                    fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600,
                    border: `1px solid ${C.line}`, cursor: 'pointer',
                  }}
                >
                  <Database size={18} /> Defect Categories
                </button>
              </div>
            </div>

            {/* SCAN TERMINAL */}
            <div className="relative hidden lg:block">
              <div style={{ background: C.ink, border: `1px solid ${C.line}`, padding: 4 }}>
                <Bracket pos={{ top: 6, left: 6, borderTop: true, borderLeft: true }} />
                <Bracket pos={{ top: 6, right: 6, borderTop: true, borderRight: true }} />
                <Bracket pos={{ bottom: 6, left: 6, borderBottom: true, borderLeft: true }} />
                <Bracket pos={{ bottom: 6, right: 6, borderBottom: true, borderRight: true }} />

                <div className="flex items-center justify-between" style={{ padding: '18px 28px', borderBottom: `1px solid ${C.line}` }}>
                  <div className="flex items-center gap-2">
                    <span className="blink-dot" style={{ width: 6, height: 6, borderRadius: '50%', background: C.pass, display: 'inline-block' }} />
                    <span style={{ fontFamily: fontMono, fontSize: 9, color: C.mutedLight, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
                      Live Scan
                    </span>
                  </div>
                  <span style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, letterSpacing: '0.15em' }}>37×37 GRID</span>
                </div>

                <div className="relative overflow-hidden" style={{ padding: 40 }}>
                  <div
                    className="scan-sweep absolute left-0 right-0"
                    style={{ height: 2, background: C.accent, boxShadow: `0 0 20px 3px ${C.accent}99` }}
                  />
                  <div className="flex items-center gap-3" style={{ marginBottom: 28 }}>
                    <Crosshair size={18} style={{ color: C.accent }} />
                    <span style={{ fontFamily: fontMono, fontSize: 12, color: C.mutedLight, textTransform: 'uppercase', letterSpacing: '0.15em' }}>
                      System Status: Active
                    </span>
                  </div>

                  <div style={{ border: `1px solid ${C.line}`, padding: 28 }}>
                    <div className="flex justify-between items-end" style={{ marginBottom: 16 }}>
                      <h4 style={{ fontFamily: fontMono, fontSize: 11, color: C.white, textTransform: 'uppercase', letterSpacing: '0.15em', margin: 0 }}>
                        Anomaly Score
                      </h4>
                      <span style={{ fontFamily: fontMono, fontSize: 26, color: C.accent, fontWeight: 700 }}>0.87</span>
                    </div>
                    <div style={{ width: '100%', background: C.lineSoft, height: 6 }}>
                      <div style={{ background: C.accent, height: '100%', width: '87%' }} />
                    </div>
                    <p style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, marginTop: 14, letterSpacing: '0.1em' }}>
                      kNN · Wasserstein · Centroid — fused
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- CHALLENGES --- */}
      <section id="problem" style={{ padding: '6rem 0', background: C.paper }}>
        <div className="flex flex-col items-center text-center" style={{ marginBottom: 64 }}>
          <h2
            style={{
              fontFamily: fontDisplay, fontSize: 'clamp(2rem, 4.5vw, 3.5rem)', textTransform: 'uppercase',
              letterSpacing: '-0.01em', color: C.ink, lineHeight: 1.05, marginBottom: 18, fontWeight: 600,
            }}
          >
            Why Traditional Inspection <br />
            <span style={{ color: C.accent, fontStyle: 'italic' }}>Is No Longer Enough</span>
          </h2>
          <div style={{ height: 2, width: 80, background: C.accent, marginBottom: 18 }} />
          <p style={{ maxWidth: 560, color: C.muted, fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
            Overcoming the limits of manual QC with pixel-level zero-shot anomaly detection
          </p>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px" style={{ background: `${C.ink}1A` }}>
            <FeatureCard icon={<EyeOff size={20} strokeWidth={1.6} style={{ color: C.accent }} />} title="Manual Risk" desc="Human speed is limited and prone to missing sub-millimeter defects." />
            <FeatureCard icon={<Zap size={20} strokeWidth={1.6} style={{ color: C.accent }} />} title="Efficiency" desc="Reduce production downtime by catching faults at the source, in real time." />
            <FeatureCard icon={<BarChart size={20} strokeWidth={1.6} style={{ color: C.accent }} />} title="Traceability" desc="Every yard of fabric is digitally logged, scored, and verifiable." />
          </div>
        </div>
      </section>

      {/* --- DEFECT CATEGORIES (was "Dataset") --- */}
      <section id="dataset" style={{ padding: '6rem 0', background: C.ink, color: C.white }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="grid lg:grid-cols-2 gap-16">
            <div>
              <span style={{ fontFamily: fontMono, fontSize: 10, color: C.accent, textTransform: 'uppercase', letterSpacing: '0.25em' }}>
                Detection Model
              </span>
              <h2 style={{ fontFamily: fontDisplay, fontSize: 'clamp(2rem, 4vw, 2.8rem)', textTransform: 'uppercase', margin: '12px 0 24px', fontWeight: 600 }}>
                Pixel-level, zero-shot
              </h2>
              <p style={{ color: C.mutedLight, fontSize: 16, lineHeight: 1.8, marginBottom: 32, maxWidth: 480 }}>
                Anomaly detection itself is unbounded  the model flags any patch that
                departs from a fabric's normal texture, whether that's a hole, a stain, a
                crack, a thread irregularity, or a color shift it has never seen labeled
                before. No retraining per defect type, no per-fabric fine-tuning.
              </p>
              <p style={{ color: C.mutedLight, fontSize: 16, lineHeight: 1.8, maxWidth: 480 }}>
                Once a region is flagged, a separate classifier names it against 10
                universal fabric defect categories — shown opposite — or marks it
                <span style={{ color: C.pass }}> Good Fabric</span> when nothing departs
                from the norm.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 self-center">
              {DEFECT_CATEGORIES.map((name, i) => (
                <div
                  key={name}
                  className="category-pill flex items-center justify-between"
                  style={{ border: `1px solid ${C.line}`, padding: '14px 18px' }}
                >
                  <span style={{ fontFamily: fontMono, fontSize: 9, color: C.muted }}>{String(i + 1).padStart(2, '0')}</span>
                  <span style={{ fontFamily: fontMono, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', color: C.mutedLight }}>
                    {name}
                  </span>
                </div>
              ))}
              <div
                className="flex items-center justify-between sm:col-span-2"
                style={{ border: `1px solid ${C.pass}4D`, padding: '14px 18px', background: `${C.pass}0D` }}
              >
                <span style={{ fontFamily: fontMono, fontSize: 9, color: C.pass }}>—</span>
                <span style={{ fontFamily: fontMono, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em', color: C.pass }}>
                  Good Fabric (no anomaly)
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- TEAM --- */}
      <section id="team" style={{ padding: '6rem 0', background: C.paper }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 text-center">
          <h2 style={{ fontFamily: fontDisplay, fontSize: 'clamp(1.8rem, 3.5vw, 2.5rem)', textTransform: 'uppercase', color: C.ink, marginBottom: 48, fontWeight: 600 }}>
            Research Team
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {teamMembers.map((member) => (
              <div
                key={member.id}
                className="team-card"
                style={{ background: C.ink, padding: '32px 20px', border: `1px solid ${C.line}` }}
              >
                <div
                  className="flex items-center justify-center mx-auto"
                  style={{ width: 48, height: 48, border: `1px solid ${C.accent}`, color: C.accent, marginBottom: 20 }}
                >
                  <Users size={20} strokeWidth={1.6} />
                </div>
                <h3 style={{ fontFamily: fontDisplay, fontSize: 13, color: C.white, textTransform: 'uppercase', letterSpacing: '0.02em', margin: 0 }}>
                  {member.name}
                </h3>
                <p style={{ fontFamily: fontMono, fontSize: 10, color: C.accent, marginTop: 8, letterSpacing: '0.1em' }}>
                  {member.id}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer style={{ padding: '3rem 0', background: C.ink, color: C.muted, borderTop: `2px solid ${C.accent}` }}>
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-6">
          <span style={{ fontFamily: fontDisplay, fontSize: 16, color: C.white, textTransform: 'uppercase', letterSpacing: '0.02em' }}>
            Anomaly Detection In Textile
          </span>
          <p style={{ fontFamily: fontMono, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
            © 2026 Automated Visual Anomaly Detection
          </p>
          <div className="flex gap-6">
            <Link to="/app/reports" style={{ fontFamily: fontMono, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.muted, textDecoration: 'none' }}>
              Logs
            </Link>
            <a href="mailto:zsad@paper.team" style={{ fontFamily: fontMono, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.2em', color: C.muted, textDecoration: 'none' }}>
              Support
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function Bracket({ pos }) {
  const style = {
    position: 'absolute',
    width: 16,
    height: 16,
    borderColor: '#FF5A36',
    pointerEvents: 'none',
    zIndex: 10,
    opacity: 0.7,
    ...(pos.top !== undefined ? { top: pos.top } : {}),
    ...(pos.bottom !== undefined ? { bottom: pos.bottom } : {}),
    ...(pos.left !== undefined ? { left: pos.left } : {}),
    ...(pos.right !== undefined ? { right: pos.right } : {}),
    borderTopWidth: pos.borderTop ? 2 : 0,
    borderBottomWidth: pos.borderBottom ? 2 : 0,
    borderLeftWidth: pos.borderLeft ? 2 : 0,
    borderRightWidth: pos.borderRight ? 2 : 0,
    borderStyle: 'solid',
  };
  return <div style={style} />;
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="feature-card" style={{ background: C.paper, padding: 36, border: `1px solid transparent` }}>
      <div className="flex items-center justify-center" style={{ width: 48, height: 48, border: `1px solid ${C.ink}1A`, marginBottom: 20 }}>
        {icon}
      </div>
      <h4 style={{ fontFamily: fontDisplay, fontSize: 16, color: C.ink, textTransform: 'uppercase', letterSpacing: '0.02em', marginBottom: 10 }}>
        {title}
      </h4>
      <p style={{ fontSize: 14, color: C.muted, lineHeight: 1.7 }}>{desc}</p>
    </div>
  );
}