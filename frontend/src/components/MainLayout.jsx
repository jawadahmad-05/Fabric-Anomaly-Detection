import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, PlusCircle, FileText, ArrowLeft, Menu, X } from 'lucide-react';

// ---------------------------------------------------------------------
// Same color tokens as InspectionSelector.jsx, applied via inline style
// rather than Tailwind arbitrary-value classes — keeps both files
// rendering identically regardless of the project's Tailwind/PostCSS
// pipeline, and keeps the whole app frame (sidebar + page) reading as
// one instrument instead of two different products stitched together.
// ---------------------------------------------------------------------
const C = {
  ink: '#0B0E14',
  paper: '#F4F0E6',
  accent: '#FF5A36',
  line: '#2A3142',
  muted: '#5C6478',
  white: '#F4F0E6',
};

const fontDisplay = "'Oswald', sans-serif";
const fontMono = "'JetBrains Mono', monospace";

export default function MainLayout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navItems = [
    { label: 'Overview', path: '/app/overview', icon: <LayoutDashboard size={17} strokeWidth={1.6} />, index: '01' },
    { label: 'New Inspection', path: '/app/new', icon: <PlusCircle size={17} strokeWidth={1.6} />, index: '02' },
    { label: 'Reports', path: '/app/reports', icon: <FileText size={17} strokeWidth={1.6} />, index: '03' },
  ];

  const sidebarContent = (
    <>
      <div
        className="flex items-center gap-3"
        style={{ padding: '22px 24px', borderBottom: `1px solid ${C.line}` }}
      >
        <div
          className="flex items-center justify-center"
          style={{ width: 30, height: 30, border: `1px solid ${C.accent}`, color: C.accent }}
        >
          <span style={{ fontFamily: fontMono, fontSize: 10, fontWeight: 700 }}>AI</span>
        </div>
          <span style={{ fontFamily: fontDisplay, fontSize: 15, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            ZSAD
          </span>
      </div>

      <nav className="flex-1" style={{ padding: '18px 14px' }}>
        {navItems.map((item) => {
          const active = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setSidebarOpen(false)}
              className={`flex items-center gap-3 ${active ? '' : 'nav-link'}`}
              style={{
                padding: '12px 14px',
                marginBottom: 4,
                background: active ? `${C.accent}1A` : 'transparent',
                borderLeft: active ? `2px solid ${C.accent}` : '2px solid transparent',
                color: active ? C.white : C.muted,
                textDecoration: 'none',
                transition: 'background 0.15s, color 0.15s',
              }}
            >
              <span style={{ fontFamily: fontMono, fontSize: 9, color: active ? C.accent : C.line, minWidth: 14 }}>
                {item.index}
              </span>
              {item.icon}
              <span style={{ fontFamily: fontMono, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <div style={{ padding: '16px 24px', borderTop: `1px solid ${C.line}` }}>
        <Link
          to="/"
          onClick={() => setSidebarOpen(false)}
          className="exit-link flex items-center gap-2"
          style={{
            color: C.muted, fontFamily: fontMono, fontSize: 10, fontWeight: 500,
            textTransform: 'uppercase', letterSpacing: '0.2em', textDecoration: 'none',
          }}
        >
          <ArrowLeft size={13} /> Exit to Website
        </Link>
      </div>
    </>
  );

  return (
    <div className="flex h-screen w-full overflow-hidden" style={{ background: C.paper }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        .nav-link:hover { background: #1A1F2B !important; color: #F4F0E6 !important; }
        .exit-link:hover { color: #F4F0E6 !important; }
      `}</style>

      {/* DESKTOP SIDEBAR */}
      <aside
        className="hidden md:flex flex-col shrink-0"
        style={{ width: 256, background: C.ink, color: C.white, borderRight: `2px solid ${C.accent}` }}
      >
        {sidebarContent}
      </aside>

      {/* MOBILE SIDEBAR OVERLAY */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 md:hidden"
          style={{ background: 'rgba(0,0,0,0.5)' }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* MOBILE SIDEBAR DRAWER */}
      <aside
        className={`fixed top-0 left-0 z-50 h-full md:hidden flex flex-col shrink-0 transition-transform duration-300`}
        style={{
          width: 256,
          background: C.ink,
          color: C.white,
          borderRight: `2px solid ${C.accent}`,
          transform: sidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
        }}
      >
        {sidebarContent}
      </aside>

      {/* MOBILE TOP BAR */}
      <div
        className="md:hidden fixed top-0 left-0 right-0 z-30 flex items-center justify-between"
        style={{ background: C.ink, color: C.white, borderBottom: `2px solid ${C.accent}`, padding: '14px 16px', height: 56 }}
      >
        <button
          onClick={() => setSidebarOpen(true)}
          style={{ background: 'none', border: 'none', color: C.white, cursor: 'pointer', padding: 4 }}
        >
          <Menu size={22} />
        </button>
        <span style={{ fontFamily: fontDisplay, fontSize: 14, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          ZSAD<span style={{ color: C.accent }}> Inspector</span>
        </span>
        <div style={{ width: 30 }} />
      </div>

      {/* MAIN CONTENT AREA */}
      <main
        className="flex-1 overflow-y-auto"
        style={{ background: C.paper, padding: '1rem', paddingTop: '4.5rem' }}
      >
        <div className="md:hidden" style={{ height: 0 }} />
        <Outlet />
      </main>
    </div>
  );
}