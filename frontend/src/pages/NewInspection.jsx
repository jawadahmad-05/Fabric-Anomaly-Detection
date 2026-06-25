import React from 'react';
import InspectionSelector from '../components/InspectionSelector';

const fontMono = "'JetBrains Mono', monospace";

export default function NewInspection() {
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <header className="mb-8">
        <h1 style={{ fontFamily: fontMono, fontSize: 22, color: '#0B0E14', textTransform: 'uppercase', letterSpacing: '0.02em', fontWeight: 700, margin: 0 }}>
          Fabric Inspection
        </h1>
        <div className="flex items-center gap-2 mt-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500" style={{ animation: 'blink 1.4s ease-in-out infinite' }}></span>
          <p style={{ fontFamily: fontMono, fontSize: 10, color: '#5C6478', margin: 0, textTransform: 'uppercase', letterSpacing: '0.15em' }}>
            AI Engine Online & Ready
          </p>
        </div>
      </header>

      <InspectionSelector />
    </div>
  );
}