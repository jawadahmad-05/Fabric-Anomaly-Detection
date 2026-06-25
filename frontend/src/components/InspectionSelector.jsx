import React, { useState, useRef, useEffect } from 'react';
import { Camera, Layers, Upload, Loader2, Save, ArrowRight, CheckCircle, AlertTriangle, Crosshair } from 'lucide-react';

const API_BASE = '';

// ---------------------------------------------------------------------
// COLOR TOKENS
// All colors live here as real hex values applied via inline `style`,
// not Tailwind arbitrary-value classes. This is deliberate: this file
// previously used bg-[#0B0E14] style classes, which depend on Tailwind's
// JIT scanner picking them up correctly. If your project's CSS pipeline
// (multiple @tailwind entrypoints, a stale output.css, etc.) doesn't
// rebuild cleanly, those classes silently do nothing and the page looks
// colorless even though the markup is correct. Inline styles can't be
// dropped by a build step, so the look is guaranteed regardless of your
// Tailwind setup. Tailwind classes are still used for layout/spacing —
// only color is inline.
// ---------------------------------------------------------------------
const C = {
  ink: '#0B0E14',        // instrument body / panel background
  paper: '#F4F0E6',      // page background, warm off-white
  paperText: '#0B0E14',
  white: '#F4F0E6',
  accent: '#FF5A36',     // terracotta — active/alert/scan
  accentHover: '#E8492A',
  pass: '#3DDC84',       // calibration green — clear/pass
  warn: '#E8B23B',       // amber — medium severity
  line: '#2A3142',       // panel borders
  lineSoft: '#1A1F2B',   // inner dividers
  muted: '#5C6478',      // secondary label text
  mutedDark: '#3A4257',  // tertiary / disabled text
  mutedLight: '#A8A296',
};

const fontDisplay = "'Oswald', sans-serif";
const fontMono = "'JetBrains Mono', monospace";

function severityColor(sev) {
  if (sev === 'high') return C.accent;
  if (sev === 'medium') return C.warn;
  return C.pass;
}

// ---------------------------------------------------------------------
// MOCK MODE
// Flip this to false once your real ZSAD backend is serving /predict.
// ---------------------------------------------------------------------
const USE_MOCK = false;

const MOCK_DEFECT_CLASSES = [
  "Hole", "Stain", "Missing End", "Broken Pick", "Slubs",
  "Reed Marks", "Thick and Thin Places", "Float",
  "Bowing and Skewing", "Filling Bar",
];

function buildMockResponse(previewUrl) {
  const defectDetected = Math.random() > 0.4;
  const primaryClass = defectDetected
    ? MOCK_DEFECT_CLASSES[Math.floor(Math.random() * MOCK_DEFECT_CLASSES.length)]
    : "Good Fabric";
  const confidenceScore = defectDetected
    ? parseFloat((Math.random() * (0.98 - 0.75) + 0.75).toFixed(4))
    : parseFloat((Math.random() * (0.99 - 0.92) + 0.92).toFixed(4));
  const defectScoreVal = defectDetected
    ? parseFloat((Math.random() * (0.95 - 0.63) + 0.63).toFixed(4))
    : parseFloat((Math.random() * 0.6).toFixed(4));
  const severity = defectScoreVal > 0.78 ? 'high' : defectScoreVal > 0.62 ? 'medium' : 'low';

  return {
    defect_detected: defectDetected,
    primary_class: primaryClass,
    confidence_score: confidenceScore,
    heatmap_path: null,
    _mockHeatmapOverride: previewUrl,
    defect_score: defectScoreVal,
    severity,
  };
}

function getPrediction(file) {
  const previewUrl = URL.createObjectURL(file);

  if (USE_MOCK) {
    return new Promise((resolve) => {
      setTimeout(() => resolve(buildMockResponse(previewUrl)), 2000);
    });
  }

  const formData = new FormData();
  formData.append('file', file);
  return fetch(`${API_BASE}/predict`, { method: 'POST', body: formData })
    .then((res) => res.json());
}

function normalizeResult(data) {
  const heatmapUrl = data?.score_map_png_base64
    ? `data:image/png;base64,${data.score_map_png_base64}`
    : (data?._mockHeatmapOverride || null);

  return {
    isAnomaly: !!(data?.is_anomaly ?? false),
    className: data?.defect_name || 'Unknown',
    confidence: typeof data?.defect_confidence === 'number' ? data.defect_confidence : 0,
    heatmapUrl,
    defectScore: typeof data?.defect_score === 'number' ? data.defect_score : null,
    severity: data?.severity || null,
    imgScore: typeof data?.img_score === 'number' ? data.img_score : null,
    hScore: typeof data?.h_score === 'number' ? data.h_score : null,
  };
}

// ---------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------
export default function IndustrialInspection() {
  const [mode, setMode] = useState('single');
  const [status, setStatus] = useState('ready');
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [capturedImage, setCapturedImage] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [cameraError, setCameraError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (status === 'inspecting') {
      const start = Date.now();
      timerRef.current = setInterval(() => setElapsedMs(Date.now() - start), 47);
    } else {
      clearInterval(timerRef.current);
      setElapsedMs(0);
    }
    return () => clearInterval(timerRef.current);
  }, [status]);

  const handleModeChange = (newMode) => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setMode(newMode);
    setStatus('ready');
    setResult(null);
    setBatchResults([]);
    setCapturedImage(null);
    setCameraError(null);
    if (newMode === 'live') startCamera();
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      videoRef.current.srcObject = stream;
      setCameraError(null);
    } catch (err) {
      console.error("Camera error", err);
      setCameraError("Camera unavailable — use Static Image mode with file upload or access via localhost");
    }
  };

  const runInference = async (file) => {
    if (!file) return;
    setOriginalFile(file);
    setStatus('inspecting');
    try {
      const data = await getPrediction(file);
      setResult(normalizeResult(data));
      setStatus('results');
    } catch (error) {
      console.error("Inference Error:", error);
      setStatus('ready');
    }
  };

  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    setStatus('inspecting');
    const tempResults = [];
    for (const file of files) {
      try {
        const data = await getPrediction(file);
        tempResults.push(normalizeResult(data));
        setBatchResults([...tempResults]);
      } catch (err) { console.error("Batch error:", err); }
    }
    setStatus('results');
  };

  const handleCapture = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      setCapturedImage(URL.createObjectURL(blob));
      runInference(blob);
    }, 'image/jpeg');
  };

  return (
    <div style={{ minHeight: '100vh', background: C.paper, fontFamily: 'Inter, sans-serif', padding: '2.5rem 1rem' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        @keyframes sweep { 0% { top: -8%; opacity: 0; } 8% { opacity: 1; } 92% { opacity: 1; } 100% { top: 104%; opacity: 0; } }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
        .scan-sweep { animation: sweep 1.8s cubic-bezier(0.4,0,0.2,1) infinite; }
        .blink-dot { animation: blink 1.4s ease-in-out infinite; }
        .weave-bg {
          background-image:
            repeating-linear-gradient(0deg, rgba(244,240,230,0.06) 0px, rgba(244,240,230,0.06) 1px, transparent 1px, transparent 6px),
            repeating-linear-gradient(90deg, rgba(244,240,230,0.06) 0px, rgba(244,240,230,0.06) 1px, transparent 1px, transparent 6px);
        }
      `}</style>

      <div className="max-w-7xl mx-auto space-y-5">

        {/* INSTRUMENT HEADER STRIP */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ background: C.ink, color: C.white, borderBottom: `2px solid ${C.accent}` }}
        >
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2">
              <span
                className="blink-dot"
                style={{ width: 6, height: 6, borderRadius: '50%', background: USE_MOCK ? C.warn : C.pass, display: 'inline-block' }}
              />
              <span style={{ fontFamily: fontMono, fontSize: 10, letterSpacing: '0.2em', textTransform: 'uppercase', color: C.mutedLight }}>
                {USE_MOCK ? 'SIMULATION MODE' : 'LIVE INFERENCE'}
              </span>
            </div>
            <div className="hidden sm:block" style={{ height: 12, width: 1, background: C.line }} />
            <span className="hidden sm:inline" style={{ fontFamily: fontMono, fontSize: 10, letterSpacing: '0.2em', textTransform: 'uppercase', color: C.muted }}>
              ZSAD CORE v1.0
            </span>
          </div>
          <h1 style={{ fontFamily: fontDisplay, fontSize: 14, textTransform: 'uppercase', letterSpacing: '0.15em', margin: 0 }}>
            ZSAD <span style={{ color: C.accent }}>SCANNER</span>
          </h1>
        </div>

        {/* MODE SELECTOR */}
        <div className="flex" style={{ border: `1px solid ${C.ink}26`, background: `${C.ink}05` }}>
          <ModeTab active={mode === 'single'} onClick={() => handleModeChange('single')} icon={Upload} label="Static Image" index="01" />
          <ModeTab active={mode === 'batch'} onClick={() => handleModeChange('batch')} icon={Layers} label="Batch Logic" index="02" />
          <ModeTab active={mode === 'live'} onClick={() => handleModeChange('live')} icon={Camera} label="Live QC Feed" index="03" last />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5" style={{ minHeight: 560 }}>

          {/* MAIN VIEWFINDER */}
          <div className="lg:col-span-8 relative" style={{ background: C.ink, padding: 4 }}>
            <Bracket pos={{ top: 8, left: 8, borderTop: true, borderLeft: true }} />
            <Bracket pos={{ top: 8, right: 8, borderTop: true, borderRight: true }} />
            <Bracket pos={{ bottom: 8, left: 8, borderBottom: true, borderLeft: true }} />
            <Bracket pos={{ bottom: 8, right: 8, borderBottom: true, borderRight: true }} />

            <div className="relative flex items-center justify-center overflow-hidden" style={{ minHeight: 552 }}>
              <canvas ref={canvasRef} className="hidden" />

              {status === 'ready' && (
                <div className="w-full h-full flex flex-col items-center justify-center">
                  {mode === 'live' ? (
                    cameraError ? (
                      <div className="flex flex-col items-center text-center" style={{ padding: 32 }}>
                        <p style={{ fontFamily: fontMono, color: C.warn, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', marginBottom: 16 }}>
                          {cameraError}
                        </p>
                        <button
                          onClick={() => { setMode('single'); setCameraError(null); }}
                          style={{ background: C.accent, color: C.ink, padding: '12px 28px', fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.15em', border: 'none', cursor: 'pointer' }}
                        >
                          Switch to Static Upload
                        </button>
                      </div>
                    ) : (
                      <video ref={videoRef} autoPlay className="w-full h-full object-cover" style={{ opacity: 0.9 }} />
                    )
                  ) : (
                    <div
                      className="flex flex-col items-center cursor-pointer"
                      onClick={() => document.getElementById('file-input').click()}
                    >
                      <div style={{ padding: 36, border: `1px solid ${C.line}` }}>
                        <Crosshair size={40} strokeWidth={1.2} style={{ color: C.muted }} />
                      </div>
                      <input
                        id="file-input"
                        type="file"
                        accept="image/*"
                        capture="environment"
                        multiple={mode === 'batch'}
                        className="hidden"
                        onChange={mode === 'batch' ? handleBatchUpload : (e) => runInference(e.target.files[0])}
                      />
                      <p style={{ marginTop: 22, fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.3em' }}>
                        Load Sample · Click to Browse
                      </p>
                    </div>
                  )}
                  <div className="absolute" style={{ bottom: 32 }}>
                    <button
                      onClick={mode === 'live' ? handleCapture : () => document.getElementById('file-input').click()}
                      style={{
                        background: C.accent, color: C.ink, padding: '15px 38px', fontFamily: fontDisplay,
                        fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.2em', fontWeight: 600,
                        border: 'none', cursor: 'pointer',
                      }}
                    >
                      Begin Scan
                    </button>
                  </div>
                </div>
              )}

              {status === 'inspecting' && (
                <div className="weave-bg relative w-full h-full flex flex-col items-center justify-center">
                  <div
                    className="scan-sweep absolute left-0 right-0"
                    style={{ height: 2, background: C.accent, boxShadow: `0 0 20px 3px ${C.accent}99` }}
                  />
                  <Loader2 className="animate-spin" size={32} strokeWidth={1.5} style={{ color: C.muted, marginBottom: 18 }} />
                  <p style={{ fontFamily: fontMono, color: C.white, fontSize: 10, letterSpacing: '0.3em', textTransform: 'uppercase' }}>
                    Analyzing Surface — {(elapsedMs / 1000).toFixed(2)}s
                  </p>
                  <p style={{ fontFamily: fontMono, color: C.muted, fontSize: 9, letterSpacing: '0.2em', textTransform: 'uppercase', marginTop: 4 }}>
                    kNN · Wasserstein · Centroid
                  </p>
                </div>
              )}

              {status === 'results' && (
                <div className="w-full h-full">
                  {mode === 'batch' ? (
                    <div
                      className="grid grid-cols-2 sm:grid-cols-3 h-full overflow-y-auto"
                      style={{ gap: 1, background: C.line, padding: 1 }}
                    >
                      {batchResults.map((res, i) => (
                        <div key={i} style={{ background: C.ink }}>
                          <img src={res.heatmapUrl} className="w-full object-cover" style={{ height: 128, opacity: 0.9 }} alt="Scan result" />
                          <div className="flex justify-between items-center" style={{ padding: 10 }}>
                            <span style={{ fontFamily: fontMono, fontSize: 9, color: C.mutedLight, textTransform: 'uppercase' }} className="truncate">
                              {res.className}
                            </span>
                            <span style={{ fontFamily: fontMono, fontSize: 9, color: severityColor(res.severity) }}>
                              {(res.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="relative w-full h-full">
                      <img src={result?.heatmapUrl || capturedImage} className="w-full h-full object-contain" alt="Inspection result" />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* READOUT PANEL */}
          <div className="lg:col-span-4 flex flex-col" style={{ background: C.ink, color: C.white, padding: 24 }}>
            <div className="flex items-center justify-between" style={{ marginBottom: 22, paddingBottom: 14, borderBottom: `1px solid ${C.line}` }}>
              <h3 style={{ fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.25em', margin: 0 }}>
                Readout
              </h3>
              <span style={{ fontFamily: fontMono, fontSize: 9, color: C.muted }}>37×37 GRID</span>
            </div>

            <div className="flex-1 overflow-y-auto">
              {status === 'ready' && (
                <div
                  className="h-full flex flex-col items-center justify-center text-center"
                  style={{ border: `1px dashed ${C.line}`, padding: 24 }}
                >
                  <Crosshair size={24} strokeWidth={1} style={{ color: C.line, marginBottom: 16 }} />
                  <p style={{ fontFamily: fontMono, color: C.mutedDark, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.25em', lineHeight: 2 }}>
                    No sample loaded
                  </p>
                </div>
              )}

              {status === 'inspecting' && (
                <div>
                  {['kNN density', 'Wasserstein', 'Centroid dist.', 'Homogeneity'].map((label) => (
                    <div key={label} className="flex justify-between items-center" style={{ padding: '12px 0', borderBottom: `1px solid ${C.lineSoft}` }}>
                      <span style={{ fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase' }}>{label}</span>
                      <span style={{ fontFamily: fontMono, fontSize: 10, color: C.mutedDark }}>
                        computing<span className="blink-dot">…</span>
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {status === 'results' && mode === 'batch' && (
                <div>
                  <Readout label="Samples scanned" value={batchResults.length} />
                  <Readout
                    label="Anomalies flagged"
                    value={batchResults.filter(r => r.isAnomaly).length}
                    color={batchResults.filter(r => r.isAnomaly).length > 0 ? C.accent : C.pass}
                  />
                  <Readout
                    label="Pass rate"
                    value={`${Math.round((batchResults.filter(r => !r.isAnomaly).length / Math.max(batchResults.length, 1)) * 100)}%`}
                  />
                </div>
              )}

              {status === 'results' && mode !== 'batch' && result && (
                <div>
                  {!result.isAnomaly ? (
                    <div className="flex flex-col items-center text-center" style={{ padding: '30px 0', borderBottom: `1px solid ${C.lineSoft}` }}>
                      <CheckCircle size={28} strokeWidth={1.5} style={{ color: C.pass, marginBottom: 10 }} />
                      <p style={{ fontFamily: fontDisplay, fontSize: 14, textTransform: 'uppercase', letterSpacing: '0.05em', color: C.pass, margin: 0 }}>
                        Good Fabric
                      </p>
                      <p style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, marginTop: 4 }}>
                        No structural deviation found
                      </p>
                    </div>
                  ) : (
                    <div style={{ padding: '18px 0', borderBottom: `1px solid ${C.lineSoft}` }}>
                      <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
                        <span style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
                          Defect class
                        </span>
                        <AlertTriangle size={13} style={{ color: C.accent }} />
                      </div>
                      <p style={{ fontFamily: fontDisplay, fontSize: 18, textTransform: 'uppercase', letterSpacing: '0.03em', color: C.white, margin: 0 }}>
                        {result.className}
                      </p>
                    </div>
                  )}

                  <Readout label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} mono />
                  {result.defectScore !== null && <Readout label="Defect score" value={result.defectScore.toFixed(3)} mono />}
                  {result.severity && (
                    <div className="flex justify-between items-center" style={{ padding: '12px 0', borderBottom: `1px solid ${C.lineSoft}` }}>
                      <span style={{ fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase' }}>Severity</span>
                      <span style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', color: severityColor(result.severity) }}>
                        {result.severity}
                      </span>
                    </div>
                  )}
                  {result.hScore !== null && <Readout label="Homogeneity" value={result.hScore.toFixed(3)} mono />}
                </div>
              )}
            </div>

            <div style={{ marginTop: 22 }}>
              {status === 'results' && (
                <>
                  <button
                    onClick={async () => {
                      if (!originalFile || !result) return;
                      const formData = new FormData();
                      formData.append('file', originalFile);
                      formData.append('defect_name', result.className);
                      formData.append('defect_confidence', String(result.confidence));
                      formData.append('defect_score', String(result.defectScore ?? 0));
                      try {
                        await fetch('/save_report', { method: 'POST', body: formData });
                        alert('Report saved');
                      } catch (e) {
                        alert('Save failed');
                      }
                    }}
                    className="w-full flex items-center justify-center gap-2"
                    style={{
                      border: `1px solid ${C.line}`, color: C.white, padding: '13px 0', fontFamily: fontMono,
                      fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', background: 'transparent',
                      cursor: 'pointer', marginBottom: 8,
                    }}
                  >
                    <Save size={14} /> Save to Reports
                  </button>
                  <button
                    onClick={() => { setStatus('ready'); setResult(null); setBatchResults([]); }}
                    className="w-full flex items-center justify-center gap-2"
                    style={{
                      background: C.accent, color: C.ink, padding: '13px 0', fontFamily: fontMono,
                      fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em', fontWeight: 500,
                      border: 'none', cursor: 'pointer',
                    }}
                  >
                    Next Sample <ArrowRight size={14} />
                  </button>
                </>
              )}
              {status === 'ready' && (
                <p style={{ fontFamily: fontMono, fontSize: 9, textAlign: 'center', color: C.mutedDark, textTransform: 'uppercase', letterSpacing: '0.25em' }}>
                  Awaiting trigger
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Bracket({ pos }) {
  const style = {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#FF5A36',
    pointerEvents: 'none',
    zIndex: 10,
    ...(pos.top !== undefined ? { top: pos.top } : {}),
    ...(pos.bottom !== undefined ? { bottom: pos.bottom } : {}),
    ...(pos.left !== undefined ? { left: pos.left } : {}),
    ...(pos.right !== undefined ? { right: pos.right } : {}),
    borderTopWidth: pos.borderTop ? 2 : 0,
    borderBottomWidth: pos.borderBottom ? 2 : 0,
    borderLeftWidth: pos.borderLeft ? 2 : 0,
    borderRightWidth: pos.borderRight ? 2 : 0,
    borderStyle: 'solid',
    opacity: 0.6,
  };
  return <div style={style} />;
}

function Readout({ label, value, mono = false, color = C.white }) {
  return (
    <div className="flex justify-between items-center" style={{ padding: '12px 0', borderBottom: `1px solid ${C.lineSoft}` }}>
      <span style={{ fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
        {label}
      </span>
      <span style={{ fontFamily: mono ? fontMono : fontDisplay, fontSize: 14, color }}>
        {value}
      </span>
    </div>
  );
}

function ModeTab({ active, onClick, icon: Icon, label, index, last = false }) {
  return (
    <button
      onClick={onClick}
      className="flex-1 flex items-center gap-3"
      style={{
        padding: '15px 18px',
        borderTop: 'none',
        borderLeft: 'none',
        borderBottom: 'none',
        borderRight: last ? 'none' : `1px solid ${C.ink}26`,
        background: active ? C.ink : 'transparent',
        color: active ? C.white : `${C.ink}80`,
        cursor: 'pointer',
      }}
    >
      <span style={{ fontFamily: fontMono, fontSize: 10, color: active ? C.accent : `${C.ink}4D` }}>{index}</span>
      <Icon size={16} strokeWidth={1.5} />
      <span style={{ fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.2em' }}>{label}</span>
    </button>
  );
}