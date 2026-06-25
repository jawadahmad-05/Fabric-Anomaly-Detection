import React, { useState, useEffect } from 'react';
import { FileText, FileSpreadsheet, Image as ImageIcon, Loader2, RefreshCw, Filter } from 'lucide-react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import * as XLSX from 'xlsx';

// ---------------------------------------------------------------------
// Same color tokens as the rest of the app.
// ---------------------------------------------------------------------
const C = {
  ink: '#0B0E14',
  paper: '#F4F0E6',
  accent: '#FF5A36',
  pass: '#3DDC84',
  muted: '#5C6478',
  white: '#F4F0E6',
};

const fontDisplay = "'Oswald', sans-serif";
const fontMono = "'JetBrains Mono', monospace";

const CLEAR_LABELS = ['Clear', 'Good Fabric', 'No Defect', 'None'];

// ---------------------------------------------------------------------
// SINGLE PLACE TO EDIT once the backend's final schema is confirmed.
// The backend is still undecided (zero-shot model, not built yet), and
// the data currently being served uses an older schema (Defect_Type,
// Image_Url) than the Category/VisualEvidence shape a zero-shot backend
// would likely use. This function checks both, so the table renders
// correctly with whichever one is actually live, and won't need any
// other changes in this file once the real schema is locked in — only
// this function would need a new fallback key added.
// ---------------------------------------------------------------------
function normalizeRow(item) {
  const category = item.Category ?? item.Defect_Type ?? item.primary_class ?? 'Unknown';
  const imageUrl = item.VisualEvidence ?? item.Image_Url ?? item.HeatmapPath ?? item.heatmap_path ?? null;
  const confidence = item.Confidence ?? item.confidence_score ?? item.confidence ?? null;
  const timestamp = item.Timestamp ?? item.timestamp ?? null;
  const id = item.ID ?? item.id ?? null;
  const isClear = CLEAR_LABELS.includes(category);

  return { category, imageUrl, confidence, timestamp, id, isClear, raw: item };
}

export default function Reports() {
  const [allData, setAllData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [filterRange, setFilterRange] = useState('all');
  const [loading, setLoading] = useState(true);

  const fetchReports = async () => {
    setLoading(true);
    try {
      const response = await fetch('/all_reports', { cache: 'no-store' });
      const json = await response.json();
      const rows = (Array.isArray(json) ? json : []).map(normalizeRow);
      setAllData(rows);
      setFilteredData(rows);
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchReports(); }, []);

  const applyFilter = (range) => {
    setFilterRange(range);
    const now = new Date();
    let filtered = allData;

    if (range !== 'all') {
      const days = range === '1d' ? 1 : range === '7d' ? 7 : 30;
      const cutoff = new Date(now.getTime() - (days * 24 * 60 * 60 * 1000));
      filtered = allData.filter((item) => item.timestamp && new Date(item.timestamp) >= cutoff);
    }
    setFilteredData(filtered);
  };

  const exportExcel = () => {
    const ws = XLSX.utils.json_to_sheet(
      filteredData.map((r) => ({
        ID: r.id ?? 'N/A',
        Timestamp: r.timestamp ?? 'N/A',
        Category: r.category,
        Confidence: r.confidence ?? 'N/A',
      }))
    );
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Reports');
    XLSX.writeFile(wb, `Textile_Report_${filterRange}.xlsx`);
  };

  const exportPDF = () => {
    try {
      if (filteredData.length === 0) return alert('No data to export');

      const doc = new jsPDF();
      doc.setFontSize(18);
      doc.text('Industrial Fabric Inspection Report', 14, 22);

      doc.setFontSize(10);
      doc.setTextColor(100);
      doc.text(`Filter: ${filterRange.toUpperCase()} | Generated: ${new Date().toLocaleString()}`, 14, 30);

      const tableColumn = ['ID', 'Timestamp', 'Category', 'Confidence'];
      const tableRows = filteredData.map((r) => [
        r.id ?? 'N/A',
        r.timestamp ?? 'N/A',
        r.category,
        r.confidence ? `${(parseFloat(r.confidence) * 100).toFixed(1)}%` : 'N/A',
      ]);

      autoTable(doc, {
        startY: 35,
        head: [tableColumn],
        body: tableRows,
        theme: 'grid',
        headStyles: { fillColor: [11, 14, 20] },
        styles: { fontSize: 8 },
      });

      doc.save(`Textile_Report_${Date.now()}.pdf`);
    } catch (err) {
      console.error('PDF Export Error:', err);
      alert('PDF generation failed. Check the browser console for details.');
    }
  };

  return (
    <div className="space-y-6">
      {/* HEADER */}
      <div
        className="flex flex-col md:flex-row md:items-center justify-between"
        style={{ background: C.ink, color: C.white, padding: '28px 32px', borderBottom: `2px solid ${C.accent}` }}
      >
        <div>
          <h1 style={{ fontFamily: fontDisplay, fontSize: 22, textTransform: 'uppercase', letterSpacing: '0.02em', margin: 0, fontWeight: 600 }}>
            Inspection History
          </h1>
          <p style={{ fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.2em', marginTop: 6 }}>
            Records found: {filteredData.length}
          </p>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <button
            onClick={exportPDF}
            className="flex items-center gap-2"
            style={{ background: C.accent, color: C.ink, padding: '10px 18px', fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, border: 'none', cursor: 'pointer' }}
          >
            <FileText size={14} /> PDF
          </button>
          <button
            onClick={exportExcel}
            className="flex items-center gap-2"
            style={{ background: C.pass, color: C.ink, padding: '10px 18px', fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600, border: 'none', cursor: 'pointer' }}
          >
            <FileSpreadsheet size={14} /> Excel
          </button>
        </div>
      </div>

      {/* FILTER BAR */}
      <div
        className="flex items-center justify-between"
        style={{ background: '#fff', border: `1px solid ${C.ink}1A`, padding: 14 }}
      >
        <div className="flex items-center gap-4">
          <Filter size={16} strokeWidth={1.6} style={{ color: C.muted, marginLeft: 8 }} />
          <div className="flex" style={{ background: C.paper, padding: 4 }}>
            {['1d', '7d', '30d', 'all'].map((r) => (
              <button
                key={r}
                onClick={() => applyFilter(r)}
                style={{
                  padding: '8px 16px', fontFamily: fontMono, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em',
                  background: filterRange === r ? C.ink : 'transparent',
                  color: filterRange === r ? C.white : C.muted,
                  border: 'none', cursor: 'pointer',
                }}
              >
                {r === 'all' ? 'All time' : r.replace('d', ' day')}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={fetchReports}
          style={{ padding: 8, background: 'transparent', border: 'none', color: C.muted, cursor: 'pointer' }}
        >
          <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* TABLE */}
      <div style={{ background: '#fff', border: `1px solid ${C.ink}1A`, overflow: 'hidden' }}>
        <table className="w-full text-left" style={{ borderCollapse: 'collapse' }}>
          <thead style={{ background: C.paper, borderBottom: `1px solid ${C.ink}1A` }}>
            <tr>
              {['Visual evidence', 'Timestamp', 'Category', 'Confidence'].map((h) => (
                <th
                  key={h}
                  style={{ fontFamily: fontMono, fontSize: 9, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.15em', padding: 16, fontWeight: 400 }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="4" style={{ padding: 80, textAlign: 'center' }}>
                  <Loader2 className="animate-spin" style={{ color: C.muted, margin: '0 auto' }} />
                </td>
              </tr>
            ) : filteredData.length === 0 ? (
              <tr>
                <td colSpan="4" style={{ padding: 80, textAlign: 'center', fontFamily: fontMono, fontSize: 10, color: C.muted, textTransform: 'uppercase', letterSpacing: '0.2em' }}>
                  No data captured
                </td>
              </tr>
            ) : filteredData.map((r, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${C.ink}0D` }}>
                <td style={{ padding: 16 }}>
                  <div style={{ width: 80, height: 56, background: C.ink, overflow: 'hidden', border: `1px solid ${C.ink}1A` }}>
                    {r.imageUrl ? (
                      <img
                        src={`${r.imageUrl}?t=${Date.now()}`}
                        className="w-full h-full object-cover"
                        alt="defect evidence"
                        onError={(e) => { e.target.style.display = 'none'; }}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full" style={{ color: C.muted }}>
                        <ImageIcon size={16} strokeWidth={1.6} />
                      </div>
                    )}
                  </div>
                </td>
                <td style={{ padding: 16, fontFamily: fontMono, fontSize: 11, color: C.muted }}>
                  {r.timestamp ?? '—'}
                </td>
                <td style={{ padding: 16 }}>
                  <span
                    style={{
                      padding: '5px 12px', fontFamily: fontMono, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.1em',
                      background: r.isClear ? `${C.pass}1A` : `${C.accent}1A`,
                      color: r.isClear ? C.pass : C.accent,
                    }}
                  >
                    {r.category}
                  </span>
                </td>
                <td style={{ padding: 16, fontFamily: fontMono, fontSize: 11, color: C.ink, fontWeight: 500 }}>
                  {r.confidence ? `${(parseFloat(r.confidence) * 100).toFixed(0)}%` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}