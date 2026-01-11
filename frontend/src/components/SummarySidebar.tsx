import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2, Sparkles, AlertCircle, FileJson, CheckCircle2, ChevronDown } from 'lucide-react';
import { api } from '../api';
import { useStore } from '../store';
import { cn } from '../utils';

export function SummarySidebar() {
  const { isSummaryOpen, setSummaryOpen } = useStore();
  const [repoUrl, setRepoUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<any>(null); // Weak type to allow accessing extra fields like citations
  const [error, setError] = useState<string | null>(null);
  const [showJson, setShowJson] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoUrl.trim()) return;

    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const result = await api.summarize({ repo_url: repoUrl });
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate summary');
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400';
    if (score >= 50) return 'text-amber-400';
    return 'text-red-400';
  };

  // Map backend report to user's "summaryView" structure
  const summaryView = report ? {
    verdict: report.viability_analysis?.match(/verdict:\s*(\w+)/i)?.[1] || 'Analyzed',
    score: report.viability_score,
    summaryParagraph: report.summary_paragraph || report.summary,
    reasons: report.summary_bullets || report.key_features,
    citations: report.citations ? (Array.isArray(report.citations) ? report.citations : Object.entries(report.citations).map(([k, v]) => `${k}: ${v}`)) : []
  } : null;

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setSummaryOpen(!isSummaryOpen)}
        className={cn(
          "absolute right-0 top-6 z-30 p-2 bg-black/60 backdrop-blur-md border hover:bg-white/10 transition-all duration-300 rounded-l-xl border-y border-l border-white/10",
          isSummaryOpen ? "translate-x-0" : "translate-x-0" // Always visible
        )}
      >
        {isSummaryOpen ? <ChevronRight className="w-4 h-4 text-primary" /> : <ChevronLeft className="w-4 h-4 text-primary" />}
      </button>

      {/* Sidebar Panel */}
      <div
        className={cn(
            "absolute right-0 top-0 h-full w-[400px] bg-black/80 backdrop-blur-xl border-l border-white/5 shadow-2xl transition-transform duration-300 z-20 flex flex-col",
            isSummaryOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="p-4 border-b border-white/5 flex items-center justify-between shrink-0 bg-white/5">
            <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-purple-400" />
                <h2 className="font-medium text-sm tracking-tight text-foreground">Viability Assessment</h2>
            </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin scrollbar-thumb-white/10">
            
            {/* Input Form */}
            <form onSubmit={handleSubmit} className="space-y-3">
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Target Repository</label>
                <div className="flex gap-2">
                    <input 
                        type="text" 
                        value={repoUrl}
                        onChange={(e) => setRepoUrl(e.target.value)}
                        placeholder="https://github.com/owner/repo"
                        className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-purple-500/50 transition-colors text-foreground placeholder:text-muted-foreground/50"
                    />
                    <button 
                        type="submit" 
                        disabled={loading || !repoUrl.trim()}
                        className="bg-purple-600 hover:bg-purple-500 text-white px-3 py-2 rounded-lg disabled:opacity-50 transition-colors"
                    >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                    </button>
                </div>
            </form>

            {/* Error Message */}
            {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <p>{error}</p>
                </div>
            )}

            {/* Report */}
            {report ? (
                <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
                    
                    {/* Verdict & Score */}
                    <div className="p-4 rounded-xl bg-white/5 border border-white/10 flex justify-between items-center">
                        <div className="flex flex-col">
                            <span className="text-xs font-medium text-muted-foreground uppercase">Verdict</span>
                            <span className={cn("text-lg font-bold capitalize", getVerdictColor(summaryView?.score || 0))}>
                                {String(summaryView?.verdict ?? "—")}
                            </span>
                        </div>
                        <div className="flex flex-col items-end">
                            <span className="text-xs font-medium text-muted-foreground uppercase">Score</span>
                            <span className="text-xl font-mono font-bold text-foreground">{String(summaryView?.score ?? "—")}</span>
                        </div>
                    </div>

                    {/* Summary */}
                    <div className="space-y-2">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Summary</div>
                        <div className="p-3 rounded-xl bg-white/5 border border-white/5 text-sm leading-relaxed text-gray-300">
                            {String(summaryView?.summaryParagraph ?? "—")}
                        </div>
                    </div>

                    {/* Reasons */}
                    {Array.isArray(summaryView?.reasons) && summaryView!.reasons.length > 0 && (
                        <div className="space-y-2">
                            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <CheckCircle2 className="w-3 h-3" /> Reasons
                            </div>
                            <ul className="space-y-2">
                                {summaryView!.reasons.slice(0, 12).map((r: any, idx: number) => (
                                    <li key={idx} className="flex gap-2 text-xs text-gray-400 p-2 rounded-lg bg-white/[0.02]">
                                        <div className="w-1.5 h-1.5 rounded-full bg-purple-500/50 mt-1.5 shrink-0" />
                                        {String(r)}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Citations */}
                    {Array.isArray(summaryView?.citations) && summaryView!.citations.length > 0 && (
                        <div className="space-y-2">
                            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <FileJson className="w-3 h-3" /> Citations
                            </div>
                            <ul className="space-y-2">
                                {summaryView!.citations.slice(0, 12).map((c: any, idx: number) => (
                                    <li key={idx} className="text-xs text-gray-500 italic p-2 rounded-lg bg-white/[0.02] break-words">
                                        {String(c)}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Raw JSON */}
                    <div className="pt-4 border-t border-white/5">
                        <button 
                            onClick={() => setShowJson(!showJson)}
                            className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
                        >
                            <ChevronDown className={cn("w-3 h-3 transition-transform", showJson && "rotate-180")} />
                            Raw JSON Response
                        </button>
                        {showJson && (
                            <pre className="mt-3 p-3 rounded-xl bg-black/50 border border-white/10 text-[10px] text-gray-500 overflow-x-auto font-mono">
                                {JSON.stringify(report, null, 2)}
                            </pre>
                        )}
                    </div>

                </div>
            ) : (
                !loading && (
                    <div className="flex flex-col items-center justify-center h-48 text-muted-foreground/40 text-sm">
                        <Sparkles className="w-10 h-10 mb-3 opacity-20" />
                        <p>Load a repo to generate summary.</p>
                    </div>
                )
            )}
        </div>
      </div>
    </>
  );
}
