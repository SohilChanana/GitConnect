import { X, FileText, Activity, Code } from 'lucide-react';
import { useStore } from '../store';

export function DetailsPanel() {
  const { selectedNodeId, setSelectedNodeId, nodes } = useStore();
  
  if (!selectedNodeId) return null;

  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  if (!selectedNode) return null;

  // Determine color based on node type
  const getTypeColor = (type?: string) => {
    switch(type) {
      case 'class': return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20';
      case 'function': return 'text-purple-400 bg-purple-500/10 border-purple-500/20';
      case 'file': return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
      default: return 'text-primary bg-primary/10 border-primary/20';
    }
  };

  const colorClass = getTypeColor(selectedNode.type);

  return (
    <div className="absolute top-6 right-6 w-96 bg-black/60 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-2xl z-40 transition-all animate-in slide-in-from-right-10 fade-in duration-300 flex flex-col max-h-[calc(100vh-3rem)]">
      {/* Header */}
      <div className="flex items-center justify-between p-5 border-b border-white/5 shrink-0">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-xl ${colorClass} border`}>
            <Activity className="w-4 h-4" />
          </div>
          <div>
             <h3 className="font-semibold text-foreground text-sm">Entity Details</h3>
             <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{selectedNode.type || 'Unknown Type'}</p>
          </div>
        </div>
        <button 
          onClick={() => setSelectedNodeId(null)}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors text-muted-foreground hover:text-foreground"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content - Scrollable */}
      <div className="p-5 space-y-6 overflow-y-auto custom-scrollbar">
        
        {/* Main Info */}
        <div className="space-y-1">
          <label className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold px-1">Name</label>
          <div className="font-medium text-lg text-foreground px-1">{selectedNode.data.label}</div>
        </div>

        {selectedNode.data.subLabel && (
           <div className="space-y-1">
            <label className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold px-1">Location</label>
            <div className="flex items-center gap-2 p-3 bg-white/5 rounded-xl border border-white/5 font-mono text-xs text-muted-foreground hover:text-foreground transition-colors group">
              <FileText className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
              {selectedNode.data.subLabel}
            </div>
           </div>
        )}

        <div className="space-y-1">
          <label className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold px-1">Properties</label>
          <div className="grid grid-cols-2 gap-3">
             <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <div className="text-[10px] text-muted-foreground mb-1">Type</div>
                <div className="font-mono text-xs">{selectedNode.type}</div>
             </div>
             <div className="p-3 bg-white/5 rounded-xl border border-white/5">
                <div className="text-[10px] text-muted-foreground mb-1">ID</div>
                <div className="font-mono text-xs truncate" title={selectedNode.id}>{selectedNode.id}</div>
             </div>
          </div>
        </div>
        
        {/* Code Preview */}
        <div className="space-y-2">
           <div className="flex items-center justify-between px-1">
              <label className="text-[10px] uppercase tracking-wider text-muted-foreground/60 font-semibold">Source Preview</label>
              <Code className="w-3 h-3 text-muted-foreground/40" />
           </div>
           <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/20 pointer-events-none rounded-xl" />
              <pre className="p-4 bg-black/80 rounded-xl text-[10px] text-emerald-400/90 font-mono overflow-auto max-h-80 border border-white/10 whitespace-pre-wrap shadow-inner custom-scrollbar">
                {selectedNode.data.content || '# No source code available'}
              </pre>
           </div>
        </div>
      </div>
    </div>
  );
}
