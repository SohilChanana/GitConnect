import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import { FileCode, FileType } from 'lucide-react';
import { cn } from '../../utils';

const FileNode = ({ data, selected }: NodeProps) => {
  return (
    <div className={cn(
      "min-w-[220px] bg-[#18181b] rounded-xl border-2 transition-all duration-300 shadow-xl overflow-hidden group",
      selected 
        ? "border-amber-500 ring-4 ring-amber-500/10 shadow-amber-500/20" 
        : "border-white/5 hover:border-amber-500/50"
    )}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-amber-500/10 to-transparent border-b border-white/5">
        <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400">
          <FileCode className="w-4 h-4" />
        </div>
        <div>
          <span className="text-[10px] font-bold text-amber-500/80 tracking-wider uppercase block">File</span>
          <span className="font-semibold text-sm text-foreground truncate max-w-[140px]" title={data.label}>{data.label}</span>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-4 bg-black/20">
        <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono bg-black/40 px-2 py-1.5 rounded-md border border-white/5">
          <FileType className="w-3 h-3 opacity-50" />
          <span className="truncate">{data.subLabel || data.label}</span>
        </div>
      </div>

      {/* Handles */}
      <Handle type="target" position={Position.Top} className="!bg-amber-500 !w-3 !h-3 !border-4 !border-[#18181b]" />
      <Handle type="source" position={Position.Bottom} className="!bg-amber-500 !w-3 !h-3 !border-4 !border-[#18181b]" />
    </div>
  );
};

export default memo(FileNode);
