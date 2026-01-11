import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import { Terminal, Braces } from 'lucide-react';
import { cn } from '../../utils';

const FunctionNode = ({ data, selected }: NodeProps) => {
  return (
    <div className={cn(
      "min-w-[200px] bg-[#18181b] rounded-xl border-2 transition-all duration-300 shadow-xl overflow-hidden group",
      selected 
        ? "border-purple-500 ring-4 ring-purple-500/10 shadow-purple-500/20" 
        : "border-white/5 hover:border-purple-500/50"
    )}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-purple-500/10 to-transparent border-b border-white/5">
        <div className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400">
          <Terminal className="w-4 h-4" />
        </div>
        <div>
          <span className="text-[10px] font-bold text-purple-500/80 tracking-wider uppercase block">Function</span>
          <span className="font-semibold text-sm text-foreground">{data.label}</span>
        </div>
      </div>
      
      {/* Content */}
      <div className="p-4 bg-black/20">
         {/* Optional: Add args or return type here if available in data */}
         <div className="flex items-center gap-2">
            <Braces className="w-3 h-3 text-muted-foreground/50" />
            <span className="text-[10px] text-muted-foreground/70 font-mono">def {data.label}(...)</span>
         </div>
      </div>

      {/* Handles */}
      <Handle type="target" position={Position.Top} className="!bg-purple-500 !w-3 !h-3 !border-4 !border-[#18181b]" />
      <Handle type="source" position={Position.Bottom} className="!bg-purple-500 !w-3 !h-3 !border-4 !border-[#18181b]" />
    </div>
  );
};

export default memo(FunctionNode);
