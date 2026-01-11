import React, { useState, useRef, useEffect } from 'react';
import { Send, Database, Settings, Terminal, Search, Link as LinkIcon, X, Github, ChevronRight } from 'lucide-react';
import { useStore } from '../store';
import { cn } from '../utils';
import { ConfirmationModal } from './ConfirmationModal';

export function Sidebar() {
  const { chatHistory, sendUserMessage, isLoading, isIngesting, ingestRepo, currentRepoUrl, init, deleteRepo } = useStore();
  const [inputValue, setInputValue] = useState('');
  const [repoInput, setRepoInput] = useState('');
  const [showRepoInput, setShowRepoInput] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  
  // Resizable Sidebar Logic
  const [width, setWidth] = useState(320);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  // Initialize status on mount
  useEffect(() => {
    init();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = e.clientX;
      if (newWidth > 240 && newWidth < 600) {
        setWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = 'default';
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'default';
    };
  }, [isResizing]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const message = inputValue;
    setInputValue(''); // Clear immediately
    
    await sendUserMessage(message);
  };

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repoInput.trim() || isIngesting) return;
    
    setShowRepoInput(false);
    await ingestRepo(repoInput);
    setRepoInput('');
  };

  return (
    <div 
      ref={sidebarRef}
      style={{ width: `${width}px` }}
      className="flex flex-col h-full bg-black/40 backdrop-blur-xl border-r border-white/5 relative z-20 transition-all duration-75 ease-linear group"
    >
      {/* Header */}
      <div className="p-4 border-b border-white/5 flex items-center justify-between shrink-0 bg-white/5">
        <div className="flex items-center gap-3">
          <div className="p-1.5 bg-primary/10 rounded-lg">
             <Terminal className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-medium text-sm text-foreground tracking-tight">GitConnect</h2>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Repository Explorer</p>
          </div>
        </div>
        <div className="flex gap-2 text-muted-foreground">
          <button className="p-1.5 hover:bg-white/5 rounded-md transition-colors text-muted-foreground hover:text-foreground">
             <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Repo Status & Ingestion */}
      <div className="p-4 flex flex-col gap-3 shrink-0">
        <div className="flex items-center justify-between">
           <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Current Context</span>
           {currentRepoUrl && (
             <span className="flex items-center gap-1.5 text-[10px] text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full border border-emerald-400/20">
               <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" />
               Connected
             </span>
           )}
        </div>
        
        <div className="group/repo relative">
           {!currentRepoUrl ? (
             !showRepoInput ? (
               <button 
                  onClick={() => setShowRepoInput(true)}
                  className="w-full flex items-center gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 border border-dashed border-white/10 hover:border-white/20 transition-all text-sm text-muted-foreground hover:text-foreground group-hover/repo:scale-[1.02]"
               >
                 <div className="p-1.5 bg-background rounded-lg text-primary opacity-80">
                   <Database className="w-4 h-4" />
                 </div>
                 <span>Select Repository</span>
                 <ChevronRight className="w-4 h-4 ml-auto opacity-50" />
               </button>
             ) : (
               <form onSubmit={handleIngest} className="bg-card p-3 rounded-xl border border-primary/20 shadow-lg animate-in fade-in slide-in-from-top-2 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent opacity-50" />
                  <div className="flex justify-between items-center mb-3">
                     <span className="text-[10px] uppercase font-medium text-primary flex items-center gap-1.5">
                       <Github className="w-3 h-3" />
                       GitHub URL
                     </span>
                     <button type="button" onClick={() => setShowRepoInput(false)} className="text-muted-foreground hover:text-foreground p-1 hover:bg-white/10 rounded"><X className="w-3 h-3" /></button>
                  </div>
                  <input 
                     autoFocus
                     type="text" 
                     value={repoInput}
                     onChange={(e) => setRepoInput(e.target.value)}
                     placeholder="https://github.com/owner/repo"
                     className="w-full bg-background border border-white/10 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all placeholder:text-muted-foreground/50 mb-3"
                  />
                  <button 
                     type="submit" 
                     disabled={isIngesting}
                     className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-medium text-xs py-2 rounded-lg transition-all flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
                  >
                     {isIngesting ? <span className="animate-pulse">Cloning...</span> : 'Connect'}
                  </button>
               </form>
             )
           ) : (
              <div className="w-full flex items-center gap-3 p-3 rounded-xl bg-card border border-white/5 shadow-sm">
                 <div className="p-2 bg-primary/10 rounded-lg text-primary">
                    <Github className="w-4 h-4" />
                 </div>
                 <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-foreground truncate" title={currentRepoUrl}>
                      {currentRepoUrl.replace('https://github.com/', '')}
                    </div>
                    <div className="text-[10px] text-muted-foreground">main branch</div>
                 </div>
                 <button 
                   onClick={() => setShowDeleteConfirm(true)}
                   className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
                 >
                   <X className="w-4 h-4" />
                 </button>
              </div>
           )}
        </div>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-4 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent min-h-0">
        {chatHistory.length === 0 && !isIngesting && (
          <div className="flex flex-col items-center justify-center h-48 text-muted-foreground/40 text-sm">
            <Search className="w-10 h-10 mb-3 opacity-20" />
            <p>Ask me anything about the code.</p>
          </div>
        )}

        {isIngesting && (
           <div className="flex flex-col items-center justify-center py-8 space-y-3">
              <div className="relative w-10 h-10">
                 <div className="absolute inset-0 border-2 border-primary/20 rounded-full"></div>
                 <div className="absolute inset-0 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              </div>
              <p className="text-xs font-medium text-primary animate-pulse">Analyzing Structure...</p>
           </div>
        )}
        
        {chatHistory.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              "p-3.5 rounded-2xl text-sm max-w-[95%] shadow-sm",
              msg.role === 'user' 
                ? "bg-primary text-primary-foreground ml-auto rounded-tr-sm" 
                : "bg-white/5 text-foreground border border-white/5 rounded-tl-sm backdrop-blur-sm"
            )}
          >
            <p className="leading-relaxed whitespace-pre-wrap text-[13px]">{msg.content}</p>
            <span className="text-[10px] opacity-50 mt-1.5 block text-right">
              {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        ))}
        
        {isLoading && !isIngesting && (
          <div className="bg-white/5 border border-white/5 p-4 rounded-2xl rounded-tl-sm w-16 flex items-center justify-center gap-1">
             <span className="w-1 h-1 bg-foreground/50 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
             <span className="w-1 h-1 bg-foreground/50 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
             <span className="w-1 h-1 bg-foreground/50 rounded-full animate-bounce"></span>
          </div>
        )}
        <div className="h-px bg-transparent" />
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-white/5 bg-black/20 backdrop-blur-xl shrink-0">
        <form onSubmit={handleSendMessage} className="relative group/input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading || isIngesting || !currentRepoUrl}
            className="w-full bg-white/5 border border-white/5 rounded-xl py-3.5 pl-4 pr-12 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:bg-background focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all shadow-inner disabled:opacity-50"
          />
          <button 
            type="submit"
            disabled={!inputValue.trim() || isLoading || isIngesting}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary hover:text-primary-foreground disabled:opacity-0 disabled:pointer-events-none transition-all duration-200"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </form>
      </div>

      {/* Resize Handle */}
      <div 
        onMouseDown={() => setIsResizing(true)}
        className="absolute top-0 right-[-1px] w-1 h-full cursor-col-resize hover:bg-primary/50 transition-colors z-50 flex items-center justify-center group-hover:bg-white/10 active:bg-primary"
      >
        <div className="w-[1px] h-8 bg-white/20 rounded-full" />
      </div>

      <ConfirmationModal 
        isOpen={showDeleteConfirm}
        title="Disconnect Repository?"
        message="This will remove the current repository data from the graph. You will need to re-ingest it to analyze it again."
        confirmLabel="Disconnect"
        isDestructive={true}
        onConfirm={() => {
            deleteRepo();
            setShowDeleteConfirm(false);
        }}
        onCancel={() => setShowDeleteConfirm(false)}
      />
    </div>
  );
}
