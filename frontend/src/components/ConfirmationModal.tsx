import React from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, X } from 'lucide-react';

interface ConfirmationModalProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  isDestructive?: boolean;
}

export function ConfirmationModal({
  isOpen,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
  isDestructive = false
}: ConfirmationModalProps) {
  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="relative w-full max-w-sm bg-card border border-border rounded-xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
      >
        <div className="p-6">
          <div className="flex items-start gap-4">
            <div className={`p-2 rounded-full shrink-0 ${isDestructive ? 'bg-red-500/10 text-red-500' : 'bg-primary/10 text-primary'}`}>
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div className="flex-1 space-y-2">
              <h3 className="font-semibold text-lg leading-none tracking-tight text-foreground">
                {title}
              </h3>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {message}
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 p-6 pt-0 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium transition-colors rounded-lg hover:bg-secondary text-foreground/80 hover:text-foreground"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-sm font-medium text-white transition-colors rounded-lg ${
              isDestructive 
                ? 'bg-red-500 hover:bg-red-600' 
                : 'bg-primary hover:bg-primary/90'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
        
        <button 
           onClick={onCancel}
           className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
        >
           <X className="w-4 h-4" />
        </button>
      </div>
    </div>,
    document.body
  );
}
