import React from 'react';
import { Sidebar } from './components/Sidebar';
import { GraphCanvas } from './components/GraphCanvas';
import { DetailsPanel } from './components/DetailsPanel';
import './App.css';

function App() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 relative h-full">
        <GraphCanvas />
        <DetailsPanel />
      </div>
    </div>
  );
}

export default App;
