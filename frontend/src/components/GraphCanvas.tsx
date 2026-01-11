import { useMemo } from 'react';
import ReactFlow, { 
  Background, 
  Controls, 
  MiniMap,
  BackgroundVariant,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useStore } from '../store';

import ClassNode from './nodes/ClassNode';
import FunctionNode from './nodes/FunctionNode';
import FileNode from './nodes/FileNode';
import CustomEdge from './edges/CustomEdge';

export function GraphCanvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect } = useStore();

  const nodeTypes = useMemo(() => ({
    class: ClassNode,
    function: FunctionNode,
    file: FileNode,
  }), []);

  const edgeTypes = useMemo(() => ({
    custom: CustomEdge,
  }), []);

  return (
    <div className="flex-1 h-full w-full bg-background relative overflow-hidden" style={{ width: '100%', height: '100%' }}>
      {/* Decorative Grid Overlay to match 'tech-noir' feel */}
      <div className="absolute inset-0 pointer-events-none bg-[url('/grid-pattern.svg')] opacity-[0.03] z-10" />

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(_, node) => useStore.getState().setSelectedNodeId(node.id)}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        className="bg-black/20"
      >
        <Background 
          variant={BackgroundVariant.Dots} 
          gap={20} 
          size={1} 
          color="#333" 
          className="opacity-50"
        />
        <Controls className="bg-card border border-border text-foreground" />
        {!useStore().isSummaryOpen && (
        <MiniMap 
          nodeStrokeColor="#333" 
          nodeColor="#1a1a1a"
          maskColor="rgba(0, 0, 0, 0.6)"
          className="bg-card border border-border rounded-lg overflow-hidden"
        />
        )}
        
        {/* Floating Panel for stats or legend (Moved to Top Left) */}
        <Panel position="top-left" className="bg-card/50 backdrop-blur-md p-2 rounded-lg border border-white/5 text-xs text-muted-foreground ml-4 mt-4">
          <div>Nodes: {nodes.length}</div>
          <div>Edges: {edges.length}</div>
        </Panel>
      </ReactFlow>
    </div>
  );
}
