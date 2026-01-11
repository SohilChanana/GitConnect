import { create } from 'zustand';
import { 
  type Connection, 
  type Edge, 
  type EdgeChange, 
  type Node, 
  type NodeChange, 
  type OnNodesChange, 
  type OnEdgesChange, 
  type OnConnect,
  applyNodeChanges, 
  applyEdgeChanges, 
  addEdge,
} from 'reactflow';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

interface AppState {
  // Graph Data
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  
  // Interaction State
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;
  highlightedNodeIds: string[];
  setHighlightedNodeIds: (ids: string[]) => void;

  // Chat State
  chatHistory: ChatMessage[];
  isLoading: boolean;
  isIngesting: boolean;
  currentRepoUrl: string | null;
  sendUserMessage: (content: string) => Promise<void>;
  ingestRepo: (url: string) => Promise<void>;
  deleteRepo: () => Promise<void>;
  init: () => Promise<void>;
  
  addChatMessage: (message: ChatMessage) => void;
  clearChat: () => void;
}

import { api } from './api';

export const useStore = create<AppState>((set, get) => ({
  nodes: [],
  edges: [],
  onNodesChange: (changes: NodeChange[]) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },
  onEdgesChange: (changes: EdgeChange[]) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },
  onConnect: (connection: Connection) => {
    set({
      edges: addEdge(connection, get().edges),
    });
  },
  setNodes: (nodes: Node[]) => set({ nodes }),
  setEdges: (edges: Edge[]) => set({ edges }),

  selectedNodeId: null,
  setSelectedNodeId: (id: string | null) => set({ selectedNodeId: id }),
  highlightedNodeIds: [],
  setHighlightedNodeIds: (ids: string[]) => set({ highlightedNodeIds: ids }),

  chatHistory: [],
  isLoading: false,
  isIngesting: false,
  currentRepoUrl: null,

  sendUserMessage: async (content: string) => {
    const { chatHistory } = get();
    
    // Add user message
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };
    
    set({ chatHistory: [...chatHistory, userMsg], isLoading: true });

    try {
      // Call Backend API
      const response = await api.analyze({
         query: content,
         use_vector_search: true,
         max_depth: 3
      });

      const newNodes: Node[] = [];
      const entityMap = new Map<string, string>(); // name -> id

      // Create Nodes from Relevant Entities
      response.relevant_entities.forEach((entity, index) => {
         const id = `node-${index}`;
         entityMap.set(entity.name, id);
         
         const typeMap: Record<string, string> = {
            'Class': 'class',
            'Function': 'function',
            'File': 'file'
         };

         newNodes.push({
            id,
            type: typeMap[entity.type] || 'function', // default to function if unknown
            position: { x: 200 + (index * 150), y: 100 + (index % 3) * 150 }, // Simple layout
            data: { 
               label: entity.name, 
               subLabel: entity.file_path !== 'N/A' ? entity.file_path : undefined,
               content: entity.content
            }
         });
      });

      // Create Edges
      const newEdges: Edge[] = [];
      if (response.edges) {
          response.edges.forEach((edge: any, idx: number) => {
              const sourceId = entityMap.get(edge.source);
              const targetId = entityMap.get(edge.target);
              
              if (sourceId && targetId) {
                  newEdges.push({
                      id: `edge-${idx}`,
                      source: sourceId,
                      target: targetId,
                      type: 'custom', // Use our CustomEdge
                      animated: true,
                      label: edge.type !== 'CONTAINS' ? edge.type : undefined
                  });
              }
          });
      }

      // Fallback: If no edges returned (yet), try to infer from dependencies if possible?
      // No, let's rely on backend sending 'edges'.

      // Update Graph if valid data found
      if (newNodes.length > 0) {
          set({ nodes: newNodes, edges: newEdges }); 
      }

      // Add Assistant Message
      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
      };

      set((state) => ({ 
        chatHistory: [...state.chatHistory, assistantMsg],
        isLoading: false
      }));

    } catch (error) {
      console.error("Chat Error:", error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "I encountered an error analyzing the repository. Please ensure the backend is running.",
        timestamp: Date.now(),
      };
      set((state) => ({ 
         chatHistory: [...state.chatHistory, errorMsg],
         isLoading: false
      }));
    }
  },

  ingestRepo: async (url: string) => {
    set({ isIngesting: true });
    try {
      // Add system message
      const startMsg: ChatMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Initiating excavation of ${url}... please wait while we analyze the codebase.`,
        timestamp: Date.now(),
      };
      set((state) => ({ chatHistory: [...state.chatHistory, startMsg] }));

      await api.ingest({ repo_url: url, clear_existing: true });
      
      const successMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Repository successfully ingested! The knowledge graph is ready.`,
        timestamp: Date.now(),
      };
      set((state) => ({ 
        chatHistory: [...state.chatHistory, successMsg],
        isIngesting: false,
        currentRepoUrl: url 
      }));
      
    } catch (error) {
      console.error("Ingest Error:", error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Failed to ingest repository: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
      };
      set((state) => ({ 
        chatHistory: [...state.chatHistory, errorMsg],
        isIngesting: false,
        currentRepoUrl: null
      }));
    }
  },

  deleteRepo: async () => {
    try {
      await api.deleteRepo();
      set({ 
        currentRepoUrl: null, 
        chatHistory: [], 
        nodes: [], 
        edges: [] 
      });
    } catch (error) {
      console.error("Delete Error:", error);
    }
  },

  init: async () => {
    try {
      const status = await api.getStatus();
      if (status.repo_url) {
        set({ currentRepoUrl: status.repo_url });
        // Optionally fetch initial graph stats or greeting here
      }
    } catch (error) {
       console.error("Init Error:", error);
    }
  },

  addChatMessage: (message: ChatMessage) => 
    set((state) => ({ chatHistory: [...state.chatHistory, message] })),
    
  clearChat: () => set({ chatHistory: [] }),
}));
