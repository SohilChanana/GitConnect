import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Types defining the API contract
export interface AnalyzeRequest {
  query: string;
  use_vector_search?: boolean;
  max_depth?: number;
}

export interface Entity {
  name: string;
  type: string;
  file_path: string;
  content?: string;
  [key: string]: any;
}

export interface Dependency {
  name: string;
  type: string;
  line?: number;
  [key: string]: any;
}

export interface AnalyzeResponse {
  query: string;
  answer: string;
  cypher_query?: string;
  relevant_entities: Entity[];
  dependencies: Dependency[];
  edges?: { source: string; target: string; type: string }[];
  truncation_warning?: string;
}

export interface SummaryRequest {
  repo_url: string;
  prompt?: string;
}

export interface SummaryReport {
  repo_name: string;
  summary: string;
  tech_stack: string[];
  key_features: string[];
  viability_score: number;
  viability_analysis: string;
}

export interface IngestRequest {
  repo_url: string;
  clear_existing?: boolean;
}

export interface IngestResponse {
  status: string;
  message: string;
  stats?: any;
}

// API Service
export const api = {
  // Check API Health
  health: async () => {
    const res = await axios.get(`${API_BASE_URL}/health`);
    return res.data;
  },

  // Get Repo Status
  getStatus: async () => {
    const res = await axios.get(`${API_BASE_URL}/status`);
    return res.data;
  },

  // Delete/Reset Repo
  deleteRepo: async () => {
    const res = await axios.delete(`${API_BASE_URL}/delete`);
    return res.data;
  },

  // Ingest Repository
  ingest: async (req: IngestRequest): Promise<IngestResponse> => {
    const res = await axios.post(`${API_BASE_URL}/ingest`, req);
    return res.data;
  },

  // Analyze Impact (Chat)
  analyze: async (req: AnalyzeRequest): Promise<AnalyzeResponse> => {
    const res = await axios.post(`${API_BASE_URL}/analyze`, req);
    return res.data;
  },

  // Summarize Repo
  summarize: async (req: SummaryRequest): Promise<SummaryReport> => {
    const res = await axios.post(`${API_BASE_URL}/summarize`, req);
    return res.data;
  },
  
  // Get Graph Stats
  getStats: async () => {
    const res = await axios.get(`${API_BASE_URL}/stats`);
    return res.data;
  }
};
