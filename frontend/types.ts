

export enum MessageSender {
  User = 'user',
  AI = 'ai',
}

export interface Source {
  type: 'pdf' | 'docx' | 'text' | 'image' | 'audio';
  source_filename: string;
  page_num?: number;
  image_path?: string;
  start_time?: number;
  end_time?: number;
  source_content?: string; // For docx/text content
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  sources?: Source[]; // Added for source grounding
}
