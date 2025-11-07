
import React from 'react';
import ReactMarkdown from 'react-markdown';
import { ChatMessage as MessageType, MessageSender, Source } from '../types';

interface ChatMessageProps {
  message: MessageType;
  onDocxSourceClick: (filename: string, content: string) => void;
  onAudioSourceClick: (filename: string, startTime: number) => void;
  formatTime: (totalSeconds: number) => string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  onDocxSourceClick, 
  onAudioSourceClick,
  formatTime,
}) => {
  const isUser = message.sender === MessageSender.User;
  
  const bgColorClass = isUser ? 'bg-gray-50' : 'bg-white'; 
  const textColorClass = 'text-gray-700';

  const renderSourceIcon = (type: Source['type']) => {
    switch (type) {
      case 'pdf': return 'picture_as_pdf';
      case 'docx':
      case 'text': return 'description';
      case 'audio': return 'audiotrack';
      case 'image': return 'image'; // Not explicitly used for link, but good for consistency
      default: return 'article';
    }
  };

  const renderSourceColor = (type: Source['type']) => {
    switch (type) {
      case 'pdf': return 'text-red-500';
      case 'docx':
      case 'text': return 'text-blue-500';
      case 'audio': return 'text-orange-500';
      default: return 'text-gray-500';
    }
  };

  // Show "Thinking..." only when this is an AI message that has no content yet
  // and streaming has not started. Once streaming starts we display the
  // content in real time.
  const showThinking = !isUser && (!message.content || message.content.trim() === '') && !message.isStreaming;

  return (
    <div className={`relative max-w-[90%] ${isUser ? 'ml-auto' : 'mr-auto'}`}>
      <div className={`rounded-lg ${bgColorClass} p-3 ${textColorClass} text-base shadow-sm prose prose-sm max-w-none`}>
        {showThinking ? (
          <div className="animate-pulse text-gray-500">Thinking...</div>
        ) : (
          <>
            {/* Render inline images first if any */}
            {message.sources && message.sources.map((src, index) => (
              src.type === 'image' && src.image_path ? (
                <div key={`image-${message.id}-${index}`} className="my-2 max-w-full h-auto flex justify-center">
                  <img 
                    src={`/temp/${src.image_path}`} 
                    alt={src.source_filename || "Generated image"} 
                    className="max-w-full h-auto rounded-lg border border-gray-200 object-contain"
                    style={{ maxHeight: "300px" }}
                  />
                </div>
              ) : null
            ))}

            <div
              className="chat-message-content"
              dangerouslySetInnerHTML={{ __html: message.content }}
            />

            {message.sources && message.sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-gray-100"> {/* Added a subtle separator */}
                <h4 className="text-xs font-semibold text-gray-500 mb-2">Sources:</h4>
                <div className="flex flex-wrap gap-2 justify-start"> {/* Align to start for AI messages */}
                  {message.sources.map((src, index) => {
                    const icon = renderSourceIcon(src.type);
                    const iconColor = renderSourceColor(src.type);
                    
                    if (src.type === 'image') return null; // Already rendered inline

                    if (src.type === 'pdf' && src.page_num !== undefined) {
                      return (
                        <a 
                          key={`source-${message.id}-${index}`} 
                          href={`/temp/${src.source_filename}#page=${src.page_num}`} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-md text-gray-600 text-xs hover:bg-gray-200 transition-colors cursor-pointer"
                          aria-label={`Open PDF source: ${src.source_filename} page ${src.page_num}`}
                        >
                          <span className={`material-symbols-outlined !text-base ${iconColor}`}>{icon}</span>
                          <span className="truncate max-w-[120px]">{src.source_filename.split('/').pop()}</span> p.{src.page_num}
                        </a>
                      );
                    } else if ((src.type === 'docx' || src.type === 'text') && src.source_content) {
                      return (
                        <button 
                          key={`source-${message.id}-${index}`} 
                          onClick={() => onDocxSourceClick(src.source_filename, src.source_content || '')}
                          className="flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-md text-gray-600 text-xs hover:bg-gray-200 transition-colors cursor-pointer"
                          aria-label={`View document source: ${src.source_filename}`}
                        >
                          <span className={`material-symbols-outlined !text-base ${iconColor}`}>{icon}</span>
                          <span className="truncate max-w-[120px]">{src.source_filename.split('/').pop()}</span>
                        </button>
                      );
                    } else if (src.type === 'audio' && src.start_time !== undefined && src.end_time !== undefined) {
                      return (
                        <button 
                          key={`source-${message.id}-${index}`} 
                          onClick={() => onAudioSourceClick(src.source_filename, src.start_time || 0)}
                          className="flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-md text-gray-600 text-xs hover:bg-gray-200 transition-colors cursor-pointer"
                          aria-label={`Play audio from ${formatTime(src.start_time)} to ${formatTime(src.end_time)} in ${src.source_filename}`}
                        >
                          <span className={`material-symbols-outlined !text-base ${iconColor}`}>{icon}</span>
                          <span className="truncate max-w-[120px]">{src.source_filename.split('/').pop()}</span> {formatTime(src.start_time)} - {formatTime(src.end_time)}
                        </button>
                      );
                    }
                    return null;
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
      {!isUser && !message.isStreaming && (
        <div className="mt-2 flex items-center gap-2 text-gray-500">
          <button className="p-1 hover:text-gray-700" aria-label="Like message"><span className="material-symbols-outlined !text-lg">thumb_up</span></button>
          <button className="p-1 hover:text-gray-700" aria-label="Dislike message"><span className="material-symbols-outlined !text-lg">thumb_down</span></button>
          <button className="p-1 hover:text-gray-700" aria-label="Copy message to clipboard"><span className="material-symbols-outlined !text-lg">content_copy</span></button>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
