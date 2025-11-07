
import React, { useState, useEffect, useRef, useCallback } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import BeforeChatSection from './BeforeChatSection';
import { ChatMessage as MessageType, MessageSender, Source } from '../types';
import { APP_NAME, API_BASE_URL } from '../constants';

const ChatWindow: React.FC = () => {
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // State for Audio Player
  const [showAudioPlayer, setShowAudioPlayer] = useState<boolean>(false);
  const [currentAudioSrc, setCurrentAudioSrc] = useState<string>('');
  const [currentAudioStartTime, setCurrentAudioStartTime] = useState<number>(0);
  const audioRef = useRef<HTMLAudioElement>(null);

  // State for DOCX/Text Modal
  const [showDocxModal, setShowDocxModal] = useState<boolean>(false);
  const [docxModalTitle, setDocxModalTitle] = useState<string>('');
  const [docxModalContent, setDocxModalContent] = useState<string>('');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Effect to play audio when src and start time change
  useEffect(() => {
    if (showAudioPlayer && currentAudioSrc && audioRef.current) {
      audioRef.current.src = currentAudioSrc;
      audioRef.current.currentTime = currentAudioStartTime;
      audioRef.current.play().catch(e => console.error("Error playing audio:", e));
    } else if (audioRef.current) {
      audioRef.current.pause();
    }
  }, [showAudioPlayer, currentAudioSrc, currentAudioStartTime]);

  const handleInputChange = (value: string) => {
    setInputMessage(value);
  };

  const handleSendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    // Create a new user message
    const userMessage: MessageType = {
      id: Date.now().toString(),
      sender: MessageSender.User,
      content: content,
      timestamp: new Date(),
    };
    
    const thinkingMessage: MessageType = {
      id: `ai-thinking-${Date.now()}`,
      sender: MessageSender.AI,
      content: 'Thinking...',
      timestamp: new Date(),
      isStreaming: true,
    };
    setMessages(prev => [...prev, userMessage, thinkingMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Send message to backend
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: content }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }
      
      // Create a stable ID for the definitive AI message
      const aiMessageId = Date.now().toString();
      const aiMessage: MessageType = {
        id: aiMessageId,
        sender: MessageSender.AI,
        content: '',
        timestamp: new Date(),
        isStreaming: true, // Start in streaming state
      };

      // Replace "Thinking..." message with the definitive AI message shell
      setMessages(prev => {
        const newMessages = prev.filter(m => m.id !== thinkingMessage.id);
        return [...newMessages, aiMessage];
      });


      // Read the stream
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      try {
        let buffer = '';
        let messageContent = '';
        let messageSources: Source[] | undefined;
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // Append to buffer
          buffer += decoder.decode(value, { stream: true });

          // Look for complete JSON objects at the end of the buffer
          const jsonMatch = buffer.match(/\{[\s\S]*\}$/);
          if (jsonMatch) {
            try {
              const jsonData = JSON.parse(jsonMatch[0]);
              if (jsonData.type === 'sources') {
                messageSources = jsonData.content;
                // Remove the JSON from the buffer
                buffer = buffer.replace(jsonMatch[0], '');
              }
            } catch (e) {
              // Not a complete or valid JSON object, continue
            }
          }

          // Update message content if we have non-JSON text
          if (buffer.trim()) {
            messageContent += buffer;
            buffer = ''; // Clear the buffer

            // Ensure the content is properly formatted as HTML
            const formattedContent = messageContent
              .split('\n')
              .map(line => `<p>${line}</p>`)
              .join('');

            // Update the message with new content
            setMessages(prev => {
              const newMessages = [...prev];
              const messageIndex = newMessages.findIndex(m => m.id === aiMessageId);

              if (messageIndex !== -1) {
                newMessages[messageIndex] = {
                  ...newMessages[messageIndex],
                  content: formattedContent,
                  sources: messageSources,
                  isStreaming: true, // Keep streaming state true
                };
              }
              return newMessages;
            });
          } else if (messageSources) {
            // If we have sources but no new content, update just the sources
            setMessages(prev => {
              const lastMessage = prev[prev.length - 1];
              if (lastMessage.id === aiMessage.id) {
                return [
                  ...prev.slice(0, -1),
                  {
                    ...lastMessage,
                    content: messageContent,
                    sources: messageSources,
                    isStreaming: lastMessage.isStreaming ?? false, // do not start streaming on sources-only updates
                  },
                ];
              }
              return prev;
            });
          }
        }

        // Final update to ensure everything is marked as complete
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage.id === aiMessage.id) {
            return [
              ...prev.slice(0, -1),
              {
                ...lastMessage,
                content: messageContent || 'No response generated.',
                sources: messageSources,
                isStreaming: false,
              },
            ];
          }
          return prev;
        });
      } finally {
        reader.releaseLock();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message to chat
      const errorMessage: MessageType = {
        id: Date.now().toString(),
        sender: MessageSender.AI,
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [setMessages, setInputMessage, setIsLoading]);

  const handleFileUpload = useCallback(async (files: FileList) => {
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      
      // Add a system message about the successful upload
      const systemMessage: MessageType = {
        id: Date.now().toString(),
        sender: MessageSender.AI,
        content: `Successfully uploaded ${files.length} file(s): ${Array.from(files).map(f => f.name).join(', ')}`,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, systemMessage]);
    } catch (error) {
      console.error('Error uploading files:', error);
      // Add error message to chat
      const errorMessage: MessageType = {
        id: Date.now().toString(),
        sender: MessageSender.AI,
        content: 'Sorry, there was an error uploading your files. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [setMessages, setIsLoading]);

  const handleSelectPrompt = useCallback((prompt: string) => {
    setInputMessage(prompt);
  }, []);

  const formatTime = (totalSeconds: number): string => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };

  const handleDocxSourceClick = useCallback((filename: string, content: string) => {
    setDocxModalTitle(`Source from: ${filename}`);
    setDocxModalContent(content);
    setShowDocxModal(true);
  }, []);

  const handleAudioSourceClick = useCallback((filename: string, startTime: number) => {
    setCurrentAudioSrc(`/temp/${filename}`);
    setCurrentAudioStartTime(startTime);
    setShowAudioPlayer(true);
  }, []);

  const handleAudioClose = useCallback(() => {
    setShowAudioPlayer(false);
    setCurrentAudioSrc('');
    setCurrentAudioStartTime(0);
    if (audioRef.current) {
      audioRef.current.pause();
    }
  }, []);

  const sendMessage = useCallback(async () => {
    if (inputMessage.trim()) {
      await handleSendMessage(inputMessage);
    }
  }, [inputMessage, handleSendMessage]);


  return (
    <div className="mx-auto flex h-full w-full max-w-7xl flex-col px-6 py-8 md:px-12 md:py-10">
      <h1 className="text-center text-4xl font-bold text-gray-800 mb-8 mt-4 font-display">
        <span className="font-code text-3xl text-gray-500">.{APP_NAME}</span>
      </h1>
      {messages.length === 0 ? (
        <BeforeChatSection onSelectPrompt={handleSelectPrompt} />
      ) : (
        <div className="flex-grow space-y-8 overflow-y-auto pr-2">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onDocxSourceClick={handleDocxSourceClick}
              onAudioSourceClick={handleAudioSourceClick}
              formatTime={formatTime}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}
      
      {/* Audio Player Container */}
      {showAudioPlayer && (
        <div className="p-6 pt-0 relative bg-white border-t border-gray-200 rounded-lg shadow-md mt-4">
          <audio ref={audioRef} controls className="w-full"></audio>
          <button
            id="audio-close-button"
            className="absolute top-2 right-2 bg-gray-200 text-gray-700 rounded-full w-6 h-6 flex items-center justify-center font-bold text-lg hover:bg-gray-300 transition-colors"
            onClick={handleAudioClose}
            aria-label="Close audio player"
          >&times;</button>
        </div>
      )}

      {/* Chat Input */}
      <div className="flex-shrink-0">
        <ChatInput
          inputMessage={inputMessage}
          onInputChange={handleInputChange}
          onSendMessage={handleSendMessage}
          onFileUpload={handleFileUpload}
          isLoading={isLoading}
        />
      </div>

      {/* DOCX Modal */}
      {showDocxModal && (
        <div
          className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-4 z-50"
          onClick={() => setShowDocxModal(false)} // Close on background click
          role="dialog"
          aria-modal="true"
          aria-labelledby="modal-title"
        >
          <div
            className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside modal content
          >
            <div className="flex justify-between items-center p-4 border-b border-gray-200">
              <h2 id="modal-title" className="text-lg font-bold text-gray-800">{docxModalTitle}</h2>
              <button
                id="modal-close-button"
                className="text-gray-500 hover:text-gray-700 text-3xl leading-none"
                onClick={() => setShowDocxModal(false)}
                aria-label="Close modal"
              >&times;</button>
            </div>
            <div id="modal-content" className="prose prose-sm dark:prose-invert p-4 overflow-y-auto max-w-none">
              <p dangerouslySetInnerHTML={{ __html: docxModalContent.replace(/\n/g, '<br>') }}></p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWindow;
