

import React from 'react';
import { MAX_INPUT_LENGTH } from '../constants';
import Microphone from './Microphone';

interface ChatInputProps {
  inputMessage: string;
  onInputChange: (value: string) => void;
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  onFileUpload?: (files: FileList) => Promise<void>;
}

const ChatInput: React.FC<ChatInputProps> = ({
  inputMessage,
  onInputChange,
  onSendMessage,
  isLoading,
  onFileUpload,
}) => {
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (inputMessage.trim()) {
      onSendMessage(inputMessage.trim());
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0 && onFileUpload) {
      try {
        await onFileUpload(files);
      } catch (error) {
        console.error('Error uploading files:', error);
      }
      // Reset the input value to allow uploading the same file again
      event.target.value = '';
    }
  };

  return (
    <div className="mt-8 max-w-4xl mx-auto flex-shrink-0">
      <div className="flex items-center rounded-xl border border-gray-200 bg-white shadow-sm flex items-end p-2">
        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
          multiple
          accept=".pdf,.doc,.docx,.txt,.mp3,.wav"
        />
        
        {/* Attach File Icon (left) */}
        <button
          className="flex-shrink-0 h-9 w-9 flex items-center justify-center text-gray-500 transition-colors hover:text-gray-700 hover:bg-[#e0e0e0] rounded-[50%] disabled:opacity-50"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          aria-label="Attach file"
        >
          <span className="material-symbols-outlined text-xl">attach_file</span>
        </button>

        {/* Textarea */}
        <textarea
          className="flex-grow resize-none border-0 bg-transparent text-gray-800 placeholder:text-gray-400 focus:ring-0 min-h-[2.25rem] py-2" // min-h-[2.25rem] = 36px for single line, matches button height
          placeholder="Ask anything..."
          rows={1}
          value={inputMessage}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={handleKeyPress}
          maxLength={MAX_INPUT_LENGTH}
          disabled={isLoading}
        ></textarea>

        {/* Microphone Component */}
        <Microphone onTranscriptionSuccess={onInputChange} />

        {/* Send Button (far right) */}
        <button
          className="flex-shrink-0 flex h-9 w-9 items-center justify-center rounded-lg bg-gray-800 text-white transition-opacity hover:opacity-90 disabled:opacity-50 ml-2" // Changed from gradient to bg-gray-800
          onClick={handleSend}
          disabled={isLoading || !inputMessage.trim()}
          aria-label="Send message"
        >
          <span className="material-symbols-outlined !text-xl">arrow_upward</span>
        </button>
      </div>
    </div>
  );
};

export default ChatInput;