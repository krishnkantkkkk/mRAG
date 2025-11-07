import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ChatWindow from './ChatWindow';

const ChatPage: React.FC = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isHamburgerOpen, setIsHamburgerOpen] = useState(false);
  const [showLandingPage, setShowLandingPage] = useState(true); // New state for landing page

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
    setIsHamburgerOpen(!isHamburgerOpen);
  };

  const handleStartChat = () => {
    setShowLandingPage(false); // Function to hide landing page and show chat
  };

  return (
        <div className="flex h-screen w-full p-4 lg:p-6 bg-[#E5E5E5] relative">
          {/* Hamburger icon for mobile */}
          {isHamburgerOpen ? <div className='hidden'></div> : <button
            className="fixed flex top-8 left-8 z-30 md:hidden p-2 rounded-full bg-white shadow-md text-gray-800"
            onClick={toggleSidebar}
            aria-label="Toggle sidebar"
          >
            <span className="material-symbols-outlined">menu</span>
          </button>}

          {/* Sidebar component */}
          <Sidebar isSidebarOpen={isSidebarOpen} onToggleSidebar={toggleSidebar} />

          {/* Overlay for mobile when sidebar is open */}
          {isSidebarOpen && (
            <div
              className="fixed inset-0 bg-black opacity-50 z-10 md:hidden"
              onClick={toggleSidebar}
            ></div>
          )}

          {/* Main content - adjust margin only on desktop */}
          <main className="flex-1 flex-col overflow-y-auto rounded-lg bg-white shadow-lg ml-0 md:ml-4">
            <ChatWindow />
          </main>
        </div>
  );
};
export default ChatPage;