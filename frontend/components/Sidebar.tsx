import React from 'react';
import { AI_AVATAR_URL, USER_AVATAR_URL, APP_NAME } from '../constants';
import { Link } from 'react-router-dom';

interface SidebarProps {
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen, onToggleSidebar }) => {
  return (
    <aside
      className={`
        flex flex-col items-center justify-between py-6 mt-4 mb-4 rounded-lg
        fixed inset-y-0 left-0 z-20 w-60 bg-[#E5E5E5] shadow-lg transform transition-transform duration-300 ease-in-out
        ${isSidebarOpen ? 'translate-x-4' : '-translate-x-full'}
        md:relative md:translate-x-0 md:w-10 md:flex-shrink-0 md:shadow-none
      `}
    >
      <button
        className="absolute flex top-4 right-4 md:hidden p-2 rounded-full bg-white shadow-md text-gray-800"
        onClick={onToggleSidebar}
        aria-label="Close sidebar"
      >
        <span className="material-symbols-outlined">close</span>
      </button>

      
      <Link to="/"><button className="font-code text-gray-500 font-bold">.{APP_NAME}</button></Link>
      <div className="flex flex-col items-center gap-6">
        <div className="flex flex-col items-center gap-4">
          <a className="flex items-center rounded-full text-gray-600 transition-all duration-200 hover:bg-gray-300 hover:text-gray-800 w-[120px] md:w-auto" href="#">
            <span className="material-symbols-outlined rounded-full p-2">add</span><span className='md:hidden'>New</span>
          </a>
          <a className="flex items-center rounded-full text-gray-600 transition-all duration-200 hover:bg-gray-300 hover:text-gray-800 w-[120px] md:w-auto" href="#">
            <span className="material-symbols-outlined rounded-full p-2">search</span><span className='md:hidden'>Search</span>
          </a>
          <a className="flex items-center rounded-full text-gray-600 transition-all duration-200 hover:bg-gray-300 hover:text-gray-800 w-[120px] md:w-auto" href="#">
            <span className="material-symbols-outlined rounded-full p-2">history</span><span className='md:hidden'>History</span>
          </a>
        </div>
      </div>
      <div className="flex flex-col items-center gap-4">
        <a className="flex items-center rounded-full text-gray-600 transition-all duration-200 hover:bg-gray-300 hover:text-gray-800 w-[120px] md:w-auto md:hover:rotate-90" href="#">
          <span className="material-symbols-outlined rounded-full p-2">settings</span> <span className='md:hidden'>Settings</span>
        </a>
      </div>
    </aside>
  );
};

export default Sidebar;