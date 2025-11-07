
import React from 'react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  return (
    <footer className="bg-gray-900 text-white py-8 px-6 md:px-12 mt-16">
      <div className="max-w-7xl mx-auto text-center">
        <p className="text-sm text-gray-400">
          &copy; {currentYear} <span className="font-code">.mRAG</span>. All rights reserved.
        </p>
        <div className="flex justify-center space-x-6 mt-4">
          <a href="#" className="text-gray-400 hover:text-white transition-colors duration-200 text-sm">Privacy Policy</a>
          <a href="#" className="text-gray-400 hover:text-white transition-colors duration-200 text-sm">Terms of Service</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;