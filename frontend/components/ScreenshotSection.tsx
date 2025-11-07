import React from 'react';

const ScreenshotSection: React.FC = () => {
  return (
    <section className="relative z-20 -mt-20 w-full px-4 md:px-8 bg-gradient-to-b to-gray-50 from-white">
      <div className="w-[90%] mx-auto rounded-lg shadow-2xl overflow-hidden">
        <img
          src="../assets/screenshot.png"
          className="w-full h-auto object-cover"
        />
      </div>
    </section>
  );
};

export default ScreenshotSection;