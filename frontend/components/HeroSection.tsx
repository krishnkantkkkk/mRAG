
import React from 'react';
import { Link } from 'react-router-dom';

const HeroSection: React.FC = () => {
  return (
    <section className="relative flex mb-20 bg-gradient-to-b from-gray-50 to-white pt-20 px-4 md:px-8 overflow-hidden">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-16 z-10 py-16">
        <div className="text-center md:text-left">
          <h1 className="text-6xl md:text-7xl font-extrabold leading-tight text-gray-900 mb-6 drop-shadow-sm">
            Understand Everything, <span className="text-gray-600">Instantly.</span>
          </h1>
          <p className="text-lg md:text-xl text-gray-700 mb-10 max-w-2xl mx-auto md:mx-0 leading-relaxed">
            Unlock comprehensive insights from documents and audio with precise, timestamped references, transforming how you interact with information.
          </p>
          <div className="flex flex-col sm:flex-row justify-center md:justify-start space-y-4 sm:space-y-0 sm:space-x-4">
            <Link to="/chat">
                <a className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-gray-900 hover:bg-gray-800 transition-all duration-300 shadow-md transform">
                    Open App
                </a>
            </Link>
          </div>
        </div>

        {/* Right Content - Icons */}
        <div className="relative w-full h-0 md:h-96 flex items-center justify-center md:-mt-16">
          {/* Floating Material Icons - forming a triangle */}
          <div className="absolute material-symbols-outlined text-[60rem] text-gray-200 coin-effect opacity-[20%] rotate-0 cursor-default select-none hidden md:block">
            sports_volleyball
          </div>
          <div className="font-code font-bold text-[2rem] text-gray-600 coin-effect cursor-default select-none hidden md:block">
            .mRAG
          </div>
        </div>
      </div>
      <style>{`
        /* New floating animations for icons */
        @keyframes floating {
          0% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(0, -10px) scale(1.03); }
          100% { transform: translate(0, 0) scale(1); }
        }
        .animate-float-slow {
          animation: floating 8s ease-in-out infinite;
        }
        .animate-float-medium {
          animation: floating 7s ease-in-out infinite 1s; /* Add delay */
        }
        .animate-float-fast {
          animation: floating 6s ease-in-out infinite 0.5s; /* Add delay */
        }

        /* 3D Coin Effect for icons - adjusted for darker text */
        .coin-effect {
          text-shadow:
            0px 1px 0px rgba(255, 255, 255, 0.2), /* Subtle top highlight */
            0px 2px 0px rgba(255, 255, 255, 0.1), /* Even subtler second highlight */
            0px 3px 0px rgba(255, 255, 255, 0.3),      /* Depth layer 1 */
            0px 4px 0px rgba(0, 0, 0, 0.4),      /* Depth layer 2 */
            0px 5px 5px rgba(0, 0, 0, 0.5);      /* Soft drop shadow */
        }
      `}</style>
    </section>
  );
};

export default HeroSection;