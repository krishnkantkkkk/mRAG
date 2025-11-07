import React, { useMemo } from 'react';
import './BeforeChatSection.css';
import { AI_AVATAR_URL } from '../constants';

interface BeforeChatSectionProps {
  onSelectPrompt: (prompt: string) => void;
}

const BeforeChatSection: React.FC<BeforeChatSectionProps> = ({ onSelectPrompt }) => {
  const numCircles = 25;
  const minRadius = 5;  // innermost circle radius
  const maxRadius = 12; // outermost circle radius
  const center = { x: 50, y: 50 };
  const animationBaseDuration = '320s';

  const circles = useMemo(() => {
    const arr = [];
    for (let i = 0; i < numCircles; i++) {
      const radius = minRadius + i * ((maxRadius - minRadius) / (numCircles - 1));
      const circumference = 2 * Math.PI * radius;
      const dashLength = circumference * 0.5;
      const gapLength = circumference - dashLength;
      const dashArray = `${dashLength} ${gapLength}`;
      const animationDirection = i % 2 === 0 ? 'normal' : 'reverse';

      arr.push(
        <circle
          key={i}
          cx={center.x}
          cy={center.y}
          r={radius}
          className="animated-circle"
          style={{
            strokeDasharray: dashArray,
            animationDuration: animationBaseDuration,
            animationDirection,
          }}
        />
      );
    }
    return arr;
  }, []);

  return (
    <div className="w-full h-full relative overflow-hidden bg-white" aria-hidden="true">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="xMidYMid slice"
      >
        {circles}
      </svg>
      <div
        className="font-code font-bold text-gray-500 bg-transparent absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 
                   rounded-full transition-all duration-500 ease-in-out md:text-[2vw] cursor-default"
      >.mRAG</div>
    </div>
  );
};

export default BeforeChatSection;
