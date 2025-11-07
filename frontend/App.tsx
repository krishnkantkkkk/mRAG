import React, { useState } from 'react';
import Landing from './components/Landing';
import ChatPage from './components/ChatPage';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

const App: React.FC = () => {
  return(
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing/>} />
        <Route path="/chat" element={<ChatPage/>} />
      </Routes>
    </BrowserRouter>
  );
};
export default App;