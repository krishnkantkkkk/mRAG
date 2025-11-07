import React from 'react';
import HeroSection from './HeroSection';
import FeatureCard from './FeatureCard';
import Footer from './Footer';
import ScreenshotSection from './ScreenshotSection'; // Import the new component

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen flex flex-col">
      {/* .mRAG branding in the corner */}
      <div className="fixed top-4 left-6 z-50 text-2xl font-bold text-gray-900 font-code">
        .mRAG
      </div>
      <main className="flex-grow">
        <HeroSection />
        <ScreenshotSection /> {/* Insert the new screenshot section here */}

        <section id="features" className="py-16 md:py-24 bg-gray-50 px-4 md:px-8">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 text-center mb-12">
              Powerful Features for <span className="text-gray-600">Deep Understanding</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                  </svg>
                }
                title="Document Comprehension"
                description="Process DOCX and PDF files with unparalleled accuracy, extracting key information and providing precise textual references."
              />
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a4 4 0 110-8 4 4 0 010 8z"></path>
                  </svg>
                }
                title="Audio Intelligence"
                description="Analyze audio files, transcribe spoken content, and link specific insights to precise timestamps within the recording."
              />
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z"></path>
                  </svg>
                }
                title="Unified Knowledge Base"
                description="Seamlessly integrate insights from diverse modalities into a single, comprehensive knowledge base for holistic understanding."
              />
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 21h7a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414a1 1 0 00-.707-.293H7a2 2 0 00-2 2v11a2 2 0 002 2h1.5"></path>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 12H9"></path>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 16H9"></path>
                  </svg>
                }
                title="Pinpoint Accuracy & Context"
                description="Retrieve answers with direct, verifiable references to the exact source location in documents or precise moments in audio."
              />
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                }
                title="Faster Insights"
                description="Dramatically reduce research time with intelligent retrieval that cuts through noise to deliver only relevant, contextual information."
              />
              <FeatureCard
                icon={
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path>
                  </svg>
                }
                title="Scalable & Secure"
                description="Built on robust infrastructure, our RAG solution scales with your needs while ensuring the security and privacy of your data."
              />
            </div>
          </div>
        </section>

        <section id="how-it-works" className="py-16 md:py-24 px-4 md:px-8 bg-white border-t border-gray-100">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-8">
              How It <span className="text-gray-600">Works</span>
            </h2>
            <p className="text-lg text-gray-700 mb-12 leading-relaxed">
              Our Multimodal RAG system intelligently ingests and processes various data types,
              creating a rich knowledge graph. When you ask a question, it queries this graph,
              retrieving precise information linked directly to its original source.
            </p>

            <div className="relative flex flex-col items-center py-8">
              {/* Timeline/Flow Connector */}
              <div className="absolute left-1/2 transform -translate-x-1/2 w-0.5 bg-gray-200 h-full hidden md:block"></div>

              {/* Step 1 */}
              <div className="flex flex-col md:flex-row items-center justify-center w-full my-8">
                <div className="md:w-1/2 text-center md:text-right md:pr-12">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">1. Ingest Data</h3>
                  <p className="text-gray-700">Upload your DOCX, PDF, and audio files. Our system processes them efficiently.</p>
                </div>
                <div className="flex-shrink-0 relative my-4 md:my-0">
                  <div className="w-12 h-12 rounded-full bg-gray-900 flex items-center justify-center text-white text-lg font-bold shadow-md">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                  </div>
                </div>
                <div className="md:w-1/2 hidden md:block"></div>
              </div>

              {/* Step 2 */}
              <div className="flex flex-col md:flex-row items-center justify-center w-full my-8">
                <div className="md:w-1/2 hidden md:block"></div>
                <div className="flex-shrink-0 relative my-4 md:my-0">
                  <div className="w-12 h-12 rounded-full bg-gray-900 flex items-center justify-center text-white text-lg font-bold shadow-md">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                  </div>
                </div>
                <div className="md:w-1/2 text-center md:text-left md:pl-12">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">2. Intelligent Indexing</h3>
                  <p className="text-gray-700">Our advanced AI models analyze content, creating rich embeddings and identifying key entities, references, and timestamps.</p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex flex-col md:flex-row items-center justify-center w-full my-8">
                <div className="md:w-1/2 text-center md:text-right md:pr-12">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">3. Ask Questions</h3>
                  <p className="text-gray-700">Pose complex queries. Our system understands your intent across all data types.</p>
                </div>
                <div className="flex-shrink-0 relative my-4 md:my-0">
                  <div className="w-12 h-12 rounded-full bg-gray-900 flex items-center justify-center text-white text-lg font-bold shadow-md">
                    <span className="material-symbols-outlined">help</span>
                  </div>
                </div>
                <div className="md:w-1/2 hidden md:block"></div>
              </div>

              {/* Step 4 */}
              <div className="flex flex-col md:flex-row items-center justify-center w-full my-8">
                <div className="md:w-1/2 hidden md:block"></div>
                <div className="flex-shrink-0 relative my-4 md:my-0">
                  <div className="w-12 h-12 rounded-full bg-gray-900 flex items-center justify-center text-white text-lg font-bold shadow-md">
                    <span className="material-symbols-outlined">comment</span>
                  </div>
                </div>
                <div className="md:w-1/2 text-center md:text-left md:pl-12">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-2">4. Get Precise Answers</h3>
                  <p className="text-gray-700">Receive accurate, context-rich answers with direct links to the relevant sections in your documents or audio files.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="contact" className="py-16 md:py-24 bg-gray-50 px-4 md:px-8 border-t border-gray-100">
          <div className="max-w-xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-8">
              Ready to <span className="text-gray-600">Transform</span> Your Data?
            </h2>
            <p className="text-lg text-gray-700 mb-10 leading-relaxed">
              Join the waitlist or get in touch with our team to learn how Multimodal RAG can revolutionize your information retrieval.
            </p>
            <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
              <a
                href="#"
                className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-gray-900 hover:bg-gray-800 transition-all duration-300 shadow-md transform hover:-translate-y-0.5"
              >
                Join Waitlist
              </a>
              <a
                href="#"
                className="inline-flex items-center justify-center px-8 py-3 border border-gray-300 text-base font-medium rounded-md text-gray-800 bg-white hover:bg-gray-50 transition-all duration-300 shadow-sm"
              >
                Contact Sales
              </a>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default Landing;