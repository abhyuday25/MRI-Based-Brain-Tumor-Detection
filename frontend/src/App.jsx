import { useState, useRef } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Image as ImageIcon, 
  Loader2, 
  CheckCircle2, 
  AlertCircle,
  Brain,
  FileWarning
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file (JPG, PNG).');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file (JPG, PNG).');
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'An error occurred during classification.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-950 p-6 md:p-12">
      {/* Header */}
      <div className="max-w-4xl w-full text-center mb-12 animate-fade-in">
        <div className="flex justify-center mb-4">
          <div className="p-3 bg-primary-900/30 rounded-2xl border border-primary-500/20">
            <Brain className="w-10 h-10 text-primary-400" />
          </div>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          Brain Tumor Classification
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Upload an MRI scan image to detect and classify brain tumors using deep learning.
        </p>
      </div>

      {/* Main Content */}
      <main className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* Upload Section */}
        <div className="glass rounded-3xl p-8 animate-fade-in" style={{animationDelay: '0.1s'}}>
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Upload className="w-5 h-5 text-primary-400" />
            Upload MRI Scan
          </h2>
          
          <div 
            onClick={() => !loading && fileInputRef.current.click()}
            onDragOver={onDragOver}
            onDrop={onDrop}
            className={`
              relative border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer h-64 flex flex-col items-center justify-center
              ${previewUrl ? 'border-primary-500/50 bg-primary-500/5' : 'border-gray-700 hover:border-gray-500 hover:bg-white/5'}
            `}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileSelect} 
              className="hidden" 
              accept="image/*"
            />
            
            {previewUrl ? (
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="absolute inset-0 w-full h-full object-contain p-2 rounded-xl"
              />
            ) : (
              <div className="text-center group">
                <div className="mx-auto w-16 h-16 bg-gray-900 rounded-full flex items-center justify-center mb-4 border border-gray-800 group-hover:scale-110 transition-transform">
                  <ImageIcon className="w-8 h-8 text-gray-500" />
                </div>
                <p className="text-gray-300 font-medium mb-1">Click or drag & drop</p>
                <p className="text-gray-500 text-sm">PNG, JPG or JPEG</p>
              </div>
            )}
          </div>

          <div className="mt-6 flex gap-3">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || loading}
              className="flex-1 py-4 px-6 bg-primary-600 hover:bg-primary-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-primary-600/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-5 h-5" />
                  Classify Image
                </>
              )}
            </button>
            
            <button
              onClick={reset}
              disabled={loading || !selectedFile}
              className="py-4 px-6 bg-gray-900 hover:bg-gray-800 text-gray-400 hover:text-white font-semibold rounded-xl transition-all border border-gray-800"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="h-full flex flex-col">
          {error && (
            <div className="glass border-red-500/30 bg-red-500/5 rounded-3xl p-8 mb-6 animate-fade-in flex items-start gap-4">
              <AlertCircle className="w-6 h-6 text-red-500 shrink-0 mt-1" />
              <div>
                <h3 className="text-red-400 font-semibold mb-1">Error</h3>
                <p className="text-gray-400">{error}</p>
              </div>
            </div>
          )}

          {result ? (
            <div className="glass border-green-500/30 bg-green-500/5 rounded-3xl p-8 animate-fade-in flex-1">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                  Classification Result
                </h2>
                <span className="px-3 py-1 bg-green-500/20 text-green-400 text-xs font-bold rounded-full uppercase tracking-wider">
                  Complete
                </span>
              </div>

              <div className="space-y-8">
                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 block">
                    Tumor Type
                  </label>
                  <p className="text-4xl font-bold text-white capitalize">
                    {result.class.replace('_', ' ')}
                  </p>
                </div>

                <div>
                  <label className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 block">
                    Confidence Score
                  </label>
                  <div className="flex items-end gap-3">
                    <p className="text-6xl font-black text-primary-400">
                      {result.confidence}%
                    </p>
                  </div>
                  <div className="w-full bg-gray-800 h-3 rounded-full mt-4 overflow-hidden">
                    <div 
                      className="bg-primary-500 h-full transition-all duration-1000 ease-out rounded-full"
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ) : !loading && !error ? (
            <div className="glass rounded-3xl p-8 border-gray-800/50 flex-1 flex flex-col items-center justify-center text-center opacity-70">
              <div className="w-20 h-20 bg-gray-900/50 rounded-full flex items-center justify-center mb-6">
                <Brain className="w-10 h-10 text-gray-700" />
              </div>
              <h3 className="text-gray-500 font-medium text-lg">Waiting for input</h3>
              <p className="text-gray-600 max-w-xs mx-auto mt-2">
                Classification results will appear here after analysis.
              </p>
            </div>
          ) : loading ? (
            <div className="glass rounded-3xl p-8 border-primary-500/20 flex-1 flex flex-col items-center justify-center text-center">
              <Loader2 className="w-16 h-16 text-primary-500 animate-spin mb-6" />
              <h3 className="text-white font-medium text-xl">Processing Image</h3>
              <p className="text-gray-400 max-w-xs mx-auto mt-2">
                Our neural network is analyzing the scan markers...
              </p>
            </div>
          ) : null}
        </div>
      </main>

      {/* Footer / Disclaimer */}
      <footer className="mt-12 max-w-4xl w-full">
        <div className="glass border-yellow-500/20 bg-yellow-500/5 rounded-2xl p-6 flex flex-col md:flex-row items-center gap-4">
          <FileWarning className="w-8 h-8 text-yellow-500 shrink-0" />
          <p className="text-sm text-gray-400 leading-relaxed text-center md:text-left">
            <strong className="text-yellow-500 font-semibold block mb-1">Medical Disclaimer:</strong>
            THIS APP IS FOR EDUCATIONAL PURPOSES ONLY. This application is a demonstration of deep learning and is not intended for diagnostic use. All medical decisions should be made by qualified healthcare professionals based on personal consultation.
          </p>
        </div>
        <p className="text-center text-gray-600 text-xs mt-8 pb-4">
          © 2026 Brain MRI Analysis Deep Learning Lab
        </p>
      </footer>
    </div>
  );
}

export default App;
