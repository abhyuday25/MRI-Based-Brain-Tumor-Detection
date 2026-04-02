import { useState, useRef } from 'react';
import axios from 'axios';
import {
  Upload,
  Image as ImageIcon,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Brain,
  FileWarning,
  Microscope,
  ShieldAlert,
  Stethoscope,
  Activity,
  Pill,
  HeartPulse,
  Info,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const CLASS_META = {
  glioma_tumor: {
    label: 'Glioma Tumor',
    color: 'red',
    desc: 'A tumor originating from glial cells in the brain or spine. Gliomas range from Grade I (slow-growing) to Grade IV (glioblastoma, aggressive).',
    overview: 'Gliomas account for about 33% of all brain tumors. They arise from the supportive glial cells and can vary widely in aggressiveness. Grades I–II are typically slower-growing; Grades III–IV grow rapidly and require intensive treatment.',
    symptoms: [
      'Persistent or worsening headaches',
      'Seizures or convulsions',
      'Memory loss and cognitive decline',
      'Nausea and vomiting',
      'Vision, speech, or motor impairment',
      'Personality or mood changes',
    ],
    treatments: [
      'Surgical resection (craniotomy)',
      'Radiation therapy (external beam or stereotactic)',
      'Chemotherapy — temozolomide is standard for GBM',
      'Targeted therapy (e.g., bevacizumab for high-grade)',
      'Clinical trials for experimental immunotherapy',
    ],
    precautions: [
      'Schedule regular MRI follow-ups as advised by your oncologist',
      'Take anti-seizure medications as prescribed',
      'Avoid activities risking head trauma',
      'Inform your doctor of any new or worsening symptoms immediately',
      'Build a support network — caregivers and mental health support are crucial',
      'Discuss genetic counseling if family history of brain tumors is present',
    ],
    prognosis: 'Prognosis depends heavily on grade and location. Grade I–II gliomas can have 5–10+ year survival. Grade IV (glioblastoma) has a median survival of ~15 months with treatment.',
  },
  meningioma_tumor: {
    label: 'Meningioma Tumor',
    color: 'orange',
    desc: 'A tumor arising from the meninges, the membranes surrounding the brain and spinal cord. Most meningiomas are benign and slow-growing.',
    overview: 'Meningiomas are the most common primary brain tumors (~37%). They grow from the meninges and are usually benign (Grade I). Grades II and III are rarer but more aggressive. They occur more frequently in women and older adults.',
    symptoms: [
      'Headaches that worsen over time',
      'Vision disturbances or loss',
      'Hearing loss or tinnitus',
      'Weakness in arms or legs',
      'Seizures (especially in larger tumors)',
      'Memory and concentration difficulties',
    ],
    treatments: [
      'Active surveillance (watchful waiting) for small, asymptomatic tumors',
      'Surgical removal — often curative for benign tumors',
      'Stereotactic radiosurgery (Gamma Knife / CyberKnife)',
      'Conventional radiation for Grade II/III or recurrent tumors',
      'Hormone therapy research is ongoing (meningiomas have progesterone receptors)',
    ],
    precautions: [
      'Attend all scheduled imaging follow-ups even if asymptomatic',
      'Report any sudden changes in vision, hearing, or balance promptly',
      'Minimize unnecessary radiation exposure to the head',
      'Discuss hormone therapy risks with your doctor if applicable',
      'Maintain a healthy lifestyle — good nutrition, moderate exercise',
      'Seek neuropsychological support for cognitive symptoms',
    ],
    prognosis: 'Prognosis is generally excellent for Grade I. 5-year recurrence rate is ~7–20% after complete resection. Grades II and III carry higher recurrence risk and require closer monitoring.',
  },
  no_tumor: {
    label: 'No Tumor Detected',
    color: 'green',
    desc: 'No signs of a brain tumor were found in this MRI scan. Continue monitoring symptoms and consult a specialist if concerns persist.',
    overview: 'The model found no evidence of a tumor in this scan. This is a reassuring result, but a clean scan does not replace professional medical diagnosis. If you are experiencing symptoms, consult a neurologist.',
    symptoms: [
      'Persistent headaches should still be evaluated by a doctor',
      'Dizziness, vision changes, or seizures warrant professional review',
      'Cognitive changes can have many non-tumor causes',
      'Regular check-ups are recommended for at-risk individuals',
    ],
    treatments: [
      'No tumor-specific treatment needed based on this scan',
      'Address any underlying causes of symptoms with your doctor',
      'Maintain routine annual health check-ups',
    ],
    precautions: [
      'Do not rely solely on this AI tool — confirm with a radiologist',
      'Maintain regular neurological check-ups if symptoms exist',
      'Live a brain-healthy lifestyle: exercise, sleep, balanced diet',
      'Manage stress and blood pressure — key factors in brain health',
      'Avoid smoking and excessive alcohol consumption',
      'Stay mentally active — reading, puzzles, and learning new skills',
    ],
    prognosis: 'No evidence of tumor. Continue healthy habits and routine medical care. Follow up with a healthcare professional if symptoms persist or worsen.',
  },
  pituitary_tumor: {
    label: 'Pituitary Tumor',
    color: 'purple',
    desc: 'A tumor found in the pituitary gland at the base of the brain. Most are benign adenomas, but they can disrupt hormone regulation.',
    overview: 'Pituitary adenomas account for ~15% of all brain tumors and are almost always benign. They can be "functioning" (secreting excess hormones) or "non-functioning." Hormone-active tumors can cause systemic endocrine disorders even when small.',
    symptoms: [
      'Visual field defects — classically bitemporal hemianopia (tunnel vision)',
      'Persistent headaches centered behind the eyes',
      'Hormonal imbalances: irregular periods, infertility, low libido',
      'Unexplained weight gain and easy bruising (Cushing\'s disease)',
      'Abnormal growth of hands/feet/face (acromegaly from excess GH)',
      'Fatigue, low energy, and cold intolerance (hypopituitarism)',
    ],
    treatments: [
      'Medication — dopamine agonists (cabergoline, bromocriptine) for prolactinomas',
      'Transsphenoidal surgery — minimally invasive through the nasal passage',
      'Stereotactic radiosurgery (Gamma Knife) for residual or recurrent tumor',
      'Hormone replacement therapy to correct deficiencies',
      'Regular endocrine monitoring to adjust treatment',
    ],
    precautions: [
      'Get regular vision (ophthalmology) and hormonal blood tests',
      'Take prescribed hormonal medications consistently',
      'Inform all doctors about pituitary disease — it affects many body systems',
      'Wear a medical ID if you have adrenal insufficiency (risk of crisis)',
      'Avoid abrupt steroid withdrawal — taper under medical guidance',
      'Monitor bone density — hormonal changes can cause osteoporosis',
    ],
    prognosis: 'Prognosis is generally very good. Most pituitary adenomas are benign and highly treatable. Functioning tumors may require long-term medical management. Recurrence is possible but manageable with regular follow-up.',
  },
};

const BAR_COLORS = {
  red:    'bg-red-500',
  orange: 'bg-orange-500',
  green:  'bg-green-500',
  purple: 'bg-purple-500',
};

const TEXT_COLORS = {
  red:    'text-red-400',
  orange: 'text-orange-400',
  green:  'text-green-400',
  purple: 'text-purple-400',
};

const BORDER_COLORS = {
  red:    'border-red-500/30 bg-red-500/5',
  orange: 'border-orange-500/30 bg-orange-500/5',
  green:  'border-green-500/30 bg-green-500/5',
  purple: 'border-purple-500/30 bg-purple-500/5',
};

const ICON_COLOR = {
  red:    'text-red-400',
  orange: 'text-orange-400',
  green:  'text-green-400',
  purple: 'text-purple-400',
};

const SECTION_BG = {
  red:    'bg-red-500/10 border-red-500/20',
  orange: 'bg-orange-500/10 border-orange-500/20',
  green:  'bg-green-500/10 border-green-500/20',
  purple: 'bg-purple-500/10 border-purple-500/20',
};

function InfoSection({ icon: Icon, title, items, color, isList = true }) {
  return (
    <div className={`rounded-2xl border p-5 ${SECTION_BG[color]}`}>
      <h4 className={`text-sm font-bold uppercase tracking-widest mb-3 flex items-center gap-2 ${ICON_COLOR[color]}`}>
        <Icon className="w-4 h-4" />
        {title}
      </h4>
      {isList ? (
        <ul className="space-y-2">
          {items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
              <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${ICON_COLOR[color].replace('text-', 'bg-')}`} />
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-gray-300 leading-relaxed">{items}</p>
      )}
    </div>
  );
}

function TumorInfoCard({ meta }) {
  const { color, label, overview, symptoms, treatments, precautions, prognosis } = meta;
  return (
    <div className={`glass rounded-3xl p-8 animate-fade-in border ${BORDER_COLORS[color]}`}>
      <div className="flex items-center gap-3 mb-2">
        <div className={`p-2 rounded-xl ${SECTION_BG[color]}`}>
          <Brain className={`w-5 h-5 ${ICON_COLOR[color]}`} />
        </div>
        <h2 className="text-xl font-semibold text-white">About {label}</h2>
      </div>
      <p className="text-gray-400 text-sm mb-6 leading-relaxed">{overview}</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoSection icon={Activity}     title="Common Symptoms"   items={symptoms}    color={color} />
        <InfoSection icon={Pill}         title="Treatment Options" items={treatments}   color={color} />
        <InfoSection icon={ShieldAlert}  title="Precautions"       items={precautions}  color={color} />
        <InfoSection icon={HeartPulse}   title="Prognosis"         items={prognosis}    color={color} isList={false} />
      </div>

      <div className={`mt-5 rounded-xl p-4 flex items-start gap-3 ${SECTION_BG[color]} border`}>
        <Stethoscope className={`w-4 h-4 shrink-0 mt-0.5 ${ICON_COLOR[color]}`} />
        <p className="text-xs text-gray-400 leading-relaxed">
          <span className="font-semibold text-gray-300">Always consult a specialist.</span>{' '}
          This information is educational and does not replace a diagnosis or treatment plan from a qualified neurologist or oncologist.
        </p>
      </div>
    </div>
  );
}

function ProbabilityBar({ label, value, color, isTop }) {
  return (
    <div className={`py-2 px-3 rounded-xl transition-all ${isTop ? 'bg-white/5' : ''}`}>
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm font-medium ${isTop ? 'text-white' : 'text-gray-400'}`}>{label}</span>
        <span className={`text-sm font-bold tabular-nums ${isTop ? TEXT_COLORS[color] : 'text-gray-500'}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${isTop ? BAR_COLORS[color] : 'bg-gray-600'}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
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
      setShowHeatmap(false);
    }
  };

  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };

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
      setShowHeatmap(false);
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
    setShowHeatmap(false);
  };

  const meta = result ? CLASS_META[result.class] : null;
  const accentColor = meta?.color ?? 'green';

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
          Upload an MRI scan to detect and classify brain tumors using deep learning, with Grad-CAM visualization.
        </p>
      </div>

      {/* Main Content */}
      <main className="max-w-5xl w-full space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Upload Section */}
          <div className="glass rounded-3xl p-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
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
                  <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</>
                ) : (
                  <><CheckCircle2 className="w-5 h-5" /> Classify Image</>
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
              <div className={`glass rounded-3xl p-8 animate-fade-in flex-1 border ${BORDER_COLORS[accentColor]}`}>
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold flex items-center gap-2">
                    <CheckCircle2 className={`w-5 h-5 ${TEXT_COLORS[accentColor]}`} />
                    Classification Result
                  </h2>
                  <span className={`px-3 py-1 text-xs font-bold rounded-full uppercase tracking-wider ${TEXT_COLORS[accentColor]} bg-white/5`}>
                    Complete
                  </span>
                </div>

                {/* Diagnosis */}
                <div className="mb-6">
                  <label className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1 block">
                    Diagnosis
                  </label>
                  <p className={`text-3xl font-bold ${TEXT_COLORS[accentColor]}`}>
                    {meta.label}
                  </p>
                  <p className="text-gray-500 text-sm mt-1">{meta.desc}</p>
                </div>

                {/* Top confidence */}
                <div className="mb-6">
                  <label className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-1 block">
                    Confidence
                  </label>
                  <p className={`text-5xl font-black ${TEXT_COLORS[accentColor]}`}>
                    {result.confidence}%
                  </p>
                </div>

                {/* All class probabilities */}
                {result.all_probabilities && (
                  <div>
                    <label className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 block">
                      All Probabilities
                    </label>
                    <div className="space-y-1">
                      {Object.entries(result.all_probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([cls, prob]) => (
                          <ProbabilityBar
                            key={cls}
                            label={CLASS_META[cls]?.label ?? cls}
                            value={prob}
                            color={CLASS_META[cls]?.color ?? 'green'}
                            isTop={cls === result.class}
                          />
                        ))}
                    </div>
                  </div>
                )}
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
                  Analyzing with neural network and generating Grad-CAM visualization...
                </p>
              </div>
            ) : null}
          </div>
        </div>

        {/* Tumor Info Card */}
        {result && meta && (
          <TumorInfoCard meta={meta} />
        )}

        {/* Grad-CAM Heatmap Section */}
        {result?.heatmap && (
          <div className="glass rounded-3xl p-8 animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <Microscope className="w-5 h-5 text-primary-400" />
                Grad-CAM Visualization
              </h2>
              <button
                onClick={() => setShowHeatmap(v => !v)}
                className="text-sm px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-all"
              >
                {showHeatmap ? 'Show Original' : 'Show Heatmap'}
              </button>
            </div>
            <p className="text-gray-500 text-sm mb-6">
              Gradient-weighted Class Activation Mapping highlights the regions of the MRI scan most influential to the model's prediction. Warmer colors (red/yellow) indicate higher importance.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Original Scan</p>
                <img
                  src={previewUrl}
                  alt="Original MRI"
                  className="w-full rounded-2xl object-contain bg-black max-h-64"
                />
              </div>
              <div>
                <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Grad-CAM Overlay</p>
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="Grad-CAM Heatmap"
                  className="w-full rounded-2xl object-contain bg-black max-h-64"
                />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer / Disclaimer */}
      <footer className="mt-12 max-w-5xl w-full">
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
