import { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Loader2, Radio, Clock, CheckCircle, XCircle, AlertTriangle, Package, MapPin, Hash } from 'lucide-react';

export type VoiceState = 'listening' | 'processing' | 'idle' | 'error';

function App() {
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');
  const [currentCommand, setCurrentCommand] = useState<string>('');
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error' | 'warning'; message: string } | null>(null);
  const [currentTask, setCurrentTask] = useState<string>('READY FOR INPUT');
  const [isConnected, setIsConnected] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [confidence, setConfidence] = useState(0);
  
  const [transcriptionHistory, setTranscriptionHistory] = useState<Array<{
    id: string;
    text: string;
    confidence: number;
    timestamp: string;
    productCode?: string;
    quantity?: number;
    location?: string;
  }>>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const API_URL = 'http://127.0.0.1:8000';
  const WS_URL = 'ws://127.0.0.1:8000/ws/voice-input';

  // Connect WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message:', data);
          
          // Only process result messages
          if (data.type === 'result' || data.type === 'error') {
            handleWebSocketMessage(data);
          } else if (data.type === 'connected') {
            console.log('Connected to server:', data.message);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleWebSocketMessage = (data: any) => {
    console.log('Processing result:', data);
    
    setVoiceState('idle');
    setCurrentTask('READY FOR INPUT');

    if (data.text) {
      setTranscription(data.text);
      setCurrentCommand(data.text);
      console.log('Transcription:', data.text);
    }

    if (data.confidence !== undefined) {
      setConfidence(Math.round(data.confidence * 100));
    }

    // Add to transcription history with extracted fields
    if (data.text) {
      // Extract fields from data or parse from text
      let productCode = data.data?.product_id || data.found_fields?.product_id;
      let quantity = data.data?.quantity || data.found_fields?.quantity;
      let location = data.data?.location || data.found_fields?.location;
      
      const text = data.text;
      console.log('Parsing text:', text);
      
      // If not in response, try to parse from text with flexible patterns
      if (!productCode) {
        // Match product codes like ABC-123, ABC-1234, AB-123
        const productMatch = text.match(/\b([A-Z]{2,3}[-\s]?\d{2,5})\b/i);
        if (productMatch) {
          productCode = productMatch[1].toUpperCase().replace(/\s+/g, '-');
          console.log('Found product code:', productCode);
        }
      }
      if (!quantity) {
        // Match "quantity 100" or "quantity, 100" or just standalone numbers
        const qtyMatch = text.match(/(?:quantity|qty|count)[,:\s]*(\d+)/i) || 
                        text.match(/(\d+)\s*(?:units?|pieces?|items?|pcs)/i) ||
                        text.match(/\b(\d{1,4})\b(?![-\/])/);
        if (qtyMatch) {
          quantity = parseInt(qtyMatch[1]);
          console.log('Found quantity:', quantity);
        }
      }
      if (!location) {
        // Match "location A-9" or "A-12" or "aisle B-5"
        const locMatch = text.match(/(?:location|loc|aisle|at|section)[,:\s]*([A-Z][-\s]?\d{1,2}(?:[-\s]?\d{1,2})?)/i) ||
                        text.match(/\b([A-Z][-]\d{1,2}(?:[-]\d{1,2})?)\b/i);
        if (locMatch) {
          location = locMatch[1].toUpperCase().replace(/\s+/g, '-');
          console.log('Found location:', location);
        }
      }
      
      const newEntry = {
        id: Date.now().toString(),
        text: data.text,
        confidence: data.confidence ? Math.round(data.confidence * 100) : 0,
        timestamp: new Date().toLocaleTimeString(),
        productCode,
        quantity,
        location
      };
      setTranscriptionHistory(prev => [newEntry, ...prev].slice(0, 20));
    }

    // Check various status values the backend might send
    const status = data.status;
    if (status === 'success' || status === 'saved' || status === 'complete') {
      setFeedback({ type: 'success', message: data.message || 'ENTRY SAVED SUCCESSFULLY' });
    } else if (status === 'error') {
      setFeedback({ type: 'error', message: data.message || 'PROCESSING ERROR' });
    } else if (status === 'partial' || status === 'incomplete' || status === 'low_confidence') {
      setFeedback({ type: 'warning', message: `LOW CONFIDENCE (${Math.round((data.confidence || 0) * 100)}%)` });
    } else if (data.text) {
      setFeedback({ type: 'success', message: 'TRANSCRIPTION COMPLETE' });
    }

    setTimeout(() => {
      setFeedback(null);
      setCurrentCommand('');
    }, 4000);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setVoiceState('listening');
      setCurrentTask('LISTENING...');
      setTranscription('');
    } catch (error) {
      setFeedback({ type: 'error', message: 'MICROPHONE ACCESS DENIED' });
      setVoiceState('error');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && voiceState === 'listening') {
      mediaRecorderRef.current.stop();
      setVoiceState('processing');
      setCurrentTask('PROCESSING AUDIO...');
    }
  };

  const toggleRecording = () => {
    if (voiceState === 'listening') {
      stopRecording();
    } else if (voiceState === 'idle' || voiceState === 'error') {
      startRecording();
    }
  };

  const sendAudio = async (audioBlob: Blob) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        wsRef.current?.send(JSON.stringify({
          type: 'audio',
          audio: base64,
          format: 'webm'
        }));
      };
      reader.readAsDataURL(audioBlob);
    } else {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      
      try {
        const response = await fetch(`${API_URL}/api/process-audio`, {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        handleWebSocketMessage(data);
      } catch (error) {
        setFeedback({ type: 'error', message: 'FAILED TO PROCESS AUDIO' });
        setVoiceState('idle');
        setCurrentTask('READY FOR INPUT');
      }
    }
  };

  const getStatusConfig = () => {
    switch (voiceState) {
      case 'listening':
        return { icon: Mic, label: 'LISTENING', color: 'bg-green-500', borderColor: 'border-green-500', textColor: 'text-green-500', pulse: true };
      case 'processing':
        return { icon: Loader2, label: 'PROCESSING', color: 'bg-yellow-500', borderColor: 'border-yellow-500', textColor: 'text-yellow-500', pulse: false };
      case 'error':
        return { icon: XCircle, label: 'ERROR', color: 'bg-red-500', borderColor: 'border-red-500', textColor: 'text-red-500', pulse: true };
      default:
        return { icon: MicOff, label: 'STANDBY', color: 'bg-neutral-600', borderColor: 'border-neutral-600', textColor: 'text-neutral-400', pulse: false };
    }
  };

  const statusConfig = getStatusConfig();
  const StatusIcon = statusConfig.icon;

  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <header className="border-b-4 border-yellow-500 pb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-yellow-500 flex items-center justify-center">
                <Radio className="w-10 h-10 text-neutral-900" strokeWidth={2.5} />
              </div>
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-neutral-100">VOICEVAULT</h1>
                <p className="text-lg text-neutral-400 font-medium tracking-wide">VOICE INVENTORY CONTROL SYSTEM</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-neutral-500 font-medium tracking-wide">SECTOR</div>
              <div className="text-3xl font-bold text-yellow-500">WAREHOUSE-01</div>
              <div className={`text-sm mt-1 ${isConnected ? 'text-green-500' : 'text-red-500'}`}>
                ● {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
              </div>
            </div>
          </div>
        </header>

        {/* Voice Status - Clickable */}
        <div 
          onClick={toggleRecording}
          className={`bg-neutral-800 border-4 ${statusConfig.borderColor} p-8 cursor-pointer transition-all hover:bg-neutral-750 ${statusConfig.pulse ? 'animate-pulse' : ''}`}
        >
          <div className="flex items-center gap-6">
            <div className={`w-20 h-20 ${statusConfig.color} flex items-center justify-center`}>
              <StatusIcon className={`w-12 h-12 text-neutral-900 ${voiceState === 'processing' ? 'animate-spin' : ''}`} strokeWidth={2.5} />
            </div>
            <div className="flex-1">
              <div className="text-sm text-neutral-500 font-bold tracking-wider mb-1">VOICE STATUS</div>
              <div className={`text-4xl font-bold ${statusConfig.textColor} tracking-tight`}>{statusConfig.label}</div>
              {currentCommand && (
                <div className="mt-3 text-xl text-neutral-300 font-mono tracking-wide">"{currentCommand}"</div>
              )}
              {confidence > 0 && (
                <div className="mt-2 text-sm text-neutral-500">CONFIDENCE: {confidence}%</div>
              )}
            </div>
            <div className="text-neutral-500 text-sm">
              {voiceState === 'idle' ? 'CLICK TO RECORD' : voiceState === 'listening' ? 'CLICK TO STOP' : ''}
            </div>
          </div>
        </div>

        {/* Task Display */}
        <div className="bg-neutral-950 border-2 border-yellow-500 p-6">
          <div className="text-sm text-neutral-500 font-bold tracking-wider mb-2">CURRENT TASK</div>
          <div className="text-5xl font-bold text-yellow-500 tracking-tight font-mono">{currentTask}</div>
        </div>

        {/* Command Feedback */}
        {feedback && (
          <div className={`bg-neutral-800 border-4 p-6 ${
            feedback.type === 'success' ? 'border-green-500' : 
            feedback.type === 'error' ? 'border-red-500' : 'border-yellow-500'
          }`}>
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 flex items-center justify-center flex-shrink-0 ${
                feedback.type === 'success' ? 'bg-green-500' : 
                feedback.type === 'error' ? 'bg-red-500' : 'bg-yellow-500'
              }`}>
                {feedback.type === 'success' ? <CheckCircle className="w-10 h-10 text-neutral-900" strokeWidth={2.5} /> :
                 feedback.type === 'error' ? <XCircle className="w-10 h-10 text-neutral-900" strokeWidth={2.5} /> :
                 <AlertTriangle className="w-10 h-10 text-neutral-900" strokeWidth={2.5} />}
              </div>
              <div className={`text-3xl font-bold tracking-tight ${
                feedback.type === 'success' ? 'text-green-500' : 
                feedback.type === 'error' ? 'text-red-500' : 'text-yellow-500'
              }`}>
                {feedback.message}
              </div>
            </div>
          </div>
        )}

        {/* Transcription History */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-neutral-300 tracking-wide">TRANSCRIPTION HISTORY</h2>
            <div className="text-neutral-500 font-mono text-lg">{transcriptionHistory.length} ENTRIES</div>
          </div>
          
          {transcriptionHistory.length === 0 ? (
            <div className="bg-neutral-800 border-2 border-neutral-700 p-8 text-center">
              <Mic className="w-16 h-16 text-neutral-600 mx-auto mb-4" />
              <div className="text-2xl text-neutral-500 font-medium">NO TRANSCRIPTIONS YET</div>
              <div className="text-neutral-600 mt-2">Click the voice status panel above to start recording</div>
            </div>
          ) : (
            <div className="space-y-3">
              {transcriptionHistory.map((entry) => (
                <div key={entry.id} className={`bg-neutral-800 border-l-8 p-6 ${
                  entry.confidence >= 80 ? 'border-green-500' : 
                  entry.confidence >= 50 ? 'border-yellow-500' : 'border-red-500'
                }`}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="text-lg text-neutral-400 italic mb-3">"{entry.text}"</div>
                    </div>
                    <div className={`px-4 py-2 flex-shrink-0 ml-4 ${
                      entry.confidence >= 80 ? 'bg-green-500' : 
                      entry.confidence >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}>
                      <div className="text-neutral-900 font-bold text-sm tracking-wide">{entry.confidence}%</div>
                    </div>
                  </div>
                  
                  {/* Extracted Fields */}
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-neutral-900 p-4 border-l-4 border-blue-500">
                      <div className="flex items-center gap-2 text-neutral-500 text-sm mb-1">
                        <Hash className="w-4 h-4" />
                        <span className="font-medium">PRODUCT CODE</span>
                      </div>
                      <div className={`text-2xl font-bold font-mono ${entry.productCode ? 'text-blue-400' : 'text-neutral-600'}`}>
                        {entry.productCode || '—'}
                      </div>
                    </div>
                    <div className="bg-neutral-900 p-4 border-l-4 border-green-500">
                      <div className="flex items-center gap-2 text-neutral-500 text-sm mb-1">
                        <Package className="w-4 h-4" />
                        <span className="font-medium">QUANTITY</span>
                      </div>
                      <div className={`text-2xl font-bold font-mono ${entry.quantity ? 'text-green-400' : 'text-neutral-600'}`}>
                        {entry.quantity || '—'}
                      </div>
                    </div>
                    <div className="bg-neutral-900 p-4 border-l-4 border-yellow-500">
                      <div className="flex items-center gap-2 text-neutral-500 text-sm mb-1">
                        <MapPin className="w-4 h-4" />
                        <span className="font-medium">LOCATION</span>
                      </div>
                      <div className={`text-2xl font-bold font-mono ${entry.location ? 'text-yellow-400' : 'text-neutral-600'}`}>
                        {entry.location || '—'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4 text-neutral-500 text-sm">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>{entry.timestamp}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Voice Commands Reference */}
        <div className="bg-neutral-800 border-2 border-neutral-700 p-6">
          <h3 className="text-xl font-bold text-neutral-300 mb-4 tracking-wide">AVAILABLE VOICE COMMANDS</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-neutral-900 p-4 border-l-4 border-blue-500">
              <div className="text-sm text-neutral-500 font-medium mb-1">CHECK</div>
              <div className="text-neutral-200 font-mono">"CHECK INVENTORY [SKU]"</div>
            </div>
            <div className="bg-neutral-900 p-4 border-l-4 border-green-500">
              <div className="text-sm text-neutral-500 font-medium mb-1">UPDATE</div>
              <div className="text-neutral-200 font-mono">"PRODUCT [CODE] QTY [NUM] LOC [LOC]"</div>
            </div>
            <div className="bg-neutral-900 p-4 border-l-4 border-yellow-500">
              <div className="text-sm text-neutral-500 font-medium mb-1">LOCATE</div>
              <div className="text-neutral-200 font-mono">"LOCATE [SKU]"</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
