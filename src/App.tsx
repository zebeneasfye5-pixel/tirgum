import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftRight,
  Check,
  ChevronDown,
  Copy,
  Globe,
  Loader2,
  Mic,
  MicOff,
  Search,
  Send,
  Square,
  Trash2,
  Volume2,
  VolumeX,
  Download,
} from "lucide-react";
import { UI_COPY, UiLang } from "./data/i18n";
import {
  getLanguageByCode,
  VOICE_PROFILES,
  VoiceId,
  WORLD_LANGUAGES,
} from "./data/languages";
import {
  blobToBase64,
  downsampleTo16k,
  float32ToPcm16Base64,
  GaplessPcmPlayer,
  playWavBase64,
} from "./utils/audio";
import { WaveformCanvas } from "./components/WaveformCanvas";
import { LanguagePickerModal } from "./components/LanguagePickerModal";
import { RemoteRoomView } from "./components/RemoteRoomView";
import { OfflineIndicator, PWAInstallButton } from "./components/PWAInstallButton";

type StudioMode = "face" | "room";
type EngineMode = "live" | "turn";

interface TranscriptEntry {
  id: string;
  sourceLang: string;
  targetLang: string;
  sourceText: string;
  translatedText: string;
  audioWavBase64?: string | null;
  timestamp: number;
}

const ATLAS_REGIONS = ["All", "Africa", "Europe", "Asia", "Middle East", "Americas"] as const;

const QUICK_PAIRS: Array<{ from: string; to: string; label: string }> = [
  { from: "am", to: "en", label: "አማርኛ ⇄ English" },
  { from: "am", to: "zh", label: "አማርኛ ⇄ 中文" },
  { from: "am", to: "ar", label: "አማርኛ ⇄ العربية" },
  { from: "am", to: "fr", label: "አማርኛ ⇄ Français" },
  { from: "en", to: "es", label: "English ⇄ Español" },
];

export default function App() {
  const initialRoomParam = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return (params.get("room") || "").trim().toUpperCase();
  }, []);

  const [uiLang, setUiLang] = useState<UiLang>(() => {
    const saved = localStorage.getItem("lissan_ui_lang");
    return saved === "en" || saved === "am" ? saved : "am";
  });
  const [studioMode, setStudioMode] = useState<StudioMode>(
    initialRoomParam ? "room" : "face"
  );
  const [engineMode, setEngineMode] = useState<EngineMode>("live");
  const [bidirectional, setBidirectional] = useState<boolean>(true);

  const [sourceLang, setSourceLang] = useState<string>(() => {
    return localStorage.getItem("lissan_source_lang") || "am";
  });
  const [targetLang, setTargetLang] = useState<string>(() => {
    return localStorage.getItem("lissan_target_lang") || "en";
  });
  const [voiceName, setVoiceName] = useState<VoiceId>(() => {
    return (localStorage.getItem("lissan_voice") as VoiceId) || "Zephyr";
  });
  const [autoPlay, setAutoPlay] = useState<boolean>(true);

  const [pickerTarget, setPickerTarget] = useState<"source" | "target" | null>(null);

  // Live continuous state
  const [liveStatus, setLiveStatus] = useState<"idle" | "connecting" | "active">("idle");
  const [liveSeconds, setLiveSeconds] = useState<number>(0);
  const [liveInputText, setLiveInputText] = useState<string>("");
  const [liveOutputText, setLiveOutputText] = useState<string>("");
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);
  const [outputAnalyser, setOutputAnalyser] = useState<AnalyserNode | null>(null);

  // Turn-by-turn state
  const [turnRecording, setTurnRecording] = useState<boolean>(false);
  const [turnProcessing, setTurnProcessing] = useState<boolean>(false);
  const [lastTurnAudio, setLastTurnAudio] = useState<string | null>(null);

  // Shared UI state
  const [phraseInput, setPhraseInput] = useState<string>("");
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  const [history, setHistory] = useState<TranscriptEntry[]>(() => {
    try {
      const raw = localStorage.getItem("lissan_history");
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Language Atlas filters
  const [atlasQuery, setAtlasQuery] = useState<string>("");
  const [atlasRegion, setAtlasRegion] = useState<(typeof ATLAS_REGIONS)[number]>("All");

  // Refs for WebAudio / WebSocket callbacks (avoid stale closures)
  const liveWsRef = useRef<WebSocket | null>(null);
  const inputAudioCtxRef = useRef<AudioContext | null>(null);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const playerRef = useRef<GaplessPcmPlayer | null>(null);
  const liveInputAccRef = useRef<string>("");
  const liveOutputAccRef = useRef<string>("");
  const autoPlayRef = useRef<boolean>(autoPlay);
  const sourceLangRef = useRef<string>(sourceLang);
  const targetLangRef = useRef<string>(targetLang);

  // Turn recorder refs
  const turnRecorderRef = useRef<MediaRecorder | null>(null);
  const turnStreamRef = useRef<MediaStream | null>(null);
  const turnCtxRef = useRef<AudioContext | null>(null);
  const turnChunksRef = useRef<Blob[]>([]);

  const t = UI_COPY[uiLang];

  useEffect(() => {
    localStorage.setItem("lissan_ui_lang", uiLang);
  }, [uiLang]);

  useEffect(() => {
    localStorage.setItem("lissan_source_lang", sourceLang);
  }, [sourceLang]);

  useEffect(() => {
    localStorage.setItem("lissan_target_lang", targetLang);
  }, [targetLang]);

  useEffect(() => {
    localStorage.setItem("lissan_voice", voiceName);
  }, [voiceName]);

  useEffect(() => {
    try {
      const serializable = history.slice(0, 50).map(({ audioWavBase64: _a, ...rest }) => rest);
      localStorage.setItem("lissan_history", JSON.stringify(serializable));
    } catch {
      // ignore storage quota errors
    }
  }, [history]);

  useEffect(() => {
    if (liveStatus !== "active") {
      setLiveSeconds(0);
      return;
    }
    const id = setInterval(() => {
      setLiveSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(id);
  }, [liveStatus]);

  useEffect(() => {
    autoPlayRef.current = autoPlay;
    playerRef.current?.setMuted(!autoPlay);
  }, [autoPlay]);

  useEffect(() => {
    sourceLangRef.current = sourceLang;
  }, [sourceLang]);

  useEffect(() => {
    targetLangRef.current = targetLang;
  }, [targetLang]);

  const stopLiveSession = useCallback(() => {
    const finalIn = liveInputAccRef.current.trim();
    const finalOut = liveOutputAccRef.current.trim();
    if (finalIn || finalOut) {
      setHistory((prev) => [
        {
          id: `live_${Date.now()}`,
          sourceLang: sourceLangRef.current,
          targetLang: targetLangRef.current,
          sourceText: finalIn,
          translatedText: finalOut,
          timestamp: Date.now(),
        },
        ...prev,
      ]);
      liveInputAccRef.current = "";
      liveOutputAccRef.current = "";
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    if (inputStreamRef.current) {
      inputStreamRef.current.getTracks().forEach((track) => track.stop());
      inputStreamRef.current = null;
    }
    if (inputAudioCtxRef.current) {
      void inputAudioCtxRef.current.close();
      inputAudioCtxRef.current = null;
    }
    if (liveWsRef.current) {
      liveWsRef.current.close();
      liveWsRef.current = null;
    }
    if (playerRef.current) {
      playerRef.current.close();
      playerRef.current = null;
    }
    setInputAnalyser(null);
    setOutputAnalyser(null);
    setLiveStatus("idle");
  }, []);

  useEffect(() => {
    return () => {
      stopLiveSession();
    };
  }, [stopLiveSession]);

  const startLiveSession = useCallback(async () => {
    setErrorBanner(null);
    setLiveStatus("connecting");
    setLiveInputText("");
    setLiveOutputText("");
    liveInputAccRef.current = "";
    liveOutputAccRef.current = "";

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      inputStreamRef.current = stream;

      const player = new GaplessPcmPlayer();
      player.setMuted(!autoPlayRef.current);
      const { analyser: outAnal } = player.ensureContext();
      playerRef.current = player;
      setOutputAnalyser(outAnal);

      const inputCtx = new AudioContext({ sampleRate: 16000 });
      if (inputCtx.state === "suspended") {
        await inputCtx.resume();
      }
      inputAudioCtxRef.current = inputCtx;

      const sourceNode = inputCtx.createMediaStreamSource(stream);
      const inAnal = inputCtx.createAnalyser();
      inAnal.fftSize = 256;
      sourceNode.connect(inAnal);
      setInputAnalyser(inAnal);

      const processor = inputCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // Silent sink so ScriptProcessorNode fires without microphone loopback echo
      const silentGain = inputCtx.createGain();
      silentGain.gain.value = 0;
      sourceNode.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(inputCtx.destination);

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const params = new URLSearchParams({
        sourceLang,
        targetLang,
        voiceName,
        bidirectional: String(bidirectional && sourceLang !== "auto"),
      });
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live?${params.toString()}`);
      liveWsRef.current = ws;

      ws.onopen = () => {
        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const rawChannel = e.inputBuffer.getChannelData(0);
          const pcm16k = downsampleTo16k(rawChannel, inputCtx.sampleRate);
          const base64 = float32ToPcm16Base64(pcm16k);
          ws.send(JSON.stringify({ type: "audio", data: base64 }));
        };
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "status") {
            if (msg.connected) {
              setLiveStatus("active");
            } else {
              stopLiveSession();
            }
          } else if (msg.type === "inputTranscript" && msg.text) {
            liveInputAccRef.current += msg.text;
            setLiveInputText(liveInputAccRef.current);
          } else if (msg.type === "outputTranscript" && msg.text) {
            liveOutputAccRef.current += msg.text;
            setLiveOutputText(liveOutputAccRef.current);
          } else if (msg.type === "audio" && msg.data) {
            playerRef.current?.enqueuePcm24kBase64(msg.data);
          } else if (msg.type === "interrupted") {
            playerRef.current?.interrupt();
          } else if (msg.type === "turnComplete") {
            const committedIn = liveInputAccRef.current.trim();
            const committedOut = liveOutputAccRef.current.trim();
            if (committedIn || committedOut) {
              setHistory((prev) => [
                {
                  id: `live_${Date.now()}`,
                  sourceLang: sourceLangRef.current,
                  targetLang: targetLangRef.current,
                  sourceText: committedIn,
                  translatedText: committedOut,
                  timestamp: Date.now(),
                },
                ...prev,
              ]);
              liveInputAccRef.current = "";
              liveOutputAccRef.current = "";
            }
          } else if (msg.type === "error") {
            setErrorBanner(msg.message || t.errorGeneric);
            stopLiveSession();
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        setErrorBanner(t.errorGeneric);
        stopLiveSession();
      };

      ws.onclose = () => {
        if (liveWsRef.current === ws) {
          stopLiveSession();
        }
      };
    } catch {
      setErrorBanner(t.errorMicDenied);
      stopLiveSession();
    }
  }, [
    bidirectional,
    sourceLang,
    targetLang,
    voiceName,
    stopLiveSession,
    t.errorGeneric,
    t.errorMicDenied,
  ]);

  const executeTurnTranslation = useCallback(
    async (payload: { audioBase64?: string; mimeType?: string; text?: string }) => {
      setTurnProcessing(true);
      setErrorBanner(null);
      try {
        const res = await fetch("/api/translate-turn", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            sourceLang,
            targetLang,
            voiceName,
            bidirectional: bidirectional && sourceLang !== "auto",
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || t.errorGeneric);
        }

        if (data.sourceTranscript || data.translatedText) {
          setLiveInputText(data.sourceTranscript || "");
          setLiveOutputText(data.translatedText || "");
          setLastTurnAudio(data.audioWavBase64 || null);

          const resolvedSrc =
            data.resolvedSourceLang ||
            (sourceLang === "auto" ? data.detectedLanguage || "auto" : sourceLang);
          const resolvedTgt = data.resolvedTargetLang || targetLang;

          setHistory((prev) => [
            {
              id: `turn_${Date.now()}`,
              sourceLang: resolvedSrc,
              targetLang: resolvedTgt,
              sourceText: data.sourceTranscript || "",
              translatedText: data.translatedText || "",
              audioWavBase64: data.audioWavBase64 || null,
              timestamp: Date.now(),
            },
            ...prev,
          ]);

          if (autoPlay && data.audioWavBase64) {
            void playWavBase64(data.audioWavBase64);
          }
        }
      } catch (err) {
        setErrorBanner(err instanceof Error ? err.message : t.errorGeneric);
      } finally {
        setTurnProcessing(false);
      }
    },
    [autoPlay, bidirectional, sourceLang, targetLang, voiceName, t.errorGeneric]
  );

  const startTurnRecording = async () => {
    setErrorBanner(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      turnStreamRef.current = stream;

      const ctx = new AudioContext();
      turnCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const anal = ctx.createAnalyser();
      anal.fftSize = 256;
      src.connect(anal);
      setInputAnalyser(anal);

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      turnChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) turnChunksRef.current.push(ev.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(turnChunksRef.current, { type: mimeType });
        turnStreamRef.current?.getTracks().forEach((tr) => tr.stop());
        void turnCtxRef.current?.close();
        setInputAnalyser(null);

        if (blob.size < 400) return;
        const base64 = await blobToBase64(blob);
        await executeTurnTranslation({
          audioBase64: base64,
          mimeType: mimeType.split(";")[0],
        });
      };

      turnRecorderRef.current = recorder;
      recorder.start();
      setTurnRecording(true);
    } catch {
      setErrorBanner(t.errorMicDenied);
    }
  };

  const stopTurnRecording = () => {
    if (turnRecorderRef.current && turnRecorderRef.current.state !== "inactive") {
      turnRecorderRef.current.stop();
    }
    setTurnRecording(false);
  };

  const handleSendPhrase = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = phraseInput.trim();
    if (!cleaned || turnProcessing) return;
    setPhraseInput("");
    await executeTurnTranslation({ text: cleaned });
  };

  const handleSwapLanguages = () => {
    if (liveStatus !== "idle") {
      stopLiveSession();
    }
    const nextSource = targetLang;
    const nextTarget = sourceLang === "auto" ? "en" : sourceLang;
    setSourceLang(nextSource);
    setTargetLang(nextTarget);
  };

  const handleEngineSwitch = (next: EngineMode) => {
    if (next === engineMode) return;
    if (liveStatus !== "idle") stopLiveSession();
    if (turnRecording) stopTurnRecording();
    setEngineMode(next);
  };

  const handleModeSwitch = (next: StudioMode) => {
    if (next === studioMode) return;
    if (liveStatus !== "idle") stopLiveSession();
    if (turnRecording) stopTurnRecording();
    setStudioMode(next);
  };

  const handleLanguagePick = (code: string) => {
    if (liveStatus !== "idle") {
      stopLiveSession();
    }
    if (pickerTarget === "source") {
      setSourceLang(code);
    } else if (pickerTarget === "target") {
      setTargetLang(code);
    }
  };

  const handleCopyText = async (id: string, text: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const formatDuration = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const rem = sec % 60;
    return `${String(mins).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  };

  const exportHistory = () => {
    if (history.length === 0) return;
    const lines = [...history].reverse().map((item) => {
      const timeStr = new Date(item.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      return `[${timeStr}] ${item.sourceLang} -> ${item.targetLang}\n  ${item.translatedText}\n  (${item.sourceText})`;
    });
    const blob = new Blob(
      [`Lissan Transcript\n${"=".repeat(28)}\n\n${lines.join("\n\n")}\n`],
      { type: "text/plain;charset=utf-8" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lissan-transcript-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredAtlas = useMemo(() => {
    const q = atlasQuery.trim().toLowerCase();
    return WORLD_LANGUAGES.filter((lang) => {
      if (atlasRegion !== "All" && lang.region !== atlasRegion) return false;
      if (!q) return true;
      return (
        lang.name.toLowerCase().includes(q) ||
        lang.nativeName.toLowerCase().includes(q) ||
        lang.amharicName.toLowerCase().includes(q) ||
        lang.code.toLowerCase().includes(q)
      );
    });
  }, [atlasQuery, atlasRegion]);

  const sourceLangObj = sourceLang === "auto" ? null : getLanguageByCode(sourceLang);
  const targetLangObj = getLanguageByCode(targetLang);

  return (
    <div className="min-h-screen bg-[#F7F8FA] text-[#0F172A]">
      {/* Top Bar Contract: 3 exact zones separated by gap-8 */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-8 px-6 py-4">
          <a
            href="#studio"
            className="font-display text-xl font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            {t.brand}
          </a>

          <nav
            aria-label="Primary"
            className="hidden md:flex items-center gap-7 text-sm font-medium text-slate-600"
          >
            <a
              href="#studio"
              onClick={() => handleModeSwitch("face")}
              className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navStudio}
            </a>
            <a
              href="#studio"
              onClick={() => handleModeSwitch("room")}
              className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navRooms}
            </a>
            <a
              href="#history"
              className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navHistory}
            </a>
            <a
              href="#languages"
              className="hover:text-slate-900 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navLanguages}
            </a>
          </nav>

          <div className="flex items-center shrink-0">
            <a
              href="#studio"
              onClick={() => {
                handleModeSwitch("face");
                if (liveStatus === "idle" && engineMode === "live") {
                  void startLiveSession();
                }
              }}
              className="inline-flex h-10 items-center justify-center rounded-lg bg-blue-700 px-4 text-xs font-semibold text-white transition-colors hover:bg-blue-800 whitespace-nowrap shrink-0"
            >
              {t.navStart}
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10 space-y-14">
        {/* Hero & Studio Workspace */}
        <section id="studio" className="space-y-6">
          <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
            <div className="max-w-2xl space-y-3">
              <h1
                className="font-display font-bold tracking-tight text-slate-900"
                style={{
                  fontSize: "clamp(1.85rem, 3.2vw, 2.75rem)",
                  lineHeight: 1.18,
                  textWrap: "balance",
                }}
              >
                {t.heroTitle}
              </h1>
              <p className="text-base leading-relaxed text-slate-600">
                {t.heroSubtitle}
              </p>
            </div>

            {/* Workspace Toolbar (Mode, UI Language, Install App) */}
            <div className="flex flex-wrap items-center gap-3 shrink-0">
              <PWAInstallButton uiLang={uiLang} />

              <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white p-1">
                <button
                  type="button"
                  onClick={() => handleModeSwitch("face")}
                  className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    studioMode === "face"
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {t.modeFaceToFace}
                </button>
                <button
                  type="button"
                  onClick={() => handleModeSwitch("room")}
                  className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    studioMode === "room"
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {t.modeRemoteRoom}
                </button>
              </div>

              <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white p-1">
                <button
                  type="button"
                  onClick={() => setUiLang("am")}
                  className={`rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    uiLang === "am"
                      ? "bg-blue-700 text-white"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  አማርኛ
                </button>
                <button
                  type="button"
                  onClick={() => setUiLang("en")}
                  className={`rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    uiLang === "en"
                      ? "bg-blue-700 text-white"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  English
                </button>
              </div>
            </div>
          </div>

          {studioMode === "room" ? (
            <RemoteRoomView uiLang={uiLang} initialRoomCode={initialRoomParam} />
          ) : (
            <div className="space-y-4">
              {/* Sub-toolbar for Face-to-Face Engine & Voice */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-200 bg-white px-5 py-3">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="inline-flex items-center rounded-lg bg-slate-100 p-1">
                    <button
                      type="button"
                      onClick={() => handleEngineSwitch("live")}
                      className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                        engineMode === "live"
                          ? "bg-white text-slate-900 shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {t.engineContinuous}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleEngineSwitch("turn")}
                      className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                        engineMode === "turn"
                          ? "bg-white text-slate-900 shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      {t.engineTurn}
                    </button>
                  </div>

                  <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        liveStatus === "active" || turnRecording
                          ? "bg-red-600"
                          : liveStatus === "connecting" || turnProcessing
                          ? "bg-amber-500"
                          : "bg-slate-300"
                      }`}
                      aria-hidden="true"
                    />
                    <span>
                      {liveStatus === "active"
                        ? t.liveConnected
                        : liveStatus === "connecting"
                        ? t.connectingLive
                        : turnRecording
                        ? t.micActive
                        : turnProcessing
                        ? t.translatingAudio
                        : t.readyState}
                    </span>
                    {liveStatus === "active" && (
                      <span className="font-mono text-xs font-semibold text-slate-900 tabular-nums">
                        · {formatDuration(liveSeconds)}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  {sourceLang !== "auto" && (
                    <button
                      type="button"
                      onClick={() => {
                        if (liveStatus !== "idle") stopLiveSession();
                        setBidirectional((b) => !b);
                      }}
                      aria-pressed={bidirectional}
                      className={`inline-flex h-9 items-center gap-1.5 rounded-lg border px-3 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                        bidirectional
                          ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                          : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      <ArrowLeftRight className="h-3.5 w-3.5" />
                      <span>{bidirectional ? t.twoWayAutoLabel : t.oneWayLabel}</span>
                    </button>
                  )}

                  <label className="flex items-center gap-2 text-xs font-medium text-slate-600">
                    <span>{t.voiceLabel}</span>
                    <select
                      value={voiceName}
                      onChange={(e) => setVoiceName(e.target.value as VoiceId)}
                      className="h-9 rounded-lg border border-slate-200 bg-slate-50 px-2.5 text-xs font-semibold text-slate-900 focus:border-blue-600 focus:outline-none"
                    >
                      {VOICE_PROFILES.map((v) => (
                        <option key={v.id} value={v.id}>
                          {uiLang === "am" ? v.labelAm : `${v.label} · ${v.tone}`}
                        </option>
                      ))}
                    </select>
                  </label>

                  <button
                    type="button"
                    onClick={() => setAutoPlay((v) => !v)}
                    aria-pressed={autoPlay}
                    className={`inline-flex h-9 items-center gap-1.5 rounded-lg border px-3 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                      autoPlay
                        ? "border-blue-200 bg-blue-50 text-blue-800"
                        : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {autoPlay ? (
                      <Volume2 className="h-3.5 w-3.5" />
                    ) : (
                      <VolumeX className="h-3.5 w-3.5" />
                    )}
                    <span>{t.autoPlayAudio}</span>
                  </button>
                </div>
              </div>

              {/* Quick Language Pair Presets */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium text-slate-500">{t.quickPairsLabel}:</span>
                {QUICK_PAIRS.map((pair) => {
                  const isActive =
                    (sourceLang === pair.from && targetLang === pair.to) ||
                    (sourceLang === pair.to && targetLang === pair.from);
                  return (
                    <button
                      key={pair.label}
                      type="button"
                      onClick={() => {
                        if (liveStatus !== "idle") stopLiveSession();
                        setSourceLang(pair.from);
                        setTargetLang(pair.to);
                      }}
                      className={`rounded-md border px-2.5 py-1 font-medium transition-colors whitespace-nowrap shrink-0 ${
                        isActive
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      {pair.label}
                    </button>
                  );
                })}
              </div>

              {errorBanner && (
                <div
                  role="alert"
                  className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
                >
                  {errorBanner}
                </div>
              )}

              {/* Dual-Channel Studio Grid (Speaker <-> Listener) */}
              <div className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-[1fr_auto_1fr]">
                {/* Speaker Panel */}
                <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                        {t.speakerLabel}
                      </span>
                      <button
                        type="button"
                        onClick={() => setPickerTarget("source")}
                        className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3.5 text-sm font-semibold text-slate-900 transition-colors hover:border-slate-300 hover:bg-white whitespace-nowrap shrink-0"
                      >
                        <span>
                          {sourceLangObj
                            ? `${sourceLangObj.nativeName} · ${
                                uiLang === "am"
                                  ? sourceLangObj.amharicName
                                  : sourceLangObj.name
                              }`
                            : t.autoDetect}
                        </span>
                        <ChevronDown className="h-4 w-4 text-slate-400" />
                      </button>
                    </div>

                    <div className="border-t border-slate-100 pt-4">
                      <WaveformCanvas
                        analyser={inputAnalyser}
                        active={liveStatus === "active" || turnRecording}
                        accentColor="#DC2626"
                      />
                    </div>

                    <div className="min-h-[150px] py-2">
                      {liveInputText ? (
                        <p className="text-lg font-medium leading-relaxed text-slate-900">
                          {liveInputText}
                        </p>
                      ) : (
                        <p className="text-sm leading-relaxed text-slate-400">
                          {t.speakerPlaceholder}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-3 border-t border-slate-100 pt-4">
                    {engineMode === "live" ? (
                      <button
                        type="button"
                        onClick={
                          liveStatus === "idle" ? startLiveSession : stopLiveSession
                        }
                        className={`inline-flex h-12 w-full items-center justify-center gap-2.5 rounded-lg px-5 text-sm font-semibold text-white transition-colors ${
                          liveStatus === "active"
                            ? "bg-red-600 hover:bg-red-700"
                            : liveStatus === "connecting"
                            ? "bg-slate-700 hover:bg-slate-800"
                            : "bg-blue-700 hover:bg-blue-800"
                        }`}
                      >
                        {liveStatus === "connecting" ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : liveStatus === "active" ? (
                          <MicOff className="h-4 w-4" />
                        ) : (
                          <Mic className="h-4 w-4" />
                        )}
                        <span>
                          {liveStatus === "connecting"
                            ? t.connectingLive
                            : liveStatus === "active"
                            ? t.stopLiveMic
                            : t.startLiveMic}
                        </span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={
                          turnRecording ? stopTurnRecording : startTurnRecording
                        }
                        disabled={turnProcessing}
                        className={`inline-flex h-12 w-full items-center justify-center gap-2.5 rounded-lg px-5 text-sm font-semibold text-white transition-colors disabled:opacity-50 ${
                          turnRecording
                            ? "bg-red-600 hover:bg-red-700"
                            : "bg-blue-700 hover:bg-blue-800"
                        }`}
                      >
                        {turnProcessing ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : turnRecording ? (
                          <Square className="h-4 w-4 fill-current" />
                        ) : (
                          <Mic className="h-4 w-4" />
                        )}
                        <span>
                          {turnProcessing
                            ? t.translatingAudio
                            : turnRecording
                            ? t.stopAndTranslate
                            : t.holdOrClickRecord}
                        </span>
                      </button>
                    )}

                    <form onSubmit={handleSendPhrase} className="flex items-center gap-2">
                      <input
                        type="text"
                        value={phraseInput}
                        onChange={(e) => setPhraseInput(e.target.value)}
                        placeholder={t.typeToSpeak}
                        className="h-10 min-w-0 flex-1 rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:bg-white focus:outline-none"
                      />
                      <button
                        type="submit"
                        disabled={!phraseInput.trim() || turnProcessing}
                        className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 text-xs font-semibold text-slate-800 transition-colors hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
                      >
                        <Send className="h-3.5 w-3.5" />
                        <span>{t.sendAndSpeak}</span>
                      </button>
                    </form>
                  </div>
                </div>

                {/* Swap Action */}
                <div className="flex items-center justify-center">
                  <button
                    type="button"
                    onClick={handleSwapLanguages}
                    title={t.swapLanguages}
                    aria-label={t.swapLanguages}
                    className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 transition-colors hover:border-blue-600 hover:text-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700"
                  >
                    <ArrowLeftRight className="h-4 w-4" />
                  </button>
                </div>

                {/* Listener Panel */}
                <div className="flex flex-col justify-between rounded-xl border border-slate-200 bg-white p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-xs font-semibold uppercase tracking-wider text-blue-700">
                        {t.listenerLabel}
                      </span>
                      <button
                        type="button"
                        onClick={() => setPickerTarget("target")}
                        className="inline-flex h-10 items-center gap-2 rounded-lg border border-blue-200 bg-blue-50/70 px-3.5 text-sm font-semibold text-blue-900 transition-colors hover:border-blue-300 hover:bg-blue-50 whitespace-nowrap shrink-0"
                      >
                        <span>
                          {targetLangObj.nativeName} ·{" "}
                          {uiLang === "am"
                            ? targetLangObj.amharicName
                            : targetLangObj.name}
                        </span>
                        <ChevronDown className="h-4 w-4 text-blue-700" />
                      </button>
                    </div>

                    <div className="border-t border-slate-100 pt-4">
                      <WaveformCanvas
                        analyser={outputAnalyser}
                        active={liveStatus === "active" && Boolean(liveOutputText)}
                        accentColor="#1D4ED8"
                      />
                    </div>

                    <div className="min-h-[150px] py-2">
                      {liveOutputText ? (
                        <p className="text-xl font-semibold leading-relaxed text-slate-900">
                          {liveOutputText}
                        </p>
                      ) : (
                        <p className="text-sm leading-relaxed text-slate-400">
                          {t.listenerPlaceholder}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between border-t border-slate-100 pt-4">
                    <div className="text-xs text-slate-500">
                      {targetLangObj.nativeName} ·{" "}
                      <span className="font-mono uppercase">{targetLangObj.code}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      {lastTurnAudio && (
                        <button
                          type="button"
                          onClick={() => void playWavBase64(lastTurnAudio)}
                          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-800 transition-colors hover:bg-slate-50 whitespace-nowrap"
                        >
                          <Volume2 className="h-3.5 w-3.5 text-blue-700" />
                          <span>{t.replayAudio}</span>
                        </button>
                      )}
                      {liveOutputText && (
                        <button
                          type="button"
                          onClick={() => void handleCopyText("current", liveOutputText)}
                          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-800 transition-colors hover:bg-slate-50 whitespace-nowrap"
                        >
                          {copiedId === "current" ? (
                            <Check className="h-3.5 w-3.5 text-emerald-700" />
                          ) : (
                            <Copy className="h-3.5 w-3.5" />
                          )}
                          <span>{copiedId === "current" ? t.copied : t.copyText}</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* How It Works */}
        <section id="how-it-works" className="border-t border-slate-200 pt-10">
          <h2 className="font-display text-xl font-semibold text-slate-900">
            {t.howItWorksTitle}
          </h2>
          <div className="mt-5 grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="rounded-xl border border-slate-200 bg-white p-5">
              <h3 className="font-display text-base font-semibold text-slate-900">
                {t.step1Title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {t.step1Desc}
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-5">
              <h3 className="font-display text-base font-semibold text-slate-900">
                {t.step2Title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {t.step2Desc}
              </p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-5">
              <h3 className="font-display text-base font-semibold text-slate-900">
                {t.step3Title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {t.step3Desc}
              </p>
            </div>
          </div>
        </section>

        {/* Session Transcript History */}
        <section id="history" className="border-t border-slate-200 pt-10">
          <div className="flex items-center justify-between gap-4">
            <h2 className="font-display text-xl font-semibold text-slate-900">
              {t.historyTitle}
            </h2>
            {history.length > 0 && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={exportHistory}
                  className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 transition-colors hover:bg-slate-50 whitespace-nowrap"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>{t.exportTranscript}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setHistory([])}
                  className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 transition-colors hover:bg-slate-50 whitespace-nowrap"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>{t.clearHistory}</span>
                </button>
              </div>
            )}
          </div>

          {history.length === 0 ? (
            <div className="mt-4 rounded-xl border border-slate-200 bg-white px-6 py-10 text-center">
              <p className="text-sm text-slate-500">{t.historyEmpty}</p>
            </div>
          ) : (
            <div className="mt-4 divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-200 bg-white">
              {history.map((item) => {
                const srcLabel =
                  item.sourceLang === "auto"
                    ? t.autoDetect
                    : getLanguageByCode(item.sourceLang).nativeName;
                const tgtLabel = getLanguageByCode(item.targetLang).nativeName;
                const timeStr = new Date(item.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });
                return (
                  <div
                    key={item.id}
                    className="flex flex-col justify-between gap-4 px-6 py-4 sm:flex-row sm:items-start"
                  >
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <span>{srcLabel}</span>
                        <span aria-hidden="true">→</span>
                        <span className="font-medium text-slate-700">{tgtLabel}</span>
                        <span aria-hidden="true">·</span>
                        <span className="font-mono tabular-nums">{timeStr}</span>
                      </div>
                      <p className="text-base font-semibold text-slate-900">
                        {item.translatedText}
                      </p>
                      {item.sourceText && (
                        <p className="text-sm text-slate-600">{item.sourceText}</p>
                      )}
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      {item.audioWavBase64 && (
                        <button
                          type="button"
                          onClick={() => void playWavBase64(item.audioWavBase64!)}
                          className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 text-xs font-semibold text-slate-800 hover:bg-slate-100 whitespace-nowrap"
                        >
                          <Volume2 className="h-3.5 w-3.5 text-blue-700" />
                          <span>{t.replayAudio}</span>
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void handleCopyText(item.id, item.translatedText)}
                        className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 hover:bg-slate-50 whitespace-nowrap"
                      >
                        {copiedId === item.id ? (
                          <Check className="h-3.5 w-3.5 text-emerald-700" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                        <span>{copiedId === item.id ? t.copied : t.copyText}</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Supported World Languages Atlas */}
        <section id="languages" className="border-t border-slate-200 pt-10">
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <h2 className="flex items-center gap-2 font-display text-xl font-semibold text-slate-900">
                <Globe className="h-5 w-5 text-blue-700" />
                <span>{t.languagesTitle}</span>
              </h2>
              <p className="mt-1 text-sm text-slate-600">{t.languagesSubtitle}</p>
            </div>

            <div className="relative w-full md:w-72">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={atlasQuery}
                onChange={(e) => setAtlasQuery(e.target.value)}
                placeholder={t.searchLanguage}
                className="h-10 w-full rounded-lg border border-slate-200 bg-white pl-9 pr-3 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-1.5">
            {ATLAS_REGIONS.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setAtlasRegion(r)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  atlasRegion === r
                    ? "bg-slate-900 text-white"
                    : "border border-slate-200 bg-white text-slate-600 hover:text-slate-900"
                }`}
              >
                {r === "All" ? t.allRegions : r}
              </button>
            ))}
          </div>

          <div className="mt-5 grid grid-cols-1 gap-2.5 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {filteredAtlas.map((lang) => (
              <div
                key={lang.code}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3"
              >
                <div className="min-w-0 pr-2">
                  <div className="truncate text-sm font-semibold text-slate-900">
                    {lang.nativeName}
                  </div>
                  <div className="truncate text-xs text-slate-500">
                    {uiLang === "am" ? `${lang.amharicName} · ${lang.name}` : lang.name}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <button
                    type="button"
                    onClick={() => {
                      if (liveStatus !== "idle") stopLiveSession();
                      setSourceLang(lang.code);
                      setStudioMode("face");
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                    title={t.speakerLabel}
                    className={`rounded px-2 py-1 text-[11px] font-semibold transition-colors ${
                      sourceLang === lang.code
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {uiLang === "am" ? "ተናጋሪ" : "In"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (liveStatus !== "idle") stopLiveSession();
                      setTargetLang(lang.code);
                      setStudioMode("face");
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                    title={t.listenerLabel}
                    className={`rounded px-2 py-1 text-[11px] font-semibold transition-colors ${
                      targetLang === lang.code
                        ? "bg-blue-700 text-white"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {uiLang === "am" ? "አድማጭ" : "Out"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="mt-16 border-t border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 py-6 text-xs text-slate-500 sm:flex-row">
          <div className="font-display font-semibold text-slate-800">{t.brand}</div>
          <div>
            {uiLang === "am"
              ? "በቀጥታ ድምፅ የዓለምን ቋንቋዎች የሚያገናኝ መድረክ"
              : "Real-time voice translation across world languages"}
          </div>
        </div>
      </footer>

      <LanguagePickerModal
        open={pickerTarget !== null}
        uiLang={uiLang}
        selectedCode={pickerTarget === "source" ? sourceLang : targetLang}
        allowAuto={pickerTarget === "source"}
        onSelect={handleLanguagePick}
        onClose={() => setPickerTarget(null)}
      />

      <OfflineIndicator uiLang={uiLang} />
    </div>
  );
}
