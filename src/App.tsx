import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftRight,
  Check,
  ChevronDown,
  Copy,
  Ear,
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
  BoostedMicPipeline,
  createBoostedMicPipeline,
  downsampleTo16k,
  float32ToPcm16Base64,
  GaplessPcmPlayer,
  playWavBase64,
  speakBrowserFallback,
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
  const [whisperBoost, setWhisperBoost] = useState<boolean>(() => {
    return localStorage.getItem("lissan_whisper_boost") !== "false";
  });

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
  const [noticeBanner, setNoticeBanner] = useState<string | null>(null);
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
  const inputStreamRef = useRef<MediaStream | null>(null);
  const micPipelineRef = useRef<BoostedMicPipeline | null>(null);
  const playerRef = useRef<GaplessPcmPlayer | null>(null);
  const turnPlayerRef = useRef<GaplessPcmPlayer | null>(null);
  const liveInputAccRef = useRef<string>("");
  const liveOutputAccRef = useRef<string>("");
  const autoPlayRef = useRef<boolean>(autoPlay);
  const whisperBoostRef = useRef<boolean>(whisperBoost);
  const sourceLangRef = useRef<string>(sourceLang);
  const targetLangRef = useRef<string>(targetLang);
  const userStoppedLiveRef = useRef<boolean>(true);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wakeLockRef = useRef<{ release: () => Promise<void> } | null>(null);

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
    whisperBoostRef.current = whisperBoost;
    localStorage.setItem("lissan_whisper_boost", String(whisperBoost));
    micPipelineRef.current?.setBoost(whisperBoost);
  }, [whisperBoost]);

  useEffect(() => {
    sourceLangRef.current = sourceLang;
  }, [sourceLang]);

  useEffect(() => {
    targetLangRef.current = targetLang;
  }, [targetLang]);

  const stopLiveSession = useCallback(() => {
    userStoppedLiveRef.current = true;
    setNoticeBanner(null);
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wakeLockRef.current) {
      void wakeLockRef.current.release().catch(() => undefined);
      wakeLockRef.current = null;
    }

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

    if (micPipelineRef.current) {
      micPipelineRef.current.cleanup();
      micPipelineRef.current = null;
    }
    if (inputStreamRef.current) {
      inputStreamRef.current.getTracks().forEach((track) => track.stop());
      inputStreamRef.current = null;
    }
    if (liveWsRef.current) {
      const wsToClose = liveWsRef.current;
      liveWsRef.current = null;
      wsToClose.close();
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
      if (turnPlayerRef.current) {
        turnPlayerRef.current.close();
        turnPlayerRef.current = null;
      }
    };
  }, [stopLiveSession]);

  const openLiveSocket = useCallback(
    (pipeline: BoostedMicPipeline) => {
      if (userStoppedLiveRef.current) return;
      if (liveWsRef.current) {
        try {
          liveWsRef.current.close();
        } catch {
          // ignore
        }
      }

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const params = new URLSearchParams({
        sourceLang,
        targetLang,
        voiceName,
        bidirectional: String(bidirectional && sourceLang !== "auto"),
      });
      const ws = new WebSocket(
        `${protocol}//${window.location.host}/ws/live?${params.toString()}`
      );
      liveWsRef.current = ws;

      ws.onopen = () => {
        pipeline.processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN || userStoppedLiveRef.current) return;
          if (pipeline.ctx.state === "suspended") {
            void pipeline.ctx.resume();
          }
          const rawChannel = e.inputBuffer.getChannelData(0);
          const pcm16k = downsampleTo16k(rawChannel, pipeline.ctx.sampleRate);
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
              setErrorBanner(null);
              setNoticeBanner(null);
            }
          } else if (msg.type === "inputTranscript" && msg.text) {
            setNoticeBanner(null);
            liveInputAccRef.current += msg.text;
            setLiveInputText(liveInputAccRef.current);
          } else if (msg.type === "outputTranscript" && msg.text) {
            setNoticeBanner(null);
            liveOutputAccRef.current += msg.text;
            setLiveOutputText(liveOutputAccRef.current);
          } else if (msg.type === "audio" && msg.data) {
            playerRef.current?.enqueuePcm24kBase64(msg.data);
          } else if (msg.type === "audioWav" && msg.data) {
            void playerRef.current?.enqueueWavBase64(msg.data);
          } else if (msg.type === "fallbackSpeak" && msg.text) {
            if (autoPlayRef.current) {
              speakBrowserFallback(msg.text, msg.lang || targetLangRef.current);
            }
          } else if (msg.type === "notice") {
            setNoticeBanner(t.highTrafficWait);
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
        // Reconnection handled in onclose while userStoppedLiveRef is false
      };

      ws.onclose = () => {
        if (!userStoppedLiveRef.current && liveWsRef.current === ws) {
          setLiveStatus("connecting");
          reconnectTimerRef.current = setTimeout(() => {
            if (!userStoppedLiveRef.current && micPipelineRef.current) {
              openLiveSocket(micPipelineRef.current);
            }
          }, 1500);
        }
      };
    },
    [bidirectional, sourceLang, targetLang, voiceName, stopLiveSession, t.errorGeneric, t.highTrafficWait]
  );

  const startLiveSession = useCallback(async () => {
    setErrorBanner(null);
    setNoticeBanner(null);
    userStoppedLiveRef.current = false;
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
      if (userStoppedLiveRef.current) {
        stream.getTracks().forEach((tr) => tr.stop());
        return;
      }
      inputStreamRef.current = stream;

      if ("wakeLock" in navigator) {
        void (navigator as Navigator & {
          wakeLock?: { request: (type: string) => Promise<{ release: () => Promise<void> }> };
        }).wakeLock
          ?.request("screen")
          .then((lock) => {
            wakeLockRef.current = lock;
          })
          .catch(() => undefined);
      }

      const player = new GaplessPcmPlayer();
      player.setMuted(!autoPlayRef.current);
      const { analyser: outAnal } = player.ensureContext();
      playerRef.current = player;
      setOutputAnalyser(outAnal);

      const pipeline = await createBoostedMicPipeline(stream, whisperBoostRef.current);
      if (userStoppedLiveRef.current) {
        pipeline.cleanup();
        return;
      }
      micPipelineRef.current = pipeline;
      setInputAnalyser(pipeline.analyser);

      openLiveSocket(pipeline);
    } catch {
      setErrorBanner(t.errorMicDenied);
      stopLiveSession();
    }
  }, [openLiveSocket, stopLiveSession, t.errorMicDenied]);

  const streamTtsAudio = useCallback(
    async (text: string, langCode: string) => {
      const cleaned = text.trim();
      if (!cleaned) return;
      if (!turnPlayerRef.current) {
        turnPlayerRef.current = new GaplessPcmPlayer();
      }
      const player = turnPlayerRef.current;
      player.setMuted(!autoPlayRef.current);
      player.interrupt();
      const { analyser } = player.ensureContext();
      setOutputAnalyser(analyser);

      try {
        const res = await fetch("/api/tts-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: cleaned, voiceName }),
        });
        if (!res.ok || res.status === 204 || !res.body) {
          if (autoPlayRef.current) {
            speakBrowserFallback(cleaned, langCode);
          }
          return;
        }
        const reader = res.body.getReader();
        let leftover: number | null = null;
        let receivedAny = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!value || value.byteLength === 0) continue;

          let bytes = value;
          if (leftover !== null) {
            const merged = new Uint8Array(value.byteLength + 1);
            merged[0] = leftover;
            merged.set(value, 1);
            bytes = merged;
            leftover = null;
          }
          if (bytes.byteLength % 2 === 1) {
            leftover = bytes[bytes.byteLength - 1];
            bytes = bytes.subarray(0, bytes.byteLength - 1);
          }
          if (bytes.byteLength >= 2) {
            receivedAny = true;
            player.enqueuePcm24kBytes(bytes);
          }
        }
        if (!receivedAny && autoPlayRef.current) {
          speakBrowserFallback(cleaned, langCode);
        }
      } catch {
        if (autoPlayRef.current) {
          speakBrowserFallback(cleaned, langCode);
        }
      }
    },
    [voiceName]
  );

  const executeTurnTranslation = useCallback(
    async (payload: { audioBase64?: string; mimeType?: string; text?: string }) => {
      setTurnProcessing(true);
      setErrorBanner(null);
      setNoticeBanner(null);
      try {
        const res = await fetch("/api/translate-fast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            sourceLang,
            targetLang,
            bidirectional: bidirectional && sourceLang !== "auto",
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          if (res.status === 429 || res.status === 503) {
            setNoticeBanner(t.highTrafficWait);
            return;
          }
          throw new Error(data.error || t.errorGeneric);
        }

        if (data.sourceTranscript || data.translatedText) {
          setLiveInputText(data.sourceTranscript || "");
          setLiveOutputText(data.translatedText || "");
          setLastTurnAudio(null);

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
              audioWavBase64: null,
              timestamp: Date.now(),
            },
            ...prev,
          ]);

          setTurnProcessing(false);
          if (autoPlay && data.translatedText) {
            void streamTtsAudio(data.translatedText, resolvedTgt);
          }
        }
      } catch (err) {
        setErrorBanner(err instanceof Error ? err.message : t.errorGeneric);
      } finally {
        setTurnProcessing(false);
      }
    },
    [autoPlay, bidirectional, sourceLang, streamTtsAudio, targetLang, t.errorGeneric, t.highTrafficWait]
  );

  const startTurnRecording = async () => {
    setErrorBanner(null);
    setNoticeBanner(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
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
    <div className="min-h-screen bg-[#F7F8FA] text-slate-950 font-medium">
      {/* Top Bar Contract: 3 exact zones separated by gap-8 */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-8 px-6 py-4">
          <a
            href="#studio"
            className="font-display text-2xl font-extrabold tracking-tight text-slate-950 whitespace-nowrap shrink-0"
          >
            {t.brand}
          </a>

          <nav
            aria-label="Primary"
            className="hidden md:flex items-center gap-7 text-base font-bold text-slate-800"
          >
            <a
              href="#studio"
              onClick={() => handleModeSwitch("face")}
              className="hover:text-blue-700 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navStudio}
            </a>
            <a
              href="#studio"
              onClick={() => handleModeSwitch("room")}
              className="hover:text-blue-700 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navRooms}
            </a>
            <a
              href="#history"
              className="hover:text-blue-700 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
            >
              {t.navHistory}
            </a>
            <a
              href="#languages"
              className="hover:text-blue-700 hover:underline underline-offset-4 transition-colors whitespace-nowrap shrink-0"
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
              className="inline-flex h-11 items-center justify-center rounded-lg bg-blue-700 px-5 text-sm font-bold text-white transition-colors hover:bg-blue-800 whitespace-nowrap shrink-0"
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
                className="font-display font-extrabold tracking-tight text-slate-950"
                style={{
                  fontSize: "clamp(1.95rem, 3.4vw, 2.85rem)",
                  lineHeight: 1.18,
                  textWrap: "balance",
                }}
              >
                {t.heroTitle}
              </h1>
              <p className="text-lg font-semibold leading-relaxed text-slate-700">
                {t.heroSubtitle}
              </p>
            </div>

            {/* Workspace Toolbar (Mode, UI Language, Install App) */}
            <div className="flex flex-wrap items-center gap-3 shrink-0">
              <PWAInstallButton uiLang={uiLang} />

              <div className="inline-flex items-center rounded-lg border border-slate-300 bg-white p-1">
                <button
                  type="button"
                  onClick={() => handleModeSwitch("face")}
                  className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                    studioMode === "face"
                      ? "bg-slate-900 text-white"
                      : "text-slate-700 hover:text-slate-950"
                  }`}
                >
                  {t.modeFaceToFace}
                </button>
                <button
                  type="button"
                  onClick={() => handleModeSwitch("room")}
                  className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                    studioMode === "room"
                      ? "bg-slate-900 text-white"
                      : "text-slate-700 hover:text-slate-950"
                  }`}
                >
                  {t.modeRemoteRoom}
                </button>
              </div>

              <div className="inline-flex items-center rounded-lg border border-slate-300 bg-white p-1">
                <button
                  type="button"
                  onClick={() => setUiLang("am")}
                  className={`rounded-md px-3 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                    uiLang === "am"
                      ? "bg-blue-700 text-white"
                      : "text-slate-700 hover:text-slate-950"
                  }`}
                >
                  አማርኛ
                </button>
                <button
                  type="button"
                  onClick={() => setUiLang("en")}
                  className={`rounded-md px-3 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                    uiLang === "en"
                      ? "bg-blue-700 text-white"
                      : "text-slate-700 hover:text-slate-950"
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
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-300 bg-white px-5 py-3.5">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="inline-flex items-center rounded-lg bg-slate-100 p-1">
                    <button
                      type="button"
                      onClick={() => handleEngineSwitch("live")}
                      className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                        engineMode === "live"
                          ? "bg-white text-slate-950 shadow-xs"
                          : "text-slate-700 hover:text-slate-950"
                      }`}
                    >
                      {t.engineContinuous}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleEngineSwitch("turn")}
                      className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                        engineMode === "turn"
                          ? "bg-white text-slate-950 shadow-xs"
                          : "text-slate-700 hover:text-slate-950"
                      }`}
                    >
                      {t.engineTurn}
                    </button>
                  </div>

                  <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        liveStatus === "active" || turnRecording
                          ? "bg-red-600"
                          : liveStatus === "connecting" || turnProcessing
                          ? "bg-amber-500"
                          : "bg-slate-400"
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
                      <span className="font-mono text-sm font-bold text-slate-950 tabular-nums">
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
                      className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                        bidirectional
                          ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                          : "border-slate-300 bg-white text-slate-800 hover:bg-slate-50"
                      }`}
                    >
                      <ArrowLeftRight className="h-4 w-4" />
                      <span>{bidirectional ? t.twoWayAutoLabel : t.oneWayLabel}</span>
                    </button>
                  )}

                  <label className="flex items-center gap-2 text-sm font-bold text-slate-800">
                    <span>{t.voiceLabel}</span>
                    <select
                      value={voiceName}
                      onChange={(e) => setVoiceName(e.target.value as VoiceId)}
                      className="h-10 rounded-lg border border-slate-300 bg-slate-50 px-3 text-sm font-bold text-slate-950 focus:border-blue-600 focus:outline-none"
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
                    onClick={() => setWhisperBoost((v) => !v)}
                    aria-pressed={whisperBoost}
                    className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                      whisperBoost
                        ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                        : "border-slate-300 bg-white text-slate-800 hover:bg-slate-50"
                    }`}
                  >
                    <Ear className="h-4 w-4" />
                    <span>{t.whisperBoostLabel}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setAutoPlay((v) => !v)}
                    aria-pressed={autoPlay}
                    className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                      autoPlay
                        ? "border-blue-300 bg-blue-50 text-blue-900"
                        : "border-slate-300 bg-white text-slate-800 hover:bg-slate-50"
                    }`}
                  >
                    {autoPlay ? (
                      <Volume2 className="h-4 w-4" />
                    ) : (
                      <VolumeX className="h-4 w-4" />
                    )}
                    <span>{t.autoPlayAudio}</span>
                  </button>
                </div>
              </div>

              {/* Quick Language Pair Presets */}
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="font-bold text-slate-700">{t.quickPairsLabel}:</span>
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
                      className={`rounded-md border px-3 py-1.5 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                        isActive
                          ? "border-slate-900 bg-slate-900 text-white"
                          : "border-slate-300 bg-white text-slate-800 hover:border-slate-400 hover:bg-slate-50"
                      }`}
                    >
                      {pair.label}
                    </button>
                  );
                })}
              </div>

              {noticeBanner && !errorBanner && (
                <div
                  role="status"
                  className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-base font-bold text-amber-950"
                >
                  {noticeBanner}
                </div>
              )}

              {errorBanner && (
                <div
                  role="alert"
                  className="rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-base font-bold text-red-900"
                >
                  {errorBanner}
                </div>
              )}

              {/* Dual-Channel Studio Grid (Speaker <-> Listener) */}
              <div className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-[1fr_auto_1fr]">
                {/* Speaker Panel */}
                <div className="flex flex-col justify-between rounded-xl border border-slate-300 bg-white p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-extrabold uppercase tracking-wider text-slate-800">
                          {t.speakerLabel}
                        </span>
                        {liveInputText && (
                          <button
                            type="button"
                            onClick={() => {
                              liveInputAccRef.current = "";
                              setLiveInputText("");
                            }}
                            title={t.clearScreenText}
                            aria-label={t.clearScreenText}
                            className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 text-xs font-bold text-slate-700 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                            <span>{t.deleteItem}</span>
                          </button>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => setPickerTarget("source")}
                        className="inline-flex h-11 items-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-4 text-base font-bold text-slate-950 transition-colors hover:border-slate-400 hover:bg-white whitespace-nowrap shrink-0"
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
                        <ChevronDown className="h-4 w-4 text-slate-700" />
                      </button>
                    </div>

                    <div className="border-t border-slate-200 pt-4">
                      <WaveformCanvas
                        analyser={inputAnalyser}
                        active={liveStatus === "active" || turnRecording}
                        accentColor="#DC2626"
                      />
                    </div>

                    <div className="min-h-[150px] py-2">
                      {liveInputText ? (
                        <p className="text-xl sm:text-2xl font-bold leading-relaxed text-slate-950">
                          {liveInputText}
                        </p>
                      ) : (
                        <p className="text-base font-semibold leading-relaxed text-slate-600">
                          {t.speakerPlaceholder}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-3 border-t border-slate-200 pt-4">
                    {engineMode === "live" ? (
                      <button
                        type="button"
                        onClick={
                          liveStatus === "idle" ? startLiveSession : stopLiveSession
                        }
                        className={`inline-flex h-14 w-full items-center justify-center gap-2.5 rounded-lg px-5 text-base font-extrabold text-white transition-colors ${
                          liveStatus === "active"
                            ? "bg-red-600 hover:bg-red-700"
                            : liveStatus === "connecting"
                            ? "bg-slate-800 hover:bg-slate-900"
                            : "bg-blue-700 hover:bg-blue-800"
                        }`}
                      >
                        {liveStatus === "connecting" ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : liveStatus === "active" ? (
                          <MicOff className="h-5 w-5" />
                        ) : (
                          <Mic className="h-5 w-5" />
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
                        className={`inline-flex h-14 w-full items-center justify-center gap-2.5 rounded-lg px-5 text-base font-extrabold text-white transition-colors disabled:opacity-50 ${
                          turnRecording
                            ? "bg-red-600 hover:bg-red-700"
                            : "bg-blue-700 hover:bg-blue-800"
                        }`}
                      >
                        {turnProcessing ? (
                          <Loader2 className="h-5 w-5 animate-spin" />
                        ) : turnRecording ? (
                          <Square className="h-5 w-5 fill-current" />
                        ) : (
                          <Mic className="h-5 w-5" />
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
                        className="h-11 min-w-0 flex-1 rounded-lg border border-slate-300 bg-slate-50 px-3.5 text-base font-semibold text-slate-950 placeholder:text-slate-500 focus:border-blue-600 focus:bg-white focus:outline-none"
                      />
                      <button
                        type="submit"
                        disabled={!phraseInput.trim() || turnProcessing}
                        className="inline-flex h-11 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-4 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 disabled:opacity-50 whitespace-nowrap shrink-0"
                      >
                        <Send className="h-4 w-4" />
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
                    className="flex h-12 w-12 items-center justify-center rounded-full border border-slate-300 bg-white text-slate-800 transition-colors hover:border-blue-600 hover:text-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700"
                  >
                    <ArrowLeftRight className="h-5 w-5" />
                  </button>
                </div>

                {/* Listener Panel */}
                <div className="flex flex-col justify-between rounded-xl border border-slate-300 bg-white p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm font-extrabold uppercase tracking-wider text-blue-800">
                        {t.listenerLabel}
                      </span>
                      <button
                        type="button"
                        onClick={() => setPickerTarget("target")}
                        className="inline-flex h-11 items-center gap-2 rounded-lg border border-blue-300 bg-blue-50 px-4 text-base font-bold text-blue-950 transition-colors hover:border-blue-400 hover:bg-blue-100 whitespace-nowrap shrink-0"
                      >
                        <span>
                          {targetLangObj.nativeName} ·{" "}
                          {uiLang === "am"
                            ? targetLangObj.amharicName
                            : targetLangObj.name}
                        </span>
                        <ChevronDown className="h-4 w-4 text-blue-800" />
                      </button>
                    </div>

                    <div className="border-t border-slate-200 pt-4">
                      <WaveformCanvas
                        analyser={outputAnalyser}
                        active={liveStatus === "active" && Boolean(liveOutputText)}
                        accentColor="#1D4ED8"
                      />
                    </div>

                    <div className="min-h-[150px] py-2">
                      {liveOutputText ? (
                        <p className="text-xl sm:text-2xl font-bold leading-relaxed text-slate-950">
                          {liveOutputText}
                        </p>
                      ) : (
                        <p className="text-base font-semibold leading-relaxed text-slate-600">
                          {t.listenerPlaceholder}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between border-t border-slate-200 pt-4">
                    <div className="text-sm font-bold text-slate-700">
                      {targetLangObj.nativeName} ·{" "}
                      <span className="font-mono uppercase">{targetLangObj.code}</span>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      {liveOutputText && (
                        <button
                          type="button"
                          onClick={() =>
                            lastTurnAudio
                              ? void playWavBase64(lastTurnAudio)
                              : void streamTtsAudio(liveOutputText, targetLang)
                          }
                          className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap"
                        >
                          <Volume2 className="h-4 w-4 text-blue-700" />
                          <span>{t.replayAudio}</span>
                        </button>
                      )}
                      {liveOutputText && (
                        <button
                          type="button"
                          onClick={() => void handleCopyText("current", liveOutputText)}
                          className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap"
                        >
                          {copiedId === "current" ? (
                            <Check className="h-4 w-4 text-emerald-700" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                          <span>{copiedId === "current" ? t.copied : t.copyText}</span>
                        </button>
                      )}
                      {liveOutputText && (
                        <button
                          type="button"
                          onClick={() => {
                            liveOutputAccRef.current = "";
                            setLiveOutputText("");
                            setLastTurnAudio(null);
                          }}
                          title={t.clearScreenText}
                          aria-label={t.clearScreenText}
                          className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 text-sm font-bold text-slate-700 transition-colors hover:border-red-300 hover:bg-red-50 hover:text-red-700 whitespace-nowrap"
                        >
                          <Trash2 className="h-4 w-4" />
                          <span>{t.deleteItem}</span>
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
          <h2 className="font-display text-2xl font-extrabold text-slate-950">
            {t.howItWorksTitle}
          </h2>
          <div className="mt-5 grid grid-cols-1 gap-6 md:grid-cols-3">
            <div className="rounded-xl border border-slate-300 bg-white p-6">
              <h3 className="font-display text-lg font-extrabold text-slate-950">
                {t.step1Title}
              </h3>
              <p className="mt-2.5 text-base font-semibold leading-relaxed text-slate-700">
                {t.step1Desc}
              </p>
            </div>
            <div className="rounded-xl border border-slate-300 bg-white p-6">
              <h3 className="font-display text-lg font-extrabold text-slate-950">
                {t.step2Title}
              </h3>
              <p className="mt-2.5 text-base font-semibold leading-relaxed text-slate-700">
                {t.step2Desc}
              </p>
            </div>
            <div className="rounded-xl border border-slate-300 bg-white p-6">
              <h3 className="font-display text-lg font-extrabold text-slate-950">
                {t.step3Title}
              </h3>
              <p className="mt-2.5 text-base font-semibold leading-relaxed text-slate-700">
                {t.step3Desc}
              </p>
            </div>
          </div>
        </section>

        {/* Session Transcript History */}
        <section id="history" className="border-t border-slate-200 pt-10">
          <div className="flex items-center justify-between gap-4">
            <h2 className="font-display text-2xl font-extrabold text-slate-950">
              {t.historyTitle}
            </h2>
            {history.length > 0 && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={exportHistory}
                  className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap"
                >
                  <Download className="h-4 w-4" />
                  <span>{t.exportTranscript}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setHistory([])}
                  className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap"
                >
                  <Trash2 className="h-4 w-4" />
                  <span>{t.clearHistory}</span>
                </button>
              </div>
            )}
          </div>

          {history.length === 0 ? (
            <div className="mt-4 rounded-xl border border-slate-300 bg-white px-6 py-10 text-center">
              <p className="text-base font-semibold text-slate-700">{t.historyEmpty}</p>
            </div>
          ) : (
            <div className="mt-4 divide-y divide-slate-200 overflow-hidden rounded-xl border border-slate-300 bg-white">
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
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <div className="flex items-center gap-2 text-sm font-bold text-slate-700">
                        <span>{srcLabel}</span>
                        <span aria-hidden="true">→</span>
                        <span className="text-blue-800">{tgtLabel}</span>
                        <span aria-hidden="true">·</span>
                        <span className="font-mono tabular-nums">{timeStr}</span>
                      </div>
                      <p className="text-lg font-bold text-slate-950">
                        {item.translatedText}
                      </p>
                      {item.sourceText && (
                        <p className="text-base font-semibold text-slate-700">{item.sourceText}</p>
                      )}
                    </div>
                    <div className="flex shrink-0 flex-wrap items-center gap-2">
                      {item.translatedText && (
                        <button
                          type="button"
                          onClick={() =>
                            item.audioWavBase64
                              ? void playWavBase64(item.audioWavBase64)
                              : void streamTtsAudio(item.translatedText, item.targetLang)
                          }
                          className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-slate-50 px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-100 whitespace-nowrap"
                        >
                          <Volume2 className="h-4 w-4 text-blue-700" />
                          <span>{t.replayAudio}</span>
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void handleCopyText(item.id, item.translatedText)}
                        className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-50 whitespace-nowrap"
                      >
                        {copiedId === item.id ? (
                          <Check className="h-4 w-4 text-emerald-700" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                        <span>{copiedId === item.id ? t.copied : t.copyText}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          setHistory((prev) => prev.filter((entry) => entry.id !== item.id))
                        }
                        title={t.deleteItem}
                        aria-label={t.deleteItem}
                        className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 text-sm font-bold text-slate-700 transition-colors hover:border-red-300 hover:bg-red-50 hover:text-red-700 whitespace-nowrap"
                      >
                        <Trash2 className="h-4 w-4" />
                        <span>{t.deleteItem}</span>
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
              <h2 className="flex items-center gap-2 font-display text-2xl font-extrabold text-slate-950">
                <Globe className="h-6 w-6 text-blue-700" />
                <span>{t.languagesTitle}</span>
              </h2>
              <p className="mt-1.5 text-base font-semibold text-slate-700">{t.languagesSubtitle}</p>
            </div>

            <div className="relative w-full md:w-72">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                value={atlasQuery}
                onChange={(e) => setAtlasQuery(e.target.value)}
                placeholder={t.searchLanguage}
                className="h-11 w-full rounded-lg border border-slate-300 bg-white pl-9 pr-3 text-base font-semibold text-slate-950 placeholder:text-slate-500 focus:border-blue-600 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            {ATLAS_REGIONS.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setAtlasRegion(r)}
                className={`rounded-md px-3.5 py-2 text-sm font-bold transition-colors whitespace-nowrap shrink-0 ${
                  atlasRegion === r
                    ? "bg-slate-900 text-white"
                    : "border border-slate-300 bg-white text-slate-800 hover:text-slate-950"
                }`}
              >
                {r === "All" ? t.allRegions : r}
              </button>
            ))}
          </div>

          <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
            {filteredAtlas.map((lang) => (
              <div
                key={lang.code}
                className="flex items-center justify-between rounded-lg border border-slate-300 bg-white px-4 py-3.5"
              >
                <div className="min-w-0 pr-2">
                  <div className="truncate text-base font-extrabold text-slate-950">
                    {lang.nativeName}
                  </div>
                  <div className="truncate text-sm font-semibold text-slate-700">
                    {uiLang === "am" ? `${lang.amharicName} · ${lang.name}` : lang.name}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => {
                      if (liveStatus !== "idle") stopLiveSession();
                      setSourceLang(lang.code);
                      setStudioMode("face");
                      window.scrollTo({ top: 0, behavior: "smooth" });
                    }}
                    title={t.speakerLabel}
                    className={`rounded px-2.5 py-1.5 text-xs font-bold transition-colors ${
                      sourceLang === lang.code
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-800 hover:bg-slate-200"
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
                    className={`rounded px-2.5 py-1.5 text-xs font-bold transition-colors ${
                      targetLang === lang.code
                        ? "bg-blue-700 text-white"
                        : "bg-slate-100 text-slate-800 hover:bg-slate-200"
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
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 py-6 text-sm font-bold text-slate-700 sm:flex-row">
          <div className="font-display text-base font-extrabold text-slate-950">{t.brand}</div>
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
