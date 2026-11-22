import { useCallback, useEffect, useRef, useState } from "react";
import {
  Mic,
  MicOff,
  PhoneCall,
  PhoneOff,
  Copy,
  Check,
  Volume2,
  Send,
  LogOut,
  Users,
  ChevronDown,
  ShieldCheck,
  Download,
  Share2,
  Ear,
  Trash2,
} from "lucide-react";
import { UI_COPY, UiLang } from "../data/i18n";
import { getLanguageByCode, VOICE_PROFILES, VoiceId } from "../data/languages";
import {
  BoostedMicPipeline,
  createBoostedMicPipeline,
  downsampleTo16k,
  float32ToPcm16Base64,
  GaplessPcmPlayer,
  playWavBase64,
  speakBrowserFallback,
} from "../utils/audio";
import { WaveformCanvas } from "./WaveformCanvas";
import { LanguagePickerModal } from "./LanguagePickerModal";

interface Participant {
  id: string;
  name: string;
  langCode: string;
  voiceName: string;
  liveCallActive?: boolean;
}

interface RoomMessage {
  id: string;
  speakerId: string;
  speakerName: string;
  sourceLang: string;
  sourceText: string;
  targetLang: string;
  translatedText: string;
  audioWavBase64?: string | null;
  liveStreamed?: boolean;
  timestamp: number;
}

interface LiveCaption {
  speakerName: string;
  sourceText: string;
  translatedText: string;
}

interface RemoteRoomViewProps {
  uiLang: UiLang;
  initialRoomCode?: string;
}

export function RemoteRoomView({ uiLang, initialRoomCode = "" }: RemoteRoomViewProps) {
  const t = UI_COPY[uiLang];

  const [participantId] = useState(() => {
    const existing = sessionStorage.getItem("lissan_pid");
    if (existing) return existing;
    const id = `u_${Math.random().toString(36).slice(2, 10)}`;
    sessionStorage.setItem("lissan_pid", id);
    return id;
  });

  const [name, setName] = useState(() => {
    return localStorage.getItem("lissan_room_name") || (uiLang === "am" ? "ተናጋሪ" : "Speaker");
  });
  const [myLangCode, setMyLangCode] = useState<string>(() => {
    return localStorage.getItem("lissan_room_lang") || "am";
  });
  const [myVoice, setMyVoice] = useState<VoiceId>(() => {
    return (localStorage.getItem("lissan_room_voice") as VoiceId) || "Zephyr";
  });
  const [joinCodeInput, setJoinCodeInput] = useState<string>(initialRoomCode.toUpperCase());
  const [activeRoomCode, setActiveRoomCode] = useState<string | null>(null);

  const [participants, setParticipants] = useState<Participant[]>([]);
  const [messages, setMessages] = useState<RoomMessage[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  // Continuous live call state
  const [liveCallActive, setLiveCallActive] = useState(false);
  const [callSeconds, setCallSeconds] = useState(0);
  const [micMuted, setMicMuted] = useState(false);
  const [echoGuard, setEchoGuard] = useState(true);
  const [whisperBoost, setWhisperBoost] = useState<boolean>(() => {
    return localStorage.getItem("lissan_whisper_boost") !== "false";
  });
  const [liveCaption, setLiveCaption] = useState<LiveCaption | null>(null);

  const [isSendingText, setIsSendingText] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [autoPlay, setAutoPlay] = useState(true);
  const [copiedLink, setCopiedLink] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);
  const [outputAnalyser, setOutputAnalyser] = useState<AnalyserNode | null>(null);

  // Refs for WebSocket & WebAudio callbacks
  const wsRef = useRef<WebSocket | null>(null);
  const playerRef = useRef<GaplessPcmPlayer | null>(null);
  const inputStreamRef = useRef<MediaStream | null>(null);
  const micPipelineRef = useRef<BoostedMicPipeline | null>(null);
  const autoPlayRef = useRef(autoPlay);
  const micMutedRef = useRef(micMuted);
  const echoGuardRef = useRef(echoGuard);
  const whisperBoostRef = useRef(whisperBoost);
  const liveCallActiveRef = useRef(liveCallActive);
  const lastIncomingAudioAtRef = useRef<number>(0);
  const manualLeaveRef = useRef(false);
  const activeRoomRef = useRef<string | null>(null);

  useEffect(() => {
    localStorage.setItem("lissan_room_name", name);
  }, [name]);

  useEffect(() => {
    localStorage.setItem("lissan_room_lang", myLangCode);
  }, [myLangCode]);

  useEffect(() => {
    localStorage.setItem("lissan_room_voice", myVoice);
  }, [myVoice]);

  useEffect(() => {
    autoPlayRef.current = autoPlay;
    playerRef.current?.setMuted(!autoPlay);
  }, [autoPlay]);

  useEffect(() => {
    micMutedRef.current = micMuted;
  }, [micMuted]);

  useEffect(() => {
    echoGuardRef.current = echoGuard;
  }, [echoGuard]);

  useEffect(() => {
    whisperBoostRef.current = whisperBoost;
    localStorage.setItem("lissan_whisper_boost", String(whisperBoost));
    micPipelineRef.current?.setBoost(whisperBoost);
  }, [whisperBoost]);

  useEffect(() => {
    liveCallActiveRef.current = liveCallActive;
    if (!liveCallActive) {
      setCallSeconds(0);
      return;
    }
    const id = setInterval(() => {
      setCallSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(id);
  }, [liveCallActive]);

  useEffect(() => {
    activeRoomRef.current = activeRoomCode;
  }, [activeRoomCode]);

  const wakeLockRef = useRef<{ release: () => Promise<void> } | null>(null);

  const stopLiveCall = useCallback(() => {
    if (wakeLockRef.current) {
      void wakeLockRef.current.release().catch(() => undefined);
      wakeLockRef.current = null;
    }
    if (micPipelineRef.current) {
      micPipelineRef.current.cleanup();
      micPipelineRef.current = null;
    }
    if (inputStreamRef.current) {
      inputStreamRef.current.getTracks().forEach((tr) => tr.stop());
      inputStreamRef.current = null;
    }
    setInputAnalyser(null);
    setLiveCallActive(false);
    setMicMuted(false);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "room:live_stop" }));
    }
  }, []);

  const ensureOutputPlayer = useCallback(() => {
    if (!playerRef.current) {
      const player = new GaplessPcmPlayer();
      player.setMuted(!autoPlayRef.current);
      const { analyser } = player.ensureContext();
      playerRef.current = player;
      setOutputAnalyser(analyser);
    }
    return playerRef.current;
  }, []);

  const startLiveCall = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setErrorMsg(null);
    try {
      ensureOutputPlayer();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
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

      const pipeline = await createBoostedMicPipeline(stream, whisperBoostRef.current);
      micPipelineRef.current = pipeline;
      setInputAnalyser(pipeline.analyser);

      wsRef.current.send(JSON.stringify({ type: "room:live_start" }));
      setLiveCallActive(true);
      setMicMuted(false);

      pipeline.processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        if (pipeline.ctx.state === "suspended") {
          void pipeline.ctx.resume();
        }
        if (micMutedRef.current) return;
        if (
          echoGuardRef.current &&
          Date.now() - lastIncomingAudioAtRef.current < 550
        ) {
          return;
        }
        const raw = e.inputBuffer.getChannelData(0);
        const pcm16k = downsampleTo16k(raw, pipeline.ctx.sampleRate);
        const base64 = float32ToPcm16Base64(pcm16k);
        wsRef.current.send(JSON.stringify({ type: "room:live_audio", data: base64 }));
      };
    } catch {
      setErrorMsg(t.errorMicDenied);
      stopLiveCall();
    }
  }, [ensureOutputPlayer, stopLiveCall, t.errorMicDenied]);

  const connectToRoomSocket = useCallback(
    (code: string) => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      manualLeaveRef.current = false;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const params = new URLSearchParams({
        roomCode: code,
        participantId,
        name: name.trim() || "Guest",
        lang: myLangCode,
        voice: myVoice,
      });
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/room?${params.toString()}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setErrorMsg(null);
        if (liveCallActiveRef.current) {
          ws.send(JSON.stringify({ type: "room:live_start" }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "room:init") {
            setParticipants(data.participants || []);
            setMessages(data.messages || []);
          } else if (data.type === "room:presence") {
            setParticipants(data.participants || []);
          } else if (data.type === "room:live_caption") {
            setLiveCaption((prev) => ({
              speakerName: data.speakerName || "",
              sourceText: data.text || "",
              translatedText:
                prev && prev.speakerName === data.speakerName ? prev.translatedText : "",
            }));
          } else if (data.type === "room:live_translated_caption") {
            setLiveCaption((prev) => ({
              speakerName: data.speakerName || prev?.speakerName || "",
              sourceText:
                prev && prev.speakerName === data.speakerName ? prev.sourceText : "",
              translatedText: data.text || "",
            }));
          } else if (data.type === "room:live_audio_out" && data.data) {
            lastIncomingAudioAtRef.current = Date.now();
            const player = ensureOutputPlayer();
            player.enqueuePcm24kBase64(data.data);
          } else if (data.type === "room:live_audio_wav" && data.data) {
            lastIncomingAudioAtRef.current = Date.now();
            const player = ensureOutputPlayer();
            void player.enqueueWavBase64(data.data);
          } else if (data.type === "room:fallback_speak" && data.text) {
            if (autoPlayRef.current) {
              speakBrowserFallback(data.text, data.lang || myLangCode);
            }
          } else if (data.type === "room:live_interrupted") {
            playerRef.current?.interrupt();
          } else if (data.type === "room:message" && data.message) {
            const incoming = data.message as RoomMessage;
            setIsSendingText(false);
            setLiveCaption(null);
            setMessages((prev) => {
              if (prev.some((m) => m.id === incoming.id)) return prev;
              return [...prev, incoming];
            });
            if (autoPlayRef.current && incoming.audioWavBase64) {
              void playWavBase64(incoming.audioWavBase64);
            } else if (
              autoPlayRef.current &&
              !incoming.liveStreamed &&
              !incoming.audioWavBase64 &&
              incoming.speakerId !== participantId &&
              incoming.translatedText
            ) {
              speakBrowserFallback(incoming.translatedText, incoming.targetLang || myLangCode);
            }
          } else if (data.type === "room:empty_speech") {
            setIsSendingText(false);
          } else if (data.type === "error") {
            setIsSendingText(false);
            setErrorMsg(data.message || t.errorGeneric);
          }
        } catch {
          // ignore malformed frames
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (!manualLeaveRef.current && activeRoomRef.current === code) {
          setTimeout(() => {
            if (!manualLeaveRef.current && activeRoomRef.current === code) {
              connectToRoomSocket(code);
            }
          }, 2000);
        }
      };
    },
    [ensureOutputPlayer, myLangCode, myVoice, name, participantId, t.errorGeneric]
  );

  useEffect(() => {
    return () => {
      manualLeaveRef.current = true;
      stopLiveCall();
      playerRef.current?.close();
      playerRef.current = null;
      wsRef.current?.close();
    };
  }, [stopLiveCall]);

  const handleCreateRoom = async () => {
    setErrorMsg(null);
    try {
      const res = await fetch("/api/rooms", { method: "POST" });
      const data = await res.json();
      if (!res.ok || !data.roomCode) {
        throw new Error(data.error || t.errorGeneric);
      }
      setActiveRoomCode(data.roomCode);
      const url = new URL(window.location.href);
      url.searchParams.set("room", data.roomCode);
      window.history.replaceState({}, "", url.toString());
      connectToRoomSocket(data.roomCode);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t.errorGeneric);
    }
  };

  const handleJoinRoom = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = joinCodeInput.toUpperCase().trim();
    if (!cleaned) return;
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/rooms/${encodeURIComponent(cleaned)}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || t.errorGeneric);
      }
      setActiveRoomCode(cleaned);
      const url = new URL(window.location.href);
      url.searchParams.set("room", cleaned);
      window.history.replaceState({}, "", url.toString());
      connectToRoomSocket(cleaned);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t.errorGeneric);
    }
  };

  const handleLeaveRoom = () => {
    manualLeaveRef.current = true;
    stopLiveCall();
    playerRef.current?.close();
    playerRef.current = null;
    setOutputAnalyser(null);
    wsRef.current?.close();
    wsRef.current = null;
    setActiveRoomCode(null);
    setParticipants([]);
    setMessages([]);
    setLiveCaption(null);
    setWsConnected(false);
    const url = new URL(window.location.href);
    url.searchParams.delete("room");
    window.history.replaceState({}, "", url.toString());
  };

  const handleLanguageChange = (newCode: string) => {
    setMyLangCode(newCode);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "participant:update",
          langCode: newCode,
        })
      );
    }
  };

  const handleVoiceChange = (newVoice: VoiceId) => {
    setMyVoice(newVoice);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "participant:update",
          voiceName: newVoice,
        })
      );
    }
  };

  const copyInvite = async () => {
    if (!activeRoomCode) return;
    const url = new URL(window.location.href);
    url.searchParams.set("room", activeRoomCode);
    await navigator.clipboard.writeText(url.toString());
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const shareInvite = async () => {
    if (!activeRoomCode) return;
    const url = new URL(window.location.href);
    url.searchParams.set("room", activeRoomCode);
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Lissan Room ${activeRoomCode}`,
          text:
            uiLang === "am"
              ? `በቀጥታ የድምፅ ትርጉም ክፍል (${activeRoomCode}) እንነጋገር`
              : `Join my live voice translation room (${activeRoomCode})`,
          url: url.toString(),
        });
        return;
      } catch {
        // fallback to clipboard
      }
    }
    await copyInvite();
  };

  const exportRoomTranscript = () => {
    if (messages.length === 0 || !activeRoomCode) return;
    const lines = messages.map((m) => {
      const timeStr = new Date(m.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      return `[${timeStr}] ${m.speakerName} (${m.sourceLang} -> ${m.targetLang})\n  ${m.translatedText}\n  (${m.sourceText})`;
    });
    const blob = new Blob(
      [`Lissan Room ${activeRoomCode}\n${"=".repeat(32)}\n\n${lines.join("\n\n")}\n`],
      { type: "text/plain;charset=utf-8" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lissan-room-${activeRoomCode}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatDuration = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const rem = sec % 60;
    return `${String(mins).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  };

  const handleSendText = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = textInput.trim();
    if (!cleaned || wsRef.current?.readyState !== WebSocket.OPEN) return;
    setIsSendingText(true);
    wsRef.current.send(
      JSON.stringify({
        type: "room:speak",
        messageId: `${participantId}_${Date.now()}`,
        text: cleaned,
      })
    );
    setTextInput("");
  };

  const myLang = getLanguageByCode(myLangCode);

  if (!activeRoomCode) {
    return (
      <div className="mx-auto max-w-4xl rounded-xl border border-slate-300 bg-white p-6 sm:p-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          <div className="space-y-4">
            <h2 className="font-display text-2xl font-extrabold text-slate-950">
              {t.roomCreateTitle}
            </h2>
            <p className="text-base font-semibold leading-relaxed text-slate-700">
              {t.roomSpeakHint}
            </p>
            <div className="space-y-3 pt-2">
              <label className="block text-sm font-bold text-slate-800">
                {t.yourName}
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="mt-1.5 h-11 w-full rounded-lg border border-slate-300 bg-slate-50 px-3.5 text-base font-bold text-slate-950 focus:border-blue-600 focus:bg-white focus:outline-none"
                />
              </label>
              <div>
                <span className="block text-sm font-bold text-slate-800">
                  {t.yourLang}
                </span>
                <button
                  type="button"
                  onClick={() => setPickerOpen(true)}
                  className="mt-1.5 flex h-11 w-full items-center justify-between rounded-lg border border-slate-300 bg-slate-50 px-3.5 text-left text-base font-bold text-slate-950 transition-colors hover:border-slate-400 hover:bg-white"
                >
                  <span className="truncate">
                    {myLang.nativeName} · {uiLang === "am" ? myLang.amharicName : myLang.name}
                  </span>
                  <ChevronDown className="h-4 w-4 shrink-0 text-slate-600" />
                </button>
              </div>
            </div>
            <button
              type="button"
              onClick={handleCreateRoom}
              className="mt-2 inline-flex h-12 w-full items-center justify-center rounded-lg bg-blue-700 px-5 text-base font-extrabold text-white transition-colors hover:bg-blue-800 whitespace-nowrap"
            >
              {t.createRoomBtn}
            </button>
          </div>

          <form
            onSubmit={handleJoinRoom}
            className="flex flex-col justify-between space-y-4 border-t border-slate-200 pt-6 md:border-l md:border-t-0 md:pl-8 md:pt-0"
          >
            <div className="space-y-4">
              <h2 className="font-display text-2xl font-extrabold text-slate-950">
                {t.roomJoinTitle}
              </h2>
              <p className="text-base font-semibold leading-relaxed text-slate-700">
                {uiLang === "am"
                  ? "ከሌላ ሰው የተላከልዎትን ባለ 6 ፊደል የክፍል ኮድ ያስገቡ (ወይም የግብዣ ሊንኩን ይክፈቱ)።"
                  : "Enter the 6-character room code shared by the other speaker, then start the hands-free live call."}
              </p>
              <label className="block text-sm font-bold text-slate-800">
                {t.roomCodePlaceholder}
                <input
                  type="text"
                  value={joinCodeInput}
                  onChange={(e) => setJoinCodeInput(e.target.value.toUpperCase())}
                  placeholder="AB12CD"
                  maxLength={10}
                  className="mt-1.5 h-11 w-full rounded-lg border border-slate-300 bg-slate-50 px-3.5 font-mono text-lg font-bold uppercase tracking-wider text-slate-950 focus:border-blue-600 focus:bg-white focus:outline-none"
                />
              </label>
            </div>
            <button
              type="submit"
              disabled={!joinCodeInput.trim()}
              className="inline-flex h-12 w-full items-center justify-center rounded-lg border border-slate-300 bg-white px-5 text-base font-extrabold text-slate-950 transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap"
            >
              {t.joinRoomBtn}
            </button>
          </form>
        </div>

        {errorMsg && (
          <div className="mt-5 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-base font-bold text-red-900">
            {errorMsg}
          </div>
        )}

        <LanguagePickerModal
          open={pickerOpen}
          uiLang={uiLang}
          selectedCode={myLangCode}
          onSelect={handleLanguageChange}
          onClose={() => setPickerOpen(false)}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      {/* Room Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-300 bg-white px-5 py-4">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <div className="text-xs font-bold text-slate-700">
              {uiLang === "am" ? "የክፍል ኮድ" : "Room Code"}
            </div>
            <div className="font-mono text-xl font-extrabold tracking-wider text-slate-950">
              {activeRoomCode}
            </div>
          </div>

          <div className="h-8 w-px bg-slate-300" aria-hidden="true" />

          <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                liveCallActive
                  ? "bg-emerald-600"
                  : wsConnected
                  ? "bg-blue-600"
                  : "bg-amber-500"
              }`}
              aria-hidden="true"
            />
            <span>
              {liveCallActive
                ? t.liveCallActive
                : wsConnected
                ? t.liveConnected
                : t.connectingLive}
            </span>
            {liveCallActive && (
              <span className="font-mono text-sm font-extrabold text-emerald-800 tabular-nums">
                · {formatDuration(callSeconds)}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => setPickerOpen(true)}
            className="inline-flex h-10 items-center gap-2 rounded-lg border border-blue-300 bg-blue-50 px-3.5 text-sm font-bold text-blue-950 transition-colors hover:bg-blue-100 whitespace-nowrap shrink-0"
          >
            <span>{myLang.nativeName}</span>
            <ChevronDown className="h-4 w-4 text-blue-800" />
          </button>

          <select
            aria-label={t.voiceLabel}
            value={myVoice}
            onChange={(e) => handleVoiceChange(e.target.value as VoiceId)}
            className="h-10 rounded-lg border border-slate-300 bg-slate-50 px-3 text-sm font-bold text-slate-950 focus:border-blue-600 focus:outline-none"
          >
            {VOICE_PROFILES.map((v) => (
              <option key={v.id} value={v.id}>
                {uiLang === "am" ? v.labelAm : `${v.label} (${v.tone})`}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={copyInvite}
            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap shrink-0"
          >
            {copiedLink ? (
              <Check className="h-4 w-4 text-emerald-700" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
            <span>{copiedLink ? t.copied : t.copyInviteLink}</span>
          </button>

          <button
            type="button"
            onClick={shareInvite}
            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap shrink-0"
          >
            <Share2 className="h-4 w-4" />
            <span>{t.shareInvite}</span>
          </button>

          {messages.length > 0 && (
            <button
              type="button"
              onClick={exportRoomTranscript}
              className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 transition-colors hover:bg-slate-50 whitespace-nowrap shrink-0"
            >
              <Download className="h-4 w-4" />
              <span>{t.exportTranscript}</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleLeaveRoom}
            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-red-700 transition-colors hover:bg-red-50 whitespace-nowrap shrink-0"
          >
            <LogOut className="h-4 w-4" />
            <span>{t.leaveRoom}</span>
          </button>
        </div>
      </div>

      {/* Main Split: Call Controls + Participants & Live Feed */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="space-y-4 rounded-xl border border-slate-300 bg-white p-5 lg:col-span-4">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3">
            <h3 className="flex items-center gap-2 font-display text-base font-extrabold text-slate-950">
              <Users className="h-4 w-4 text-slate-700" />
              <span>{t.participantsTitle}</span>
            </h3>
            <span className="font-mono text-sm font-bold text-slate-800 tabular-nums">
              {participants.length}
            </span>
          </div>

          <div className="divide-y divide-slate-200">
            {participants.map((p) => {
              const pLang = getLanguageByCode(p.langCode);
              const isMe = p.id === participantId;
              return (
                <div key={p.id} className="flex items-center justify-between py-3 text-base">
                  <div className="min-w-0">
                    <div className="truncate font-bold text-slate-950">
                      {p.name}
                      {isMe && (
                        <span className="ml-1.5 text-sm font-bold text-blue-700">
                          ({t.youTag})
                        </span>
                      )}
                    </div>
                    <div className="truncate text-sm font-semibold text-slate-700">
                      {pLang.nativeName} · {uiLang === "am" ? pLang.amharicName : pLang.name}
                    </div>
                  </div>
                  <span className="text-sm font-bold text-slate-800 shrink-0">
                    {p.liveCallActive
                      ? uiLang === "am"
                        ? "በጥሪ ላይ"
                        : "On Call"
                      : uiLang === "am"
                      ? "ዝግጁ"
                      : "Standby"}
                  </span>
                </div>
              );
            })}
            {participants.length < 2 && (
              <p className="pt-3 text-sm font-semibold leading-relaxed text-slate-700">
                {t.noParticipantsYet}
              </p>
            )}
          </div>

          {/* Hands-Free Live Call Primary Controls */}
          <div className="space-y-3 border-t border-slate-200 pt-4">
            {!liveCallActive ? (
              <button
                type="button"
                onClick={startLiveCall}
                disabled={!wsConnected}
                className="inline-flex h-14 w-full items-center justify-center gap-2 rounded-lg bg-emerald-700 px-4 text-base font-extrabold text-white transition-colors hover:bg-emerald-800 disabled:opacity-50"
              >
                <PhoneCall className="h-5 w-5" />
                <span>{t.startLiveCall}</span>
              </button>
            ) : (
              <div className="space-y-2.5">
                <button
                  type="button"
                  onClick={() => setMicMuted((m) => !m)}
                  className={`inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg border px-4 text-base font-extrabold transition-colors ${
                    micMuted
                      ? "border-amber-300 bg-amber-50 text-amber-950 hover:bg-amber-100"
                      : "border-slate-300 bg-white text-slate-950 hover:bg-slate-50"
                  }`}
                >
                  {micMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
                  <span>{micMuted ? t.unmuteMic : t.muteMic}</span>
                </button>
                <button
                  type="button"
                  onClick={stopLiveCall}
                  className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-red-600 px-4 text-base font-extrabold text-white transition-colors hover:bg-red-700"
                >
                  <PhoneOff className="h-5 w-5" />
                  <span>{t.endLiveCall}</span>
                </button>
              </div>
            )}

            <div className="space-y-2.5 pt-2 text-sm font-bold text-slate-800">
              <label className="flex cursor-pointer items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <Ear className="h-4 w-4 text-emerald-700" />
                  <span>{t.whisperBoostLabel}</span>
                </span>
                <input
                  type="checkbox"
                  checked={whisperBoost}
                  onChange={(e) => setWhisperBoost(e.target.checked)}
                  className="h-4 w-4 accent-emerald-700"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className="h-4 w-4 text-slate-700" />
                  <span>{echoGuard ? t.echoGuardLabel : t.fullDuplexLabel}</span>
                </span>
                <input
                  type="checkbox"
                  checked={echoGuard}
                  onChange={(e) => setEchoGuard(e.target.checked)}
                  className="h-4 w-4 accent-blue-700"
                />
              </label>

              <label className="flex cursor-pointer items-center justify-between">
                <span>{t.autoPlayAudio}</span>
                <input
                  type="checkbox"
                  checked={autoPlay}
                  onChange={(e) => setAutoPlay(e.target.checked)}
                  className="h-4 w-4 accent-blue-700"
                />
              </label>
            </div>
          </div>
        </div>

        {/* Live Captions, Waveforms & Transcript Feed */}
        <div className="flex flex-col justify-between rounded-xl border border-slate-300 bg-white lg:col-span-8">
          {/* Dual Audio Waveforms + Live Streaming Caption Bar */}
          <div className="space-y-3 border-b border-slate-200 bg-slate-50/70 p-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-slate-300 bg-white px-3.5 py-2.5">
                <div className="mb-1 text-xs font-extrabold text-slate-800">
                  {uiLang === "am" ? "የእርስዎ ድምፅ (Mic)" : "Your Mic"}
                </div>
                <WaveformCanvas
                  analyser={inputAnalyser}
                  active={liveCallActive && !micMuted}
                  accentColor="#15803D"
                />
              </div>
              <div className="rounded-lg border border-slate-300 bg-white px-3.5 py-2.5">
                <div className="mb-1 text-xs font-extrabold text-blue-800">
                  {uiLang === "am" ? "የተተረጎመ የሌላው ድምፅ (Incoming)" : "Incoming Translated Voice"}
                </div>
                <WaveformCanvas
                  analyser={outputAnalyser}
                  active={liveCallActive && Boolean(liveCaption?.translatedText)}
                  accentColor="#1D4ED8"
                />
              </div>
            </div>

            {liveCaption && (liveCaption.sourceText || liveCaption.translatedText) && (
              <div className="rounded-lg border border-blue-300 bg-blue-50/70 px-4 py-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-extrabold text-blue-900">
                    {t.liveCaptionsTitle} · {liveCaption.speakerName}
                  </div>
                  <button
                    type="button"
                    onClick={() => setLiveCaption(null)}
                    title={t.deleteItem}
                    aria-label={t.deleteItem}
                    className="inline-flex h-7 items-center gap-1 rounded-md border border-blue-200 bg-white px-2 text-xs font-bold text-slate-700 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>{t.deleteItem}</span>
                  </button>
                </div>
                {liveCaption.translatedText && (
                  <p className="mt-1 text-lg font-bold text-slate-950">
                    {liveCaption.translatedText}
                  </p>
                )}
                {liveCaption.sourceText && (
                  <p className="mt-1 text-sm font-semibold text-slate-700">{liveCaption.sourceText}</p>
                )}
              </div>
            )}
          </div>

          <div className="max-h-[340px] min-h-[220px] space-y-3 overflow-y-auto p-5">
            {messages.length === 0 ? (
              <div className="flex h-full min-h-[180px] flex-col items-center justify-center text-center">
                <p className="max-w-md text-base font-semibold leading-relaxed text-slate-700">
                  {t.roomSpeakHint}
                </p>
              </div>
            ) : (
              messages.map((m) => {
                const isMine = m.speakerId === participantId;
                const srcLang = getLanguageByCode(m.sourceLang);
                const tgtLang = getLanguageByCode(m.targetLang);
                const timeStr = new Date(m.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });
                return (
                  <div
                    key={m.id}
                    className={`rounded-lg border px-4 py-3.5 ${
                      isMine
                        ? "border-blue-300 bg-blue-50/50"
                        : "border-slate-300 bg-slate-50/80"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 text-sm font-bold text-slate-700">
                      <span className="truncate text-slate-900">
                        {m.speakerName} · {srcLang.nativeName} → {tgtLang.nativeName}
                      </span>
                      <div className="flex shrink-0 items-center gap-2">
                        <span className="font-mono tabular-nums">{timeStr}</span>
                        <button
                          type="button"
                          onClick={() =>
                            setMessages((prev) => prev.filter((item) => item.id !== m.id))
                          }
                          title={t.deleteItem}
                          aria-label={t.deleteItem}
                          className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 text-xs font-bold text-slate-700 transition-colors hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          <span>{t.deleteItem}</span>
                        </button>
                      </div>
                    </div>
                    <p className="mt-1.5 text-lg font-bold leading-relaxed text-slate-950">
                      {m.translatedText}
                    </p>
                    {m.sourceText && m.sourceText !== m.translatedText && (
                      <p className="mt-1 text-sm font-semibold leading-relaxed text-slate-700">
                        {m.sourceText}
                      </p>
                    )}
                    {m.audioWavBase64 && (
                      <div className="mt-2">
                        <button
                          type="button"
                          onClick={() => void playWavBase64(m.audioWavBase64!)}
                          className="inline-flex items-center gap-1.5 text-sm font-bold text-blue-800 hover:text-blue-950"
                        >
                          <Volume2 className="h-4 w-4" />
                          <span>{t.replayAudio}</span>
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>

          <div className="border-t border-slate-200 bg-slate-50/70 p-4">
            {errorMsg && (
              <div className="mb-3 rounded-lg border border-red-300 bg-red-50 px-3.5 py-2.5 text-sm font-bold text-red-900">
                {errorMsg}
              </div>
            )}

            <form onSubmit={handleSendText} className="flex items-center gap-2">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder={t.typeToSpeak}
                className="h-11 min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-3.5 text-base font-semibold text-slate-950 placeholder:text-slate-500 focus:border-blue-600 focus:outline-none"
              />
              <button
                type="submit"
                disabled={!textInput.trim() || isSendingText || !wsConnected}
                aria-label={t.sendAndSpeak}
                className="inline-flex h-11 items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white px-4 text-sm font-bold text-slate-950 transition-colors hover:bg-slate-100 disabled:opacity-50 whitespace-nowrap shrink-0"
              >
                <Send className="h-4 w-4" />
                <span className="hidden sm:inline">{t.sendAndSpeak}</span>
              </button>
            </form>
          </div>
        </div>
      </div>

      <LanguagePickerModal
        open={pickerOpen}
        uiLang={uiLang}
        selectedCode={myLangCode}
        onSelect={handleLanguageChange}
        onClose={() => setPickerOpen(false)}
      />
    </div>
  );
}
