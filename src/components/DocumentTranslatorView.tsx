import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  Copy,
  Download,
  FileText,
  Loader2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Square,
  Trash2,
  Upload,
  Volume2,
} from "lucide-react";
import { UI_COPY, UiLang } from "../data/i18n";
import { getLanguageByCode, VOICE_PROFILES, VoiceId } from "../data/languages";
import {
  blobToBase64,
  GaplessPcmPlayer,
  speakBrowserFallback,
} from "../utils/audio";
import { LanguagePickerModal } from "./LanguagePickerModal";
import { WaveformCanvas } from "./WaveformCanvas";

interface DocumentTranslatorViewProps {
  uiLang: UiLang;
  defaultSourceLang: string;
  defaultTargetLang: string;
  voiceName: VoiceId;
  onVoiceChange: (voice: VoiceId) => void;
  lowBandwidth: boolean;
}

const MAX_FILE_BYTES = 20 * 1024 * 1024;
const SPEED_OPTIONS = [0.85, 1.0, 1.15, 1.3] as const;

export function DocumentTranslatorView({
  uiLang,
  defaultSourceLang,
  defaultTargetLang,
  voiceName,
  onVoiceChange,
  lowBandwidth,
}: DocumentTranslatorViewProps) {
  const t = UI_COPY[uiLang];

  const [sourceLang, setSourceLang] = useState<string>("auto");
  const [targetLang, setTargetLang] = useState<string>(defaultTargetLang || "am");
  const [pickerTarget, setPickerTarget] = useState<"source" | "target" | null>(null);

  const [selectedFile, setSelectedFile] = useState<{
    name: string;
    mimeType: string;
    base64: string;
    size: number;
  } | null>(null);
  const [pastedText, setPastedText] = useState<string>("");

  const [translating, setTranslating] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [sourceText, setSourceText] = useState<string>("");
  const [translatedText, setTranslatedText] = useState<string>("");
  const [sections, setSections] = useState<string[]>([]);
  const [detectedLanguage, setDetectedLanguage] = useState<string>("");
  const [copied, setCopied] = useState<boolean>(false);

  // Full report voice playback state
  const [playbackState, setPlaybackState] = useState<"idle" | "playing" | "paused">("idle");
  const [currentSectionIdx, setCurrentSectionIdx] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [outputAnalyser, setOutputAnalyser] = useState<AnalyserNode | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const playerRef = useRef<GaplessPcmPlayer | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const runTokenRef = useRef<number>(0);
  const playbackSpeedRef = useRef<number>(playbackSpeed);
  const lowBandwidthRef = useRef<boolean>(lowBandwidth);
  const voiceNameRef = useRef<VoiceId>(voiceName);

  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
  }, [playbackSpeed]);

  useEffect(() => {
    lowBandwidthRef.current = lowBandwidth;
  }, [lowBandwidth]);

  useEffect(() => {
    voiceNameRef.current = voiceName;
  }, [voiceName]);

  useEffect(() => {
    if (defaultSourceLang && defaultSourceLang !== "auto" && sourceLang === "auto") {
      // Keep auto-detect as default for documents so mixed reports translate smoothly
    }
  }, [defaultSourceLang, sourceLang]);

  const stopReading = useCallback(() => {
    runTokenRef.current += 1;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (playerRef.current) {
      playerRef.current.interrupt();
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }
    setPlaybackState("idle");
  }, []);

  useEffect(() => {
    return () => {
      stopReading();
      if (playerRef.current) {
        playerRef.current.close();
        playerRef.current = null;
      }
    };
  }, [stopReading]);

  const handleFilePick = async (file: File | undefined) => {
    if (!file) return;
    setErrorMsg(null);
    if (file.size > MAX_FILE_BYTES) {
      setErrorMsg(t.docFileTooLarge);
      return;
    }

    const lower = file.name.toLowerCase();
    const mime = (file.type || "").toLowerCase();
    const isSupported =
      mime === "application/pdf" ||
      mime.startsWith("image/") ||
      mime.startsWith("text/") ||
      mime.includes("wordprocessingml") ||
      /\.(pdf|docx|txt|md|csv|tsv|json|html|xml|log|srt|vtt|png|jpe?g|webp|heic|heif)$/.test(
        lower
      );

    if (!isSupported) {
      setErrorMsg(t.docUnsupportedFile);
      return;
    }

    const base64 = await blobToBase64(file);
    setSelectedFile({
      name: file.name,
      mimeType: file.type || "application/octet-stream",
      base64,
      size: file.size,
    });
  };

  const handleTranslateDocument = async () => {
    if (!selectedFile && !pastedText.trim()) {
      setErrorMsg(t.docEmptyError);
      return;
    }
    stopReading();
    setTranslating(true);
    setErrorMsg(null);

    try {
      const res = await fetch("/api/translate-document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: pastedText.trim() || undefined,
          fileName: selectedFile?.name,
          mimeType: selectedFile?.mimeType,
          fileBase64: selectedFile?.base64,
          sourceLang,
          targetLang,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || t.errorGeneric);
      }
      setDetectedLanguage(data.detectedLanguage || "");
      setSourceText(data.sourceText || "");
      setTranslatedText(data.translatedText || "");
      const parsedSections = Array.isArray(data.sections)
        ? data.sections
        : data.translatedText
        ? [data.translatedText]
        : [];
      setSections(parsedSections);
      setCurrentSectionIdx(0);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t.errorGeneric);
    } finally {
      setTranslating(false);
    }
  };

  const speakSingleSection = useCallback(
    async (
      text: string,
      token: number,
      signal: AbortSignal
    ): Promise<void> => {
      if (!playerRef.current) {
        playerRef.current = new GaplessPcmPlayer();
      }
      const player = playerRef.current;
      player.setMuted(false);
      player.interrupt();
      const { analyser } = player.ensureContext();
      setOutputAnalyser(analyser);

      const res = await fetch("/api/tts-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voiceName: voiceNameRef.current,
          lowBandwidth: lowBandwidthRef.current,
        }),
        signal,
      });

      if (token !== runTokenRef.current || signal.aborted) return;

      const sampleRateHeader = Number(res.headers.get("X-Sample-Rate"));
      const sampleRate =
        sampleRateHeader === 12000 || lowBandwidthRef.current ? 12000 : 24000;

      if (!res.ok || res.status === 204 || !res.body) {
        speakBrowserFallback(text, targetLang);
        await new Promise((r) => setTimeout(r, Math.max(1500, text.length * 65)));
        return;
      }

      const reader = res.body.getReader();
      let leftover: number | null = null;
      let receivedAny = false;

      while (true) {
        if (token !== runTokenRef.current || signal.aborted) return;
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
          player.enqueuePcmBytes(bytes, sampleRate, playbackSpeedRef.current);
        }
      }

      if (!receivedAny) {
        speakBrowserFallback(text, targetLang);
        await new Promise((r) => setTimeout(r, Math.max(1500, text.length * 65)));
        return;
      }

      await player.waitUntilDrained(signal);
    },
    [targetLang]
  );

  const playFromIndex = useCallback(
    async (startIdx: number) => {
      if (sections.length === 0) return;
      stopReading();

      const token = ++runTokenRef.current;
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setPlaybackState("playing");

      try {
        for (let i = Math.max(0, startIdx); i < sections.length; i++) {
          if (token !== runTokenRef.current || controller.signal.aborted) break;
          setCurrentSectionIdx(i);
          const sectionText = sections[i];
          if (!sectionText.trim()) continue;
          await speakSingleSection(sectionText, token, controller.signal);
        }
        if (token === runTokenRef.current && !controller.signal.aborted) {
          setPlaybackState("idle");
        }
      } catch (err) {
        if ((err as Error)?.name !== "AbortError" && token === runTokenRef.current) {
          setPlaybackState("idle");
        }
      }
    },
    [sections, speakSingleSection, stopReading]
  );

  const handlePauseReading = () => {
    runTokenRef.current += 1;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    playerRef.current?.interrupt();
    setPlaybackState("paused");
  };

  const handleClearAll = () => {
    stopReading();
    setSelectedFile(null);
    setPastedText("");
    setSourceText("");
    setTranslatedText("");
    setSections([]);
    setDetectedLanguage("");
    setCurrentSectionIdx(0);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleCopyTranslation = async () => {
    if (!translatedText) return;
    await navigator.clipboard.writeText(translatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownloadTranslation = () => {
    if (!translatedText) return;
    const blob = new Blob([translatedText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lissan-document-${targetLang}-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const sourceObj = sourceLang === "auto" ? null : getLanguageByCode(sourceLang);
  const targetObj = getLanguageByCode(targetLang);

  return (
    <div className="space-y-5">
      {/* Header & Configuration Card */}
      <div className="rounded-xl border border-slate-300 bg-white p-6 space-y-5">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-start">
          <div className="max-w-2xl space-y-1.5">
            <h2 className="flex items-center gap-2 font-display text-2xl font-extrabold text-slate-950">
              <FileText className="h-6 w-6 text-blue-700" />
              <span>{t.docTitle}</span>
            </h2>
            <p className="text-base font-semibold leading-relaxed text-slate-700">
              {t.docSubtitle}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => setPickerTarget("source")}
              className="inline-flex h-11 items-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-3.5 text-sm font-bold text-slate-950 hover:bg-white whitespace-nowrap"
            >
              <span>
                {sourceObj
                  ? `${sourceObj.nativeName}`
                  : t.autoDetect}
              </span>
              <ChevronDown className="h-4 w-4 text-slate-600" />
            </button>

            <span className="font-bold text-slate-500">→</span>

            <button
              type="button"
              onClick={() => setPickerTarget("target")}
              className="inline-flex h-11 items-center gap-2 rounded-lg border border-blue-300 bg-blue-50 px-3.5 text-sm font-bold text-blue-950 hover:bg-blue-100 whitespace-nowrap"
            >
              <span>
                {targetObj.nativeName} ·{" "}
                {uiLang === "am" ? targetObj.amharicName : targetObj.name}
              </span>
              <ChevronDown className="h-4 w-4 text-blue-800" />
            </button>

            <select
              aria-label={t.voiceLabel}
              value={voiceName}
              onChange={(e) => onVoiceChange(e.target.value as VoiceId)}
              className="h-11 rounded-lg border border-slate-300 bg-slate-50 px-3 text-sm font-bold text-slate-950 focus:border-blue-600 focus:outline-none"
            >
              {VOICE_PROFILES.map((v) => (
                <option key={v.id} value={v.id}>
                  {uiLang === "am" ? v.labelAm : `${v.label} · ${v.tone}`}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Upload Zone + Paste Textarea */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="lg:col-span-5">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.md,.csv,.tsv,.json,.html,.xml,.log,.srt,.vtt,image/*"
              onChange={(e) => void handleFilePick(e.target.files?.[0])}
              className="hidden"
              id="lissan-doc-upload"
            />
            <label
              htmlFor="lissan-doc-upload"
              className="flex h-full min-h-[168px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/80 p-5 text-center transition-colors hover:border-blue-500 hover:bg-blue-50/40"
            >
              <Upload className="h-7 w-7 text-blue-700" />
              <span className="mt-2.5 text-base font-extrabold text-slate-950">
                {t.docUploadBtn}
              </span>
              <span className="mt-1 text-sm font-semibold text-slate-600">
                {t.docUploadHint}
              </span>
              {selectedFile && (
                <div className="mt-3 flex items-center gap-2 rounded-lg border border-blue-300 bg-white px-3 py-1.5 text-sm font-bold text-blue-900">
                  <span className="max-w-[200px] truncate">{selectedFile.name}</span>
                  <span className="text-xs text-slate-500">
                    ({Math.max(1, Math.round(selectedFile.size / 1024))} KB)
                  </span>
                </div>
              )}
            </label>
          </div>

          <div className="lg:col-span-7 flex flex-col gap-3">
            <textarea
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder={t.docPastePlaceholder}
              rows={5}
              className="min-h-[168px] w-full rounded-xl border border-slate-300 bg-slate-50 p-4 text-base font-semibold leading-relaxed text-slate-950 placeholder:text-slate-500 focus:border-blue-600 focus:bg-white focus:outline-none"
            />
          </div>
        </div>

        {errorMsg && (
          <div
            role="alert"
            className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-base font-bold text-red-900"
          >
            {errorMsg}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => void handleTranslateDocument()}
            disabled={translating || (!selectedFile && !pastedText.trim())}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-blue-700 px-6 text-base font-extrabold text-white transition-colors hover:bg-blue-800 disabled:opacity-50"
          >
            {translating ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <FileText className="h-5 w-5" />
            )}
            <span>{translating ? t.docTranslating : t.docTranslateBtn}</span>
          </button>

          {(selectedFile || pastedText || translatedText) && (
            <button
              type="button"
              onClick={handleClearAll}
              className="inline-flex h-11 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-4 text-sm font-bold text-slate-800 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
            >
              <Trash2 className="h-4 w-4" />
              <span>{t.deleteItem}</span>
            </button>
          )}
        </div>
      </div>

      {/* Full Report Voice Reader + Side-by-Side Document View */}
      {translatedText && (
        <div className="space-y-4">
          {/* Continuous Voice Reader Controls */}
          <div className="rounded-xl border border-blue-300 bg-white p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-2.5">
                {playbackState === "playing" ? (
                  <button
                    type="button"
                    onClick={handlePauseReading}
                    className="inline-flex h-12 items-center gap-2 rounded-lg bg-amber-600 px-5 text-base font-extrabold text-white hover:bg-amber-700"
                  >
                    <Pause className="h-5 w-5" />
                    <span>{t.docPauseAudio}</span>
                  </button>
                ) : playbackState === "paused" ? (
                  <button
                    type="button"
                    onClick={() => void playFromIndex(currentSectionIdx)}
                    className="inline-flex h-12 items-center gap-2 rounded-lg bg-emerald-700 px-5 text-base font-extrabold text-white hover:bg-emerald-800"
                  >
                    <Play className="h-5 w-5" />
                    <span>{t.docResumeAudio}</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => void playFromIndex(0)}
                    className="inline-flex h-12 items-center gap-2 rounded-lg bg-emerald-700 px-5 text-base font-extrabold text-white hover:bg-emerald-800"
                  >
                    <Volume2 className="h-5 w-5" />
                    <span>{t.docReadFullBtn}</span>
                  </button>
                )}

                {playbackState !== "idle" && (
                  <button
                    type="button"
                    onClick={stopReading}
                    className="inline-flex h-12 items-center gap-1.5 rounded-lg bg-red-600 px-4 text-sm font-extrabold text-white hover:bg-red-700"
                  >
                    <Square className="h-4 w-4 fill-current" />
                    <span>{t.docStopAudio}</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => void playFromIndex(Math.max(0, currentSectionIdx - 1))}
                  disabled={sections.length === 0 || currentSectionIdx === 0}
                  title={t.docPrevSection}
                  aria-label={t.docPrevSection}
                  className="inline-flex h-11 items-center gap-1 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-50 disabled:opacity-40"
                >
                  <SkipBack className="h-4 w-4" />
                  <span className="hidden sm:inline">{t.docPrevSection}</span>
                </button>

                <button
                  type="button"
                  onClick={() =>
                    void playFromIndex(Math.min(sections.length - 1, currentSectionIdx + 1))
                  }
                  disabled={sections.length === 0 || currentSectionIdx >= sections.length - 1}
                  title={t.docNextSection}
                  aria-label={t.docNextSection}
                  className="inline-flex h-11 items-center gap-1 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-50 disabled:opacity-40"
                >
                  <span className="hidden sm:inline">{t.docNextSection}</span>
                  <SkipForward className="h-4 w-4" />
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2.5">
                <label className="flex items-center gap-2 text-sm font-bold text-slate-800">
                  <span>{t.docSpeedLabel}:</span>
                  <select
                    value={playbackSpeed}
                    onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                    className="h-10 rounded-lg border border-slate-300 bg-slate-50 px-2.5 text-sm font-bold text-slate-950"
                  >
                    {SPEED_OPTIONS.map((sp) => (
                      <option key={sp} value={sp}>
                        {sp}x
                      </option>
                    ))}
                  </select>
                </label>

                <button
                  type="button"
                  onClick={() => void handleCopyTranslation()}
                  className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-50"
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-emerald-700" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                  <span>{copied ? t.copied : t.copyText}</span>
                </button>

                <button
                  type="button"
                  onClick={handleDownloadTranslation}
                  className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 text-sm font-bold text-slate-900 hover:bg-slate-50"
                >
                  <Download className="h-4 w-4" />
                  <span>{t.docExportBtn}</span>
                </button>
              </div>
            </div>

            {/* Section Progress & Live Reader Waveform */}
            <div className="grid grid-cols-1 items-center gap-4 border-t border-slate-200 pt-3 sm:grid-cols-12">
              <div className="sm:col-span-5 space-y-1.5">
                <div className="flex items-center justify-between text-sm font-bold text-slate-800">
                  <span>
                    {t.docProgressLabel}: {Math.min(sections.length, currentSectionIdx + 1)} /{" "}
                    {sections.length}
                  </span>
                  {detectedLanguage && (
                    <span className="text-slate-600">{detectedLanguage} → {targetObj.nativeName}</span>
                  )}
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full bg-emerald-600 transition-all"
                    style={{
                      width: `${
                        sections.length > 0
                          ? Math.round(((currentSectionIdx + 1) / sections.length) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>
              <div className="sm:col-span-7">
                <WaveformCanvas
                  analyser={outputAnalyser}
                  active={playbackState === "playing"}
                  accentColor="#15803D"
                />
              </div>
            </div>
          </div>

          {/* Original vs Translated Full Report Columns */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-300 bg-white p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                <h3 className="font-display text-base font-extrabold text-slate-900">
                  {t.docOriginalTitle}
                </h3>
                {sourceText && (
                  <button
                    type="button"
                    onClick={() => setSourceText("")}
                    className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 text-xs font-bold text-slate-700 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>{t.deleteItem}</span>
                  </button>
                )}
              </div>
              <div className="max-h-[480px] overflow-y-auto whitespace-pre-wrap text-base font-semibold leading-relaxed text-slate-800">
                {sourceText}
              </div>
            </div>

            <div className="rounded-xl border border-blue-300 bg-white p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                <h3 className="font-display text-base font-extrabold text-blue-900">
                  {t.docTranslatedTitle} ({targetObj.nativeName})
                </h3>
                <button
                  type="button"
                  onClick={() => {
                    stopReading();
                    setTranslatedText("");
                    setSections([]);
                  }}
                  className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 text-xs font-bold text-slate-700 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>{t.deleteItem}</span>
                </button>
              </div>
              <div className="max-h-[480px] space-y-2.5 overflow-y-auto pr-1">
                {sections.map((sec, idx) => {
                  const isActive = playbackState !== "idle" && idx === currentSectionIdx;
                  return (
                    <button
                      key={`${idx}_${sec.slice(0, 16)}`}
                      type="button"
                      onClick={() => void playFromIndex(idx)}
                      className={`block w-full rounded-lg border px-4 py-3 text-left text-lg font-bold leading-relaxed transition-colors ${
                        isActive
                          ? "border-emerald-500 bg-emerald-50 text-slate-950"
                          : "border-slate-200 bg-slate-50/70 text-slate-950 hover:border-blue-300 hover:bg-blue-50/40"
                      }`}
                    >
                      <span className="mr-2 font-mono text-xs font-extrabold text-blue-800">
                        [{idx + 1}]
                      </span>
                      {sec}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      <LanguagePickerModal
        open={pickerTarget !== null}
        uiLang={uiLang}
        selectedCode={pickerTarget === "source" ? sourceLang : targetLang}
        allowAuto={pickerTarget === "source"}
        onSelect={(code) => {
          if (pickerTarget === "source") {
            setSourceLang(code);
          } else if (pickerTarget === "target") {
            setTargetLang(code);
          }
        }}
        onClose={() => setPickerTarget(null)}
      />
    </div>
  );
}
