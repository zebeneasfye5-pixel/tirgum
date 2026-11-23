import "dotenv/config";
import express from "express";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import { WebSocketServer, WebSocket } from "ws";
import {
  GoogleGenAI,
  LiveServerMessage,
  Modality,
  ThinkingLevel,
  Type,
} from "@google/genai";
import mammoth from "mammoth";
import { WORLD_LANGUAGES, VOICE_PROFILES, getLanguageByCode } from "./src/data/languages.ts";

const PRECISION_VOICE_LANGS = new Set(["am", "ti", "om", "so", "ha", "yo", "ig", "zu", "af"]);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const VALID_VOICES = new Set<string>(VOICE_PROFILES.map((v) => v.id));
const VALID_LANG_CODES = new Set<string>(WORLD_LANGUAGES.map((l) => l.code));

function sanitizeVoice(voice?: string): string {
  if (voice && VALID_VOICES.has(voice)) return voice;
  return "Zephyr";
}

function sanitizeLangCode(code?: string, allowAuto = false): string {
  if (allowAuto && code === "auto") return "auto";
  if (code && VALID_LANG_CODES.has(code)) return code;
  return "en";
}

function getAiClient(apiVersion?: string): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured on the server.");
  }
  return new GoogleGenAI({
    apiKey,
    ...(apiVersion ? { apiVersion } : {}),
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function extractErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  if (err && typeof err === "object") {
    const rec = err as Record<string, unknown>;
    if (typeof rec.message === "string" && rec.message) return rec.message;
    if (rec.error) {
      const nested = extractErrorMessage(rec.error);
      if (nested && nested !== "[object Object]") return nested;
    }
    try {
      const json = JSON.stringify(err);
      if (json && json !== "{}") return json;
    } catch {
      // ignore
    }
  }
  return String(err || "Unknown error");
}

function isFatalAuthError(msg: string): boolean {
  const upper = msg.toUpperCase();
  return (
    upper.includes("API_KEY_INVALID") ||
    upper.includes("PERMISSION_DENIED") ||
    upper.includes("GEMINI_API_KEY IS NOT CONFIGURED") ||
    (upper.includes("403") && upper.includes("API"))
  );
}

function isRetryableQuotaOrLoadError(err: unknown): boolean {
  const msg = extractErrorMessage(err).toUpperCase();
  return (
    msg.includes("503") ||
    msg.includes("UNAVAILABLE") ||
    msg.includes("HIGH DEMAND") ||
    msg.includes("429") ||
    msg.includes("RESOURCE_EXHAUSTED") ||
    msg.includes("RATE_LIMIT") ||
    msg.includes("QUOTA")
  );
}

function formatApiError(err: unknown): { status: number; message: string } {
  const msg = extractErrorMessage(err);
  if (isFatalAuthError(msg)) {
    return {
      status: 403,
      message: "Gemini API key permission or configuration error. Please check Settings > Secrets.",
    };
  }
  if (msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED")) {
    return {
      status: 429,
      message: "High traffic right now — please wait a second and speak again.",
    };
  }
  if (msg.includes("503") || msg.includes("UNAVAILABLE")) {
    return {
      status: 503,
      message: "Translation service is temporarily busy. Retrying automatically...",
    };
  }
  return { status: 500, message: msg || "Translation request failed." };
}

function computePcm16Rms(pcmBuffer: Buffer): number {
  const sampleCount = Math.floor(pcmBuffer.length / 2);
  if (sampleCount === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < sampleCount; i++) {
    const sample = pcmBuffer.readInt16LE(i * 2) / 32768;
    sumSq += sample * sample;
  }
  return Math.sqrt(sumSq / sampleCount);
}

function normalizePcm16Buffer(pcmBuffer: Buffer): Buffer {
  const sampleCount = Math.floor(pcmBuffer.length / 2);
  if (sampleCount === 0) return pcmBuffer;

  let peak = 0;
  for (let i = 0; i < sampleCount; i++) {
    const abs = Math.abs(pcmBuffer.readInt16LE(i * 2));
    if (abs > peak) peak = abs;
  }

  // Only normalize genuinely quiet/whispered captures (< 7,500 peak) with gentle max 4.5x gain
  // so normal Amharic ejectives & consonants never clip or distort.
  if (peak < 90 || peak >= 7500) {
    return pcmBuffer;
  }

  const gain = Math.min(5.0, 18500 / peak);
  const out = Buffer.allocUnsafe(pcmBuffer.length);
  for (let i = 0; i < sampleCount; i++) {
    const boosted = Math.round(pcmBuffer.readInt16LE(i * 2) * gain);
    const clamped = Math.max(-32768, Math.min(32767, boosted));
    out.writeInt16LE(clamped, i * 2);
  }
  return out;
}

/**
 * Downsamples 24kHz PCM16LE mono buffer 2:1 to 12kHz with a 2-tap averaging
 * anti-alias filter, cutting mobile network bandwidth in half (50% savings).
 */
function downsample24kTo12kPcm16(pcm24k: Buffer): Buffer {
  const sampleCount = Math.floor(pcm24k.length / 2);
  const outSamples = Math.floor(sampleCount / 2);
  if (outSamples <= 0) return Buffer.alloc(0);
  const out = Buffer.allocUnsafe(outSamples * 2);
  for (let i = 0; i < outSamples; i++) {
    const s0 = pcm24k.readInt16LE(i * 4);
    const s1 = pcm24k.readInt16LE(i * 4 + 2);
    out.writeInt16LE((s0 + s1) >> 1, i * 2);
  }
  return out;
}

function pcm16ChunksToWavBase64(chunks: Buffer[], sampleRate = 16000): string {
  const rawPcm = Buffer.concat(chunks);
  const pcmData = normalizePcm16Buffer(rawPcm);
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * 2;
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcmData.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcmData.length, 40);
  return Buffer.concat([header, pcmData]).toString("base64");
}

interface TranslatedTurnResult {
  sourceTranscript: string;
  detectedLanguage: string;
  translatedText: string;
  resolvedSourceLang?: string;
  resolvedTargetLang?: string;
}

async function transcribeAndTranslate(params: {
  audioBase64?: string;
  mimeType?: string;
  text?: string;
  sourceLangCode: string;
  targetLangCode: string;
  bidirectional?: boolean;
}): Promise<TranslatedTurnResult> {
  const ai = getAiClient();
  const targetLang = getLanguageByCode(params.targetLangCode);
  const sourceLang =
    params.sourceLangCode === "auto" ? null : getLanguageByCode(params.sourceLangCode);
  const isBidirectional =
    Boolean(params.bidirectional) && sourceLang !== null && sourceLang.code !== targetLang.code;
  const involvesEthiopic =
    params.sourceLangCode === "am" ||
    params.targetLangCode === "am" ||
    params.sourceLangCode === "ti" ||
    params.targetLangCode === "ti";

  const sourceHint = isBidirectional
    ? `This is a two-way conversation between ${sourceLang.name} (${sourceLang.nativeName}) [code: ${sourceLang.code}] and ${targetLang.name} (${targetLang.nativeName}) [code: ${targetLang.code}]. Detect which of these two languages was used: if it is ${sourceLang.name}, translate into ${targetLang.name}; if it is ${targetLang.name}, translate into ${sourceLang.name}.`
    : sourceLang === null
    ? `Auto-detect the spoken language and translate into ${targetLang.name} (${targetLang.nativeName}).`
    : `The speaker is speaking ${sourceLang.name} (${sourceLang.nativeName}). Translate into ${targetLang.name} (${targetLang.nativeName}).`;

  const ethiopicRules = involvesEthiopic
    ? `
ETHIOPIC / AMHARIC PRECISION RULES:
- For Amharic (አማርኛ) or Tigrinya (ትግርኛ), transcribe and translate strictly in authentic Ethiopic/Ge'ez Fidel script (ፊደል), using real, standard words and exact vowel orders (ግዕዝ፣ ካዕብ፣ ሣልስ፣ ራብዕ፣ ኃምስ፣ ሳድስ፣ ሳብዕ).
- Carefully distinguish ejective vs plain consonants (ተ/ጠ, ቸ/ጨ, ፐ/ጰ, ቀ/ከ, ሰ/ሸ/ጸ) from the speech acoustics—never substitute random look-alike Fidel characters or garbled syllables.`
    : "";

  const promptText = params.audioBase64
    ? `Listen closely to the audio—including soft, whispered, or distant speech. ${sourceHint}${ethiopicRules}
1) Transcribe EVERY single spoken word accurately in its original language and native script. Do NOT invent words from background noise; if there is no clear speech, return empty strings for all fields.
2) Translate the full meaning accurately, naturally, and completely into the destination language with zero omitted words.
Return strict JSON.`
    : `${sourceHint}${ethiopicRules}
Translate every word completely, accurately, and naturally with zero omissions.
Text to translate: """${params.text ?? ""}"""
Return strict JSON with sourceTranscript equal to the original text, detectedLanguage, resolvedSourceLang (BCP-47 code), resolvedTargetLang (BCP-47 code), and translatedText.`;

  const parts: Array<{ inlineData?: { mimeType: string; data: string }; text?: string }> = [];
  if (params.audioBase64) {
    parts.push({
      inlineData: {
        mimeType: params.mimeType || "audio/webm",
        data: params.audioBase64,
      },
    });
  }
  parts.push({ text: promptText });

  const callModel = async (modelName: string, thinkingLevel: ThinkingLevel) => {
    return ai.models.generateContent({
      model: modelName,
      contents: { parts },
      config: {
        thinkingConfig: { thinkingLevel },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            sourceTranscript: {
              type: Type.STRING,
              description: "Exact word-for-word transcription in original script.",
            },
            detectedLanguage: {
              type: Type.STRING,
              description: "Name of the detected source language.",
            },
            resolvedSourceLang: {
              type: Type.STRING,
              description: "BCP-47 code of the detected input language.",
            },
            resolvedTargetLang: {
              type: Type.STRING,
              description: "BCP-47 code of the output translated language.",
            },
            translatedText: {
              type: Type.STRING,
              description: "Complete, crystal-clear translation preserving every word.",
            },
          },
          required: ["sourceTranscript", "detectedLanguage", "translatedText"],
        },
      },
    });
  };

  let response;
  try {
    response = await callModel("gemini-4-hydrogen", ThinkingLevel.LOW);
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err)) {
      try {
        response = await callModel("gemini-3.1-flash-lite", ThinkingLevel.MINIMAL);
      } catch (err2) {
        if (isRetryableQuotaOrLoadError(err2)) {
          await new Promise((r) => setTimeout(r, 600));
          response = await callModel("gemini-4-hydrogen", ThinkingLevel.LOW);
        } else {
          throw err2;
        }
      }
    } else {
      throw err;
    }
  }

  const raw = (response.text ?? "{}").trim();
  const parsed = JSON.parse(raw) as Partial<TranslatedTurnResult>;
  return {
    sourceTranscript: (parsed.sourceTranscript ?? "").trim(),
    detectedLanguage: (parsed.detectedLanguage ?? "").trim(),
    translatedText: (parsed.translatedText ?? "").trim(),
    resolvedSourceLang: (parsed.resolvedSourceLang ?? "").trim() || undefined,
    resolvedTargetLang: (parsed.resolvedTargetLang ?? "").trim() || undefined,
  };
}

function parseStreamingTaggedOutput(rawText: string): TranslatedTurnResult {
  const clean = rawText.trim();
  if (!clean || clean === "NONE" || clean.startsWith("NONE\n")) {
    return { sourceTranscript: "", detectedLanguage: "", translatedText: "" };
  }
  let resolvedSourceLang: string | undefined;
  let resolvedTargetLang: string | undefined;
  let detectedLanguage = "";

  const langMatch = clean.match(/^LANG:\s*([^\n]+)/m);
  if (langMatch) {
    const parts = langMatch[1].split("|").map((s) => s.trim());
    if (parts[0] && parts[0] !== "none") resolvedSourceLang = parts[0];
    if (parts[1] && parts[1] !== "none") resolvedTargetLang = parts[1];
    if (parts[2]) detectedLanguage = parts[2];
  }

  const srcIdx = clean.indexOf("SRC:");
  const trIdx = clean.indexOf("\nTR:");
  let sourceTranscript = "";
  let translatedText = "";

  if (srcIdx !== -1) {
    const srcStart = srcIdx + 4;
    const srcEnd = trIdx !== -1 ? trIdx : clean.length;
    sourceTranscript = clean.slice(srcStart, srcEnd).trim();
  }
  if (trIdx !== -1) {
    translatedText = clean.slice(trIdx + 4).trim();
  }

  if (sourceTranscript.toUpperCase() === "NONE") sourceTranscript = "";
  if (translatedText.toUpperCase() === "NONE") translatedText = "";

  return {
    sourceTranscript,
    detectedLanguage,
    translatedText,
    resolvedSourceLang,
    resolvedTargetLang,
  };
}

async function transcribeAndTranslateStream(params: {
  audioBase64: string;
  mimeType: string;
  sourceLangCode: string;
  targetLangCode: string;
  bidirectional?: boolean;
  onPartial?: (partial: TranslatedTurnResult) => void;
}): Promise<TranslatedTurnResult> {
  const ai = getAiClient();
  const targetLang = getLanguageByCode(params.targetLangCode);
  const sourceLang =
    params.sourceLangCode === "auto" ? null : getLanguageByCode(params.sourceLangCode);
  const isBidirectional =
    Boolean(params.bidirectional) && sourceLang !== null && sourceLang.code !== targetLang.code;
  const involvesEthiopic =
    params.sourceLangCode === "am" ||
    params.targetLangCode === "am" ||
    params.sourceLangCode === "ti" ||
    params.targetLangCode === "ti";

  const directionPrompt = isBidirectional
    ? `Two-way conversation between ${sourceLang.name} (${sourceLang.nativeName}) [${sourceLang.code}] and ${targetLang.name} (${targetLang.nativeName}) [${targetLang.code}]. If spoken in ${sourceLang.name}, translate to ${targetLang.name}; if spoken in ${targetLang.name}, translate to ${sourceLang.name}.`
    : sourceLang === null
    ? `Auto-detect the spoken language and translate into ${targetLang.name} (${targetLang.nativeName}) [${targetLang.code}].`
    : `Speaker is using ${sourceLang.name} (${sourceLang.nativeName}) [${sourceLang.code}]. Translate into ${targetLang.name} (${targetLang.nativeName}) [${targetLang.code}].`;

  const ethiopicRules = involvesEthiopic
    ? `\nETHIOPIC / AMHARIC ACCURACY: Transcribe and translate Amharic (አማርኛ) / Tigrinya (ትግርኛ) strictly in standard Ethiopic Fidel script (ፊደል). Distinguish ejective vs plain consonants (ተ/ጠ, ቸ/ጨ, ፐ/ጰ, ቀ/ከ, ሰ/ሸ/ጸ) and exact vowel orders accurately.`
    : "";

  const promptText = `Listen very carefully to the audio—including low-tone, quiet, whispered, or distant speech. ${directionPrompt}${ethiopicRules}
Transcribe EVERY single spoken word faithfully in its native script, and translate the full natural meaning with zero omissions.
If there is NO human speech (only noise/silence), output exactly: NONE
Otherwise, output ONLY in this exact 3-line format (no markdown, no extra text):
LANG: <detected BCP-47 source code>|<target BCP-47 code>|<Language English Name>
SRC: <exact word-for-word transcript>
TR: <complete natural translation>`;

  const parts = [
    {
      inlineData: {
        mimeType: params.mimeType || "audio/wav",
        data: params.audioBase64,
      },
    },
    { text: promptText },
  ];

  try {
    const stream = await ai.models.generateContentStream({
      model: "gemini-4-hydrogen",
      contents: { parts },
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      },
    });
    let fullText = "";
    let lastSrcSent = "";
    let lastTrSent = "";
    for await (const chunk of stream) {
      if (chunk.text) {
        fullText += chunk.text;
        if (params.onPartial) {
          const parsed = parseStreamingTaggedOutput(fullText);
          if (
            parsed.sourceTranscript !== lastSrcSent ||
            parsed.translatedText !== lastTrSent
          ) {
            lastSrcSent = parsed.sourceTranscript;
            lastTrSent = parsed.translatedText;
            params.onPartial(parsed);
          }
        }
      }
    }
    return parseStreamingTaggedOutput(fullText);
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err)) {
      const fallback = await transcribeAndTranslate({
        audioBase64: params.audioBase64,
        mimeType: params.mimeType,
        sourceLangCode: params.sourceLangCode,
        targetLangCode: params.targetLangCode,
        bidirectional: params.bidirectional,
      });
      if (params.onPartial && (fallback.sourceTranscript || fallback.translatedText)) {
        params.onPartial(fallback);
      }
      return fallback;
    }
    throw err;
  }
}

async function translateTextOnly(params: {
  text: string;
  sourceLangName: string;
  targetLangCode: string;
}): Promise<string> {
  const ai = getAiClient();
  const targetLang = getLanguageByCode(params.targetLangCode);
  const prompt = `Translate this ${params.sourceLangName} text completely and accurately into ${targetLang.name} (${targetLang.nativeName}), preserving every word and detail. Return ONLY the translated text.\n\nText: ${params.text}`;
  let response;
  try {
    response = await ai.models.generateContent({
      model: "gemini-4-hydrogen",
      contents: prompt,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      },
    });
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err)) {
      await new Promise((r) => setTimeout(r, 300));
      response = await ai.models.generateContent({
        model: "gemini-3.1-flash-lite",
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
        },
      });
    } else {
      throw err;
    }
  }
  return (response.text ?? "").trim();
}

const TTS_STYLE_DIRECTIVE =
  "Crystal-clear, articulate, unhurried simultaneous interpreter voice; pronounce every single word and letter distinctly.";

function isSchemaFieldError(err: unknown): boolean {
  const msg = extractErrorMessage(err);
  return (
    msg.includes("INVALID_ARGUMENT") ||
    msg.includes("speechMetadata") ||
    msg.includes("speechConfig")
  );
}

async function synthesizeSpeechWav(text: string, voiceName: string): Promise<string | null> {
  if (!text.trim()) return null;
  const ai = getAiClient("v1beta");
  const safeVoice = sanitizeVoice(voiceName);

  const runTts = (model: string, withSpeechConfig: boolean) =>
    ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            withSpeechConfig
              ? { text, speechMetadata: { style: TTS_STYLE_DIRECTIVE } }
              : { text },
          ],
        },
      ],
      config: withSpeechConfig
        ? {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: safeVoice },
              },
            },
          }
        : {
            responseModalities: ["AUDIO"],
          },
    });

  const executeWithFallbacks = async (model: string) => {
    try {
      return await runTts(model, true);
    } catch (err) {
      if (isSchemaFieldError(err)) {
        return await runTts(model, false);
      }
      throw err;
    }
  };

  let response;
  try {
    response = await executeWithFallbacks("gemini-3.8-flash-lite-tts");
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err) || isSchemaFieldError(err)) {
      response = await executeWithFallbacks("gemini-3.8-flash-tts");
    } else {
      throw err;
    }
  }

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  return base64Audio ?? null;
}

/**
 * Streams raw headerless 24kHz PCM16 chunks immediately as Gemini synthesizes them.
 * Always passes raw 24kHz Buffer chunks so callers can relay either full 24kHz
 * or downsampled 12kHz (Low-Network Mode) per listener.
 */
async function synthesizeSpeechStream(
  text: string,
  voiceName: string,
  onPcm24kChunk: (pcm24kBuffer: Buffer) => void
): Promise<boolean> {
  if (!text.trim()) return false;
  const ai = getAiClient("v1beta");
  const safeVoice = sanitizeVoice(voiceName);

  const streamWithModel = async (model: string, withSpeechConfig: boolean): Promise<boolean> => {
    let emitted = false;
    const stream = await ai.models.generateContentStream({
      model,
      contents: [
        {
          role: "user",
          parts: [
            withSpeechConfig
              ? { text, speechMetadata: { style: TTS_STYLE_DIRECTIVE } }
              : { text },
          ],
        },
      ],
      config: withSpeechConfig
        ? {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: safeVoice },
              },
            },
          }
        : {
            responseModalities: ["AUDIO"],
          },
    });

    for await (const chunk of stream) {
      for (const part of chunk.candidates?.[0]?.content?.parts ?? []) {
        const audioData = part.inlineData?.data;
        if (audioData) {
          emitted = true;
          onPcm24kChunk(Buffer.from(audioData, "base64"));
        }
      }
    }
    return emitted;
  };

  const executeStream = async (model: string) => {
    try {
      return await streamWithModel(model, true);
    } catch (err) {
      if (isSchemaFieldError(err)) {
        return await streamWithModel(model, false);
      }
      throw err;
    }
  };

  try {
    return await executeStream("gemini-3.8-flash-lite-tts");
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err) || isSchemaFieldError(err)) {
      return await executeStream("gemini-3.8-flash-tts");
    }
    throw err;
  }
}

// ============================================================================
// Multi-Device Room Store (Server-Authoritative State)
// ============================================================================
type LiveSessionHandle = Awaited<ReturnType<GoogleGenAI["live"]["connect"]>>;

interface LiveTranslatorChannel {
  targetLangCode: string;
  session: LiveSessionHandle | null;
  connecting: boolean;
  closed: boolean;
  mode: "gemini_live" | "vad_stream";
  resumeHandle?: string;
  generation: number;
  reconnectAttempts: number;
  inputBuf: string;
  outputBuf: string;
  vadPreRoll: Buffer[];
  vadChunks: Buffer[];
  vadSpeechActive: boolean;
  vadSpeechStartAt: number;
  vadLastVoiceAt: number;
  vadNoiseFloor: number;
  vadQueue: Buffer[][];
  vadDraining: boolean;
}

interface RoomParticipant {
  id: string;
  name: string;
  langCode: string;
  voiceName: string;
  liveCallActive: boolean;
  lowBandwidth: boolean;
  whisperBoost: boolean;
  liveTranslators: Map<string, LiveTranslatorChannel>;
  ws: WebSocket;
  joinedAt: number;
}

interface StoredRoomMessage {
  id: string;
  speakerId: string;
  speakerName: string;
  sourceLang: string;
  sourceText: string;
  translations: Record<string, string>;
  timestamp: number;
}

interface RoomState {
  code: string;
  participants: Map<string, RoomParticipant>;
  messages: StoredRoomMessage[];
  processedMessageIds: Set<string>;
  cleanupTimer?: ReturnType<typeof setTimeout>;
}

const rooms = new Map<string, RoomState>();

function generateRoomCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < 6; i++) {
    code += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return rooms.has(code) ? generateRoomCode() : code;
}

function serializeParticipants(room: RoomState) {
  return Array.from(room.participants.values()).map((p) => ({
    id: p.id,
    name: p.name,
    langCode: p.langCode,
    voiceName: p.voiceName,
    liveCallActive: p.liveCallActive,
  }));
}

function broadcastRoomPresence(room: RoomState) {
  const payload = JSON.stringify({
    type: "room:presence",
    participants: serializeParticipants(room),
  });
  for (const participant of room.participants.values()) {
    if (participant.ws.readyState === WebSocket.OPEN) {
      participant.ws.send(payload);
    }
  }
}

function closeParticipantTranslators(participant: RoomParticipant) {
  for (const ch of participant.liveTranslators.values()) {
    ch.closed = true;
    try {
      ch.session?.close();
    } catch {
      // ignore
    }
    ch.session = null;
  }
  participant.liveTranslators.clear();
}

async function drainRoomChannelVadQueue(
  room: RoomState,
  speaker: RoomParticipant,
  channel: LiveTranslatorChannel
) {
  if (channel.vadDraining || channel.closed) return;
  channel.vadDraining = true;
  try {
    while (channel.vadQueue.length > 0 && !channel.closed) {
      // Coalesce queued speech buffers into a single call to eliminate latency buildup & 429s
      const utterance: Buffer[] = [];
      while (channel.vadQueue.length > 0) {
        utterance.push(...channel.vadQueue.shift()!);
      }
      const wavBase64 = pcm16ChunksToWavBase64(utterance, 16000);
      try {
        const result = await transcribeAndTranslateStream({
          audioBase64: wavBase64,
          mimeType: "audio/wav",
          sourceLangCode: speaker.langCode,
          targetLangCode: channel.targetLangCode,
          onPartial: (partial) => {
            if (channel.closed) return;
            if (partial.sourceTranscript) {
              const capPayload = JSON.stringify({
                type: "room:live_caption",
                speakerId: speaker.id,
                speakerName: speaker.name,
                sourceLang: speaker.langCode,
                text: partial.sourceTranscript,
              });
              if (speaker.ws.readyState === WebSocket.OPEN) speaker.ws.send(capPayload);
              for (const p of room.participants.values()) {
                if (
                  p.id !== speaker.id &&
                  p.langCode === channel.targetLangCode &&
                  p.ws.readyState === WebSocket.OPEN
                ) {
                  p.ws.send(capPayload);
                }
              }
            }
            if (partial.translatedText) {
              const outCapPayload = JSON.stringify({
                type: "room:live_translated_caption",
                speakerId: speaker.id,
                speakerName: speaker.name,
                sourceLang: speaker.langCode,
                targetLang: channel.targetLangCode,
                text: partial.translatedText,
              });
              for (const p of room.participants.values()) {
                if (
                  p.id !== speaker.id &&
                  p.langCode === channel.targetLangCode &&
                  p.ws.readyState === WebSocket.OPEN
                ) {
                  p.ws.send(outCapPayload);
                }
              }
            }
          },
        });
        if (channel.closed || !result.sourceTranscript || !result.translatedText) {
          continue;
        }
        const msgId = `live_${speaker.id}_${Date.now()}_${channel.targetLangCode}`;
        const ts = Date.now();
        room.messages.push({
          id: msgId,
          speakerId: speaker.id,
          speakerName: speaker.name,
          sourceLang: speaker.langCode,
          sourceText: result.sourceTranscript,
          translations: {
            [speaker.langCode]: result.sourceTranscript,
            [channel.targetLangCode]: result.translatedText,
          },
          timestamp: ts,
        });
        if (room.messages.length > 100) room.messages.shift();

        // Broadcast transcript message immediately before TTS finishes
        for (const p of room.participants.values()) {
          if (p.ws.readyState !== WebSocket.OPEN) continue;
          if (p.id === speaker.id || p.langCode === channel.targetLangCode) {
            p.ws.send(
              JSON.stringify({
                type: "room:message",
                message: {
                  id: msgId,
                  speakerId: speaker.id,
                  speakerName: speaker.name,
                  sourceLang: speaker.langCode,
                  sourceText: result.sourceTranscript,
                  targetLang: p.langCode,
                  translatedText:
                    p.langCode === channel.targetLangCode
                      ? result.translatedText
                      : result.sourceTranscript,
                  audioWavBase64: null,
                  liveStreamed: true,
                  timestamp: ts,
                },
              })
            );
          }
        }

        // Stream 24kHz (normal) or 12kHz (low-bandwidth) PCM chunks live to listeners
        let leftoverLow: Buffer | null = null;
        const streamed = await synthesizeSpeechStream(
          result.translatedText,
          speaker.voiceName,
          (pcm24kBuf) => {
            if (channel.closed) return;
            const fullPayload = JSON.stringify({
              type: "room:live_audio_out",
              speakerId: speaker.id,
              targetLang: channel.targetLangCode,
              sampleRate: 24000,
              data: pcm24kBuf.toString("base64"),
            });
            let lowPayload: string | null = null;

            for (const p of room.participants.values()) {
              if (
                p.id !== speaker.id &&
                p.langCode === channel.targetLangCode &&
                p.ws.readyState === WebSocket.OPEN
              ) {
                if (p.lowBandwidth) {
                  if (lowPayload === null) {
                    const combined = leftoverLow
                      ? Buffer.concat([leftoverLow, pcm24kBuf])
                      : pcm24kBuf;
                    const usableBytes = Math.floor(combined.length / 4) * 4;
                    leftoverLow =
                      combined.length > usableBytes ? combined.subarray(usableBytes) : null;
                    if (usableBytes >= 4) {
                      const down12k = downsample24kTo12kPcm16(combined.subarray(0, usableBytes));
                      lowPayload = JSON.stringify({
                        type: "room:live_audio_out",
                        speakerId: speaker.id,
                        targetLang: channel.targetLangCode,
                        sampleRate: 12000,
                        data: down12k.toString("base64"),
                      });
                    }
                  }
                  if (lowPayload) p.ws.send(lowPayload);
                } else {
                  p.ws.send(fullPayload);
                }
              }
            }
          }
        ).catch(() => false);

        if (!streamed && !channel.closed) {
          const fallbackPayload = JSON.stringify({
            type: "room:fallback_speak",
            speakerId: speaker.id,
            text: result.translatedText,
            lang: channel.targetLangCode,
          });
          for (const p of room.participants.values()) {
            if (
              p.id !== speaker.id &&
              p.langCode === channel.targetLangCode &&
              p.ws.readyState === WebSocket.OPEN
            ) {
              p.ws.send(fallbackPayload);
            }
          }
        }
      } catch {
        // Keep continuous stream alive across individual turn glitches
      }
    }
  } finally {
    channel.vadDraining = false;
  }
}

function feedRoomChannelVadFrame(
  room: RoomState,
  speaker: RoomParticipant,
  channel: LiveTranslatorChannel,
  base64Pcm: string
) {
  const pcmBuf = Buffer.from(base64Pcm, "base64");
  if (pcmBuf.length < 4) return;
  const rms = computePcm16Rms(pcmBuf);
  const now = Date.now();
  const minFloor = speaker.whisperBoost ? 0.0024 : 0.0040;
  const multiplier = speaker.whisperBoost ? 1.85 : 2.15;
  const SPEECH_THRESHOLD = Math.max(minFloor, channel.vadNoiseFloor * multiplier);
  const SILENCE_HANG_MS = 450;
  const MAX_UTTERANCE_MS = 6500;
  const MIN_UTTERANCE_BYTES = 16000 * 2 * 0.24; // 240ms catches crisp short words

  if (!channel.vadSpeechActive) {
    if (rms < SPEECH_THRESHOLD) {
      channel.vadNoiseFloor = channel.vadNoiseFloor * 0.92 + rms * 0.08;
    }
    channel.vadPreRoll.push(pcmBuf);
    if (channel.vadPreRoll.length > 5) channel.vadPreRoll.shift();
    if (rms >= SPEECH_THRESHOLD) {
      channel.vadSpeechActive = true;
      channel.vadSpeechStartAt = now;
      channel.vadLastVoiceAt = now;
      channel.vadChunks = [...channel.vadPreRoll, pcmBuf];
      channel.vadPreRoll = [];
    }
    return;
  }

  channel.vadChunks.push(pcmBuf);
  if (rms >= SPEECH_THRESHOLD) {
    channel.vadLastVoiceAt = now;
  }

  const silenceElapsed = now - channel.vadLastVoiceAt;
  const speechElapsed = now - channel.vadSpeechStartAt;
  if (silenceElapsed >= SILENCE_HANG_MS || speechElapsed >= MAX_UTTERANCE_MS) {
    const totalBytes = channel.vadChunks.reduce((acc, b) => acc + b.length, 0);
    const utterance = channel.vadChunks;
    channel.vadChunks = [];
    channel.vadSpeechActive = false;
    if (totalBytes >= MIN_UTTERANCE_BYTES) {
      channel.vadQueue.push(utterance);
      void drainRoomChannelVadQueue(room, speaker, channel);
    }
  }
}

async function openRoomTranslatorChannel(
  room: RoomState,
  speaker: RoomParticipant,
  targetLangCode: string
): Promise<void> {
  const channel = speaker.liveTranslators.get(targetLangCode);
  if (!channel || channel.closed) return;

  const targetLang = getLanguageByCode(targetLangCode);
  const sourceLang = getLanguageByCode(speaker.langCode);

  const requirePrecisionEngine =
    speaker.lowBandwidth ||
    PRECISION_VOICE_LANGS.has(speaker.langCode) ||
    PRECISION_VOICE_LANGS.has(targetLangCode) ||
    !targetLang.liveTranslateSupported;

  if (requirePrecisionEngine) {
    channel.mode = "vad_stream";
    channel.connecting = false;
    return;
  }

  let usedFallback = false;

  const connectWithModel = async (
    useTranslateModel: boolean,
    withCompression: boolean
  ): Promise<LiveSessionHandle> => {
    const ai = getAiClient("v1beta");
    const model = useTranslateModel ? "gemini-3.5-live-translate-preview" : "gemini-3.8-live";
    const gen = ++channel.generation;

    const baseConfig = useTranslateModel
      ? {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          translationConfig: {
            targetLanguageCode: targetLang.code,
            echoTargetLanguage: false,
          },
        }
      : {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: speaker.voiceName } },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: `You are an ultra-fast, high-precision simultaneous voice interpreter on a live phone call. Listen carefully even to soft, whispered, or distant speech from ${speaker.name} in ${sourceLang.name} (${sourceLang.nativeName}), and immediately speak ONLY the complete, word-for-word accurate translation in ${targetLang.name} (${targetLang.nativeName}), pronouncing every single word clearly without skipping or summarizing.`,
        };

    const config = withCompression
      ? {
          ...baseConfig,
          contextWindowCompression: { slidingWindow: {} },
          sessionResumption: channel.resumeHandle ? { handle: channel.resumeHandle } : {},
        }
      : baseConfig;

    return ai.live.connect({
      model,
      config,
      callbacks: {
        onmessage: (message: LiveServerMessage) => {
          if (channel.closed || gen !== channel.generation) return;
          if (message.sessionResumptionUpdate?.resumable && message.sessionResumptionUpdate?.newHandle) {
            channel.resumeHandle = message.sessionResumptionUpdate.newHandle;
          }
          if (message.goAway) {
            void reconnectRoomChannel(useTranslateModel);
            return;
          }
          const content = message.serverContent;

          if (content?.inputTranscription?.text) {
            channel.inputBuf += content.inputTranscription.text;
            const capPayload = JSON.stringify({
              type: "room:live_caption",
              speakerId: speaker.id,
              speakerName: speaker.name,
              sourceLang: speaker.langCode,
              text: channel.inputBuf,
            });
            if (speaker.ws.readyState === WebSocket.OPEN) {
              speaker.ws.send(capPayload);
            }
            for (const p of room.participants.values()) {
              if (
                p.id !== speaker.id &&
                p.langCode === targetLangCode &&
                p.ws.readyState === WebSocket.OPEN
              ) {
                p.ws.send(capPayload);
              }
            }
          }

          if (content?.outputTranscription?.text) {
            channel.outputBuf += content.outputTranscription.text;
            const outCapPayload = JSON.stringify({
              type: "room:live_translated_caption",
              speakerId: speaker.id,
              speakerName: speaker.name,
              sourceLang: speaker.langCode,
              targetLang: targetLangCode,
              text: channel.outputBuf,
            });
            for (const p of room.participants.values()) {
              if (
                p.id !== speaker.id &&
                p.langCode === targetLangCode &&
                p.ws.readyState === WebSocket.OPEN
              ) {
                p.ws.send(outCapPayload);
              }
            }
          }

          for (const part of content?.modelTurn?.parts ?? []) {
            if (part.inlineData?.data) {
              const audioPayload = JSON.stringify({
                type: "room:live_audio_out",
                speakerId: speaker.id,
                targetLang: targetLangCode,
                data: part.inlineData.data,
              });
              for (const p of room.participants.values()) {
                if (
                  p.id !== speaker.id &&
                  p.langCode === targetLangCode &&
                  p.ws.readyState === WebSocket.OPEN
                ) {
                  p.ws.send(audioPayload);
                }
              }
            }
          }

          if (content?.interrupted) {
            const intPayload = JSON.stringify({
              type: "room:live_interrupted",
              speakerId: speaker.id,
            });
            for (const p of room.participants.values()) {
              if (
                p.id !== speaker.id &&
                p.langCode === targetLangCode &&
                p.ws.readyState === WebSocket.OPEN
              ) {
                p.ws.send(intPayload);
              }
            }
          }

          if (content?.turnComplete) {
            const srcText = channel.inputBuf.trim();
            const outText = channel.outputBuf.trim();
            channel.inputBuf = "";
            channel.outputBuf = "";

            if (srcText || outText) {
              const msgId = `live_${speaker.id}_${Date.now()}_${targetLangCode}`;
              const ts = Date.now();
              room.messages.push({
                id: msgId,
                speakerId: speaker.id,
                speakerName: speaker.name,
                sourceLang: speaker.langCode,
                sourceText: srcText,
                translations: {
                  [speaker.langCode]: srcText,
                  [targetLangCode]: outText,
                },
                timestamp: ts,
              });
              if (room.messages.length > 100) {
                room.messages.shift();
              }

              for (const p of room.participants.values()) {
                if (p.ws.readyState !== WebSocket.OPEN) continue;
                if (p.id === speaker.id || p.langCode === targetLangCode) {
                  p.ws.send(
                    JSON.stringify({
                      type: "room:message",
                      message: {
                        id: msgId,
                        speakerId: speaker.id,
                        speakerName: speaker.name,
                        sourceLang: speaker.langCode,
                        sourceText: srcText,
                        targetLang: p.langCode,
                        translatedText:
                          p.langCode === targetLangCode ? outText || srcText : srcText,
                        audioWavBase64: null,
                        liveStreamed: true,
                        timestamp: ts,
                      },
                    })
                  );
                }
              }
            }
          }
        },
        onerror: () => {
          // handled via onclose / catch
        },
        onclose: () => {
          if (channel.closed || gen !== channel.generation) return;
          void reconnectRoomChannel(useTranslateModel);
        },
      },
    });
  };

  const reconnectRoomChannel = async (useTranslateModel: boolean) => {
    if (channel.closed || channel.mode === "vad_stream") return;
    if (channel.reconnectAttempts >= 3) {
      channel.mode = "vad_stream";
      channel.session = null;
      return;
    }
    channel.reconnectAttempts += 1;
    try {
      channel.session?.close();
    } catch {
      // ignore
    }
    try {
      channel.session = await connectWithModel(useTranslateModel, true);
      channel.reconnectAttempts = 0;
    } catch {
      if (useTranslateModel && !usedFallback) {
        usedFallback = true;
        try {
          channel.session = await connectWithModel(false, false);
          channel.reconnectAttempts = 0;
          return;
        } catch {
          // switch to continuous VAD stream
        }
      }
      channel.mode = "vad_stream";
      channel.session = null;
    }
  };

  try {
    channel.session = await connectWithModel(targetLang.liveTranslateSupported, true);
  } catch {
    if (targetLang.liveTranslateSupported && !usedFallback) {
      usedFallback = true;
      try {
        channel.session = await connectWithModel(false, false);
      } catch {
        channel.mode = "vad_stream";
        channel.session = null;
      }
    } else {
      channel.mode = "vad_stream";
      channel.session = null;
    }
  } finally {
    channel.connecting = false;
  }
}

function reconcileParticipantTranslators(room: RoomState, speaker: RoomParticipant) {
  if (!speaker.liveCallActive) {
    closeParticipantTranslators(speaker);
    return;
  }

  const neededLangs = new Set<string>();
  for (const p of room.participants.values()) {
    if (p.id !== speaker.id && p.langCode !== speaker.langCode) {
      neededLangs.add(p.langCode);
    }
  }

  // Close translators no longer needed
  for (const [lang, ch] of speaker.liveTranslators.entries()) {
    if (!neededLangs.has(lang)) {
      ch.closed = true;
      try {
        ch.session?.close();
      } catch {
        // ignore
      }
      speaker.liveTranslators.delete(lang);
    }
  }

  // Open new translators for newly needed languages
  for (const targetLang of neededLangs) {
    if (!speaker.liveTranslators.has(targetLang)) {
      const ch: LiveTranslatorChannel = {
        targetLangCode: targetLang,
        session: null,
        connecting: true,
        closed: false,
        mode: "gemini_live",
        generation: 0,
        reconnectAttempts: 0,
        inputBuf: "",
        outputBuf: "",
        vadPreRoll: [],
        vadChunks: [],
        vadSpeechActive: false,
        vadSpeechStartAt: 0,
        vadLastVoiceAt: 0,
        vadNoiseFloor: 0.0025,
        vadQueue: [],
        vadDraining: false,
      };
      speaker.liveTranslators.set(targetLang, ch);
      void openRoomTranslatorChannel(room, speaker, targetLang);
    }
  }
}

function reconcileAllRoomTranslators(room: RoomState) {
  for (const p of room.participants.values()) {
    reconcileParticipantTranslators(room, p);
  }
}

function splitReportIntoReadableSections(text: string): string[] {
  const cleaned = text.trim();
  if (!cleaned) return [];
  const rawParagraphs = cleaned
    .split(/\n{1,}/)
    .map((p) => p.trim())
    .filter(Boolean);

  const sections: string[] = [];
  for (const para of rawParagraphs) {
    if (para.length <= 420) {
      sections.push(para);
      continue;
    }
    // Split long paragraphs on Ethiopic (።) or Latin/CJK sentence delimiters cleanly
    const sentences = para.match(/[^.!?።;\n]+[.!?።;\n]*/g) ?? [para];
    let current = "";
    for (const s of sentences) {
      const part = s.trim();
      if (!part) continue;
      if (current && current.length + part.length + 1 > 380) {
        sections.push(current);
        current = part;
      } else {
        current = current ? `${current} ${part}` : part;
      }
    }
    if (current) sections.push(current);
  }
  return sections;
}

async function startServer() {
  const app = express();
  app.set("trust proxy", 1);
  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    next();
  });
  app.use(express.json({ limit: "25mb" }));

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true });
  });

  // Turn-by-turn audio or text translation + TTS synthesis
  app.post("/api/translate-turn", async (req, res) => {
    try {
      const {
        audioBase64,
        mimeType,
        text,
        sourceLang,
        targetLang,
        voiceName,
        bidirectional,
      } = req.body ?? {};
      const safeSource = sanitizeLangCode(sourceLang, true);
      const safeTarget = sanitizeLangCode(targetLang, false);
      const safeVoice = sanitizeVoice(voiceName);
      const isBidirectional = Boolean(bidirectional);

      if (!audioBase64 && (!text || typeof text !== "string" || !text.trim())) {
        res.status(400).json({ error: "Either audioBase64 or non-empty text is required." });
        return;
      }

      const translated = await transcribeAndTranslate({
        audioBase64: typeof audioBase64 === "string" ? audioBase64 : undefined,
        mimeType: typeof mimeType === "string" ? mimeType : "audio/webm",
        text: typeof text === "string" ? text.trim() : undefined,
        sourceLangCode: safeSource,
        targetLangCode: safeTarget,
        bidirectional: isBidirectional,
      });

      if (!translated.sourceTranscript && !translated.translatedText) {
        res.json({
          sourceTranscript: "",
          detectedLanguage: translated.detectedLanguage,
          translatedText: "",
          audioWavBase64: null,
        });
        return;
      }

      let audioWavBase64: string | null = null;
      let ttsWarning: string | undefined;
      try {
        audioWavBase64 = await synthesizeSpeechWav(translated.translatedText, safeVoice);
      } catch (ttsErr) {
        ttsWarning = ttsErr instanceof Error ? ttsErr.message : "Voice synthesis unavailable";
      }

      res.json({
        sourceTranscript: translated.sourceTranscript,
        detectedLanguage: translated.detectedLanguage,
        translatedText: translated.translatedText,
        resolvedSourceLang: translated.resolvedSourceLang,
        resolvedTargetLang: translated.resolvedTargetLang,
        audioWavBase64,
        fallbackTts: !audioWavBase64,
        ttsWarning,
      });
    } catch (err) {
      const formatted = formatApiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  // Fast translation endpoint (returns text immediately without blocking for full WAV synthesis)
  app.post("/api/translate-fast", async (req, res) => {
    try {
      const { audioBase64, mimeType, text, sourceLang, targetLang, bidirectional } = req.body ?? {};
      const safeSource = sanitizeLangCode(sourceLang, true);
      const safeTarget = sanitizeLangCode(targetLang, false);
      const isBidirectional = Boolean(bidirectional);

      if (!audioBase64 && (!text || typeof text !== "string" || !text.trim())) {
        res.status(400).json({ error: "Either audioBase64 or non-empty text is required." });
        return;
      }

      const translated = await transcribeAndTranslate({
        audioBase64: typeof audioBase64 === "string" ? audioBase64 : undefined,
        mimeType: typeof mimeType === "string" ? mimeType : "audio/webm",
        text: typeof text === "string" ? text.trim() : undefined,
        sourceLangCode: safeSource,
        targetLangCode: safeTarget,
        bidirectional: isBidirectional,
      });

      res.json({
        sourceTranscript: translated.sourceTranscript,
        detectedLanguage: translated.detectedLanguage,
        translatedText: translated.translatedText,
        resolvedSourceLang: translated.resolvedSourceLang,
        resolvedTargetLang: translated.resolvedTargetLang,
      });
    } catch (err) {
      const formatted = formatApiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  // Raw PCM16LE streaming TTS endpoint (24kHz standard OR 12kHz Low-Network Mode)
  app.post("/api/tts-stream", async (req, res) => {
    const { text, voiceName, lowBandwidth } = req.body ?? {};
    if (typeof text !== "string" || !text.trim()) {
      res.status(400).json({ error: "Text is required." });
      return;
    }
    const safeVoice = sanitizeVoice(voiceName);
    const isLowBandwidth = Boolean(lowBandwidth);

    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("X-Sample-Rate", isLowBandwidth ? "12000" : "24000");

    let leftoverLow: Buffer | null = null;
    try {
      const emitted = await synthesizeSpeechStream(text.trim(), safeVoice, (pcm24kBuf) => {
        if (res.writableEnded) return;
        if (isLowBandwidth) {
          const combined = leftoverLow ? Buffer.concat([leftoverLow, pcm24kBuf]) : pcm24kBuf;
          const usableBytes = Math.floor(combined.length / 4) * 4;
          leftoverLow = combined.length > usableBytes ? combined.subarray(usableBytes) : null;
          if (usableBytes >= 4) {
            res.write(downsample24kTo12kPcm16(combined.subarray(0, usableBytes)));
          }
        } else {
          res.write(pcm24kBuf);
        }
      });
      if (!emitted && !res.headersSent) {
        res.status(204).end();
        return;
      }
      res.end();
    } catch {
      if (!res.headersSent) {
        res.status(204).end();
      } else {
        res.end();
      }
    }
  });

  // Full Document & Report Translator (PDF, DOCX, TXT/CSV/MD, or document images -> full translation + audio sections)
  app.post("/api/translate-document", async (req, res) => {
    try {
      const { text, fileName, mimeType, fileBase64, sourceLang, targetLang } = req.body ?? {};
      const safeSource = sanitizeLangCode(sourceLang, true);
      const safeTarget = sanitizeLangCode(targetLang, false);
      const targetObj = getLanguageByCode(safeTarget);
      const sourceObj = safeSource === "auto" ? null : getLanguageByCode(safeSource);
      const ai = getAiClient();

      const lowerName = typeof fileName === "string" ? fileName.toLowerCase() : "";
      const lowerMime = typeof mimeType === "string" ? mimeType.toLowerCase() : "";

      let extractedText = typeof text === "string" ? text.trim() : "";
      let multimodalData: { mimeType: string; data: string } | null = null;

      if (typeof fileBase64 === "string" && fileBase64.length > 0) {
        const buffer = Buffer.from(fileBase64, "base64");
        if (
          lowerName.endsWith(".docx") ||
          lowerMime.includes("wordprocessingml")
        ) {
          const docResult = await mammoth.extractRawText({ buffer });
          extractedText = (docResult.value || "").trim();
          if (!extractedText) {
            res.status(400).json({ error: "Could not extract readable text from this Word document." });
            return;
          }
        } else if (
          lowerName.endsWith(".pdf") ||
          lowerMime === "application/pdf" ||
          lowerMime.startsWith("image/") ||
          /\.(png|jpe?g|webp|heic|heif)$/.test(lowerName)
        ) {
          const resolvedMime =
            lowerMime && (lowerMime === "application/pdf" || lowerMime.startsWith("image/"))
              ? lowerMime
              : lowerName.endsWith(".pdf")
              ? "application/pdf"
              : lowerName.endsWith(".png")
              ? "image/png"
              : lowerName.endsWith(".webp")
              ? "image/webp"
              : "image/jpeg";
          multimodalData = { mimeType: resolvedMime, data: fileBase64 };
        } else {
          extractedText = buffer.toString("utf8").replace(/\u0000/g, "").trim();
        }
      }

      if (!extractedText && !multimodalData) {
        res.status(400).json({ error: "No readable document content or text was provided." });
        return;
      }

      const ethiopicRule =
        safeSource === "am" || safeTarget === "am" || safeSource === "ti" || safeTarget === "ti"
          ? "\nUse authentic, standard Ethiopic/Ge'ez Fidel script (ፊደል) with accurate grammar and terminology."
          : "";

      if (multimodalData) {
        const srcDesc = sourceObj
          ? `written in ${sourceObj.name} (${sourceObj.nativeName})`
          : "written in any language (auto-detect)";
        const callMm = (model: string, level: ThinkingLevel) =>
          ai.models.generateContent({
            model,
            contents: {
              parts: [
                { inlineData: multimodalData! },
                {
                  text: `Read EVERY single page, heading, paragraph, table row, and bullet in this document/image (${srcDesc}).
Translate the ENTIRE document completely, faithfully, and naturally into ${targetObj.name} (${targetObj.nativeName}) with zero omissions or summarization.${ethiopicRule}
Return strict JSON with detectedLanguage, sourceText (full extracted original text preserving paragraphs), and translatedText (full translated report preserving paragraphs).`,
                },
              ],
            },
            config: {
              thinkingConfig: { thinkingLevel: level },
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  detectedLanguage: { type: Type.STRING },
                  sourceText: { type: Type.STRING },
                  translatedText: { type: Type.STRING },
                },
                required: ["detectedLanguage", "sourceText", "translatedText"],
              },
            },
          });

        let mmResp;
        try {
          mmResp = await callMm("gemini-4-hydrogen", ThinkingLevel.LOW);
        } catch (err) {
          if (isRetryableQuotaOrLoadError(err)) {
            mmResp = await callMm("gemini-3.1-flash-lite", ThinkingLevel.MINIMAL);
          } else {
            throw err;
          }
        }
        const parsed = JSON.parse((mmResp.text ?? "{}").trim()) as {
          detectedLanguage?: string;
          sourceText?: string;
          translatedText?: string;
        };
        const outTranslated = (parsed.translatedText ?? "").trim();
        const sections = splitReportIntoReadableSections(outTranslated);
        res.json({
          detectedLanguage: parsed.detectedLanguage || (sourceObj?.name ?? "Auto"),
          sourceText: (parsed.sourceText ?? "").trim(),
          translatedText: outTranslated,
          sections,
        });
        return;
      }

      // Chunk long plain/docx reports into <= 4,500-char paragraph batches so zero sentences are dropped
      const paragraphs = extractedText
        .split(/\n{2,}/)
        .map((p) => p.trim())
        .filter(Boolean);
      const chunks: string[] = [];
      let currentChunk = "";
      for (const p of paragraphs) {
        if (currentChunk && currentChunk.length + p.length + 2 > 4500) {
          chunks.push(currentChunk);
          currentChunk = p;
        } else {
          currentChunk = currentChunk ? `${currentChunk}\n\n${p}` : p;
        }
      }
      if (currentChunk) chunks.push(currentChunk);
      if (chunks.length === 0) chunks.push(extractedText);

      const srcHint = sourceObj
        ? `from ${sourceObj.name} (${sourceObj.nativeName})`
        : "(auto-detecting source language)";

      const translateBatch = async (batchText: string): Promise<string> => {
        const prompt = `Translate this entire document section completely and accurately ${srcHint} into ${targetObj.name} (${targetObj.nativeName}), preserving every paragraph, heading, bullet point, and number with zero omissions.${ethiopicRule}
Return ONLY the translated text preserving paragraph breaks.

Document text:
"""
${batchText}
"""`;
        try {
          const resp = await ai.models.generateContent({
            model: "gemini-4-hydrogen",
            contents: prompt,
            config: { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } },
          });
          return (resp.text ?? "").trim();
        } catch (err) {
          if (isRetryableQuotaOrLoadError(err)) {
            await new Promise((r) => setTimeout(r, 450));
            const resp2 = await ai.models.generateContent({
              model: "gemini-3.1-flash-lite",
              contents: prompt,
              config: { thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL } },
            });
            return (resp2.text ?? "").trim();
          }
          throw err;
        }
      };

      const translatedChunks: string[] = new Array(chunks.length);
      for (let i = 0; i < chunks.length; i += 2) {
        const pair = chunks.slice(i, i + 2);
        const results = await Promise.all(pair.map((c) => translateBatch(c)));
        for (let j = 0; j < results.length; j++) {
          translatedChunks[i + j] = results[j];
        }
      }

      const fullTranslated = translatedChunks.filter(Boolean).join("\n\n").trim();
      const sections = splitReportIntoReadableSections(fullTranslated);

      res.json({
        detectedLanguage: sourceObj?.name ?? "Auto",
        sourceText: extractedText,
        translatedText: fullTranslated,
        sections,
      });
    } catch (err) {
      const formatted = formatApiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  // Create a multi-device translation room
  app.post("/api/rooms", (_req, res) => {
    const code = generateRoomCode();
    rooms.set(code, {
      code,
      participants: new Map(),
      messages: [],
      processedMessageIds: new Set(),
    });
    res.json({ roomCode: code });
  });

  // Validate / inspect a room
  app.get("/api/rooms/:code", (req, res) => {
    const code = String(req.params.code || "").toUpperCase().trim();
    if (!/^[A-Z0-9]{4,10}$/.test(code)) {
      res.status(400).json({ error: "Invalid room code format." });
      return;
    }
    let room = rooms.get(code);
    if (!room) {
      // Allow deterministic joining of shared room codes even after server restart
      room = {
        code,
        participants: new Map(),
        messages: [],
        processedMessageIds: new Set(),
      };
      rooms.set(code, room);
    }
    res.json({
      exists: true,
      roomCode: code,
      participantCount: room.participants.size,
    });
  });

  const server = http.createServer(app);

  // WebSocket servers (noServer mode to avoid colliding with Vite middleware)
  const liveWss = new WebSocketServer({ noServer: true });
  const roomWss = new WebSocketServer({ noServer: true });

  // --------------------------------------------------------------------------
  // Live Continuous Speech-to-Speech Translation WebSocket (/ws/live)
  // Unlimited session duration via Context Compression, Session Resumption,
  // automatic reconnection, and seamless Continuous VAD Fallback.
  // --------------------------------------------------------------------------
  liveWss.on("connection", async (clientWs, req) => {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const targetCode = sanitizeLangCode(url.searchParams.get("targetLang") || "en", false);
    const sourceCode = sanitizeLangCode(url.searchParams.get("sourceLang") || "auto", true);
    const voiceName = sanitizeVoice(url.searchParams.get("voiceName") || "Zephyr");
    const bidirectionalParam = url.searchParams.get("bidirectional") === "true";
    const lowBandwidth = url.searchParams.get("lowBandwidth") === "true";
    const whisperBoost = url.searchParams.get("whisperBoost") !== "false";

    const targetLang = getLanguageByCode(targetCode);
    const sourceLang = sourceCode === "auto" ? null : getLanguageByCode(sourceCode);
    const isBidirectional =
      bidirectionalParam && sourceLang !== null && sourceLang.code !== targetLang.code;
    const sourceLangLabel = sourceLang ? sourceLang.name : "any spoken language";

    const usePrecisionEngineInitially =
      isBidirectional ||
      lowBandwidth ||
      sourceCode === "auto" ||
      PRECISION_VOICE_LANGS.has(sourceCode) ||
      PRECISION_VOICE_LANGS.has(targetCode) ||
      !targetLang.liveTranslateSupported;

    let activeSession: LiveSessionHandle | null = null;
    let isClosed = false;
    let mode: "gemini_live" | "vad_stream" = usePrecisionEngineInitially
      ? "vad_stream"
      : "gemini_live";
    let usedFallbackModel = false;
    let resumeHandle: string | undefined;
    let sessionGeneration = 0;
    let reconnectAttempts = 0;
    let statusSent = false;

    // Continuous Precision Streaming VAD state
    let vadPreRoll: Buffer[] = [];
    let vadChunks: Buffer[] = [];
    let vadSpeechActive = false;
    let vadSpeechStartAt = 0;
    let vadLastVoiceAt = 0;
    let vadNoiseFloor = 0.0022;
    const vadQueue: Buffer[][] = [];
    let vadDraining = false;

    const sendConnectedStatus = (activeMode: string) => {
      if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
      statusSent = true;
      clientWs.send(
        JSON.stringify({
          type: "status",
          connected: true,
          mode: activeMode,
          targetLang: targetLang.code,
          sampleRate: lowBandwidth ? 12000 : 24000,
        })
      );
    };

    const drainVadQueue = async () => {
      if (vadDraining || isClosed) return;
      vadDraining = true;
      try {
        while (vadQueue.length > 0 && !isClosed && clientWs.readyState === WebSocket.OPEN) {
          // Merge any stacked speech buffers into one turn to slash latency and RPM consumption
          const utterance: Buffer[] = [];
          while (vadQueue.length > 0) {
            utterance.push(...vadQueue.shift()!);
          }
          const wavBase64 = pcm16ChunksToWavBase64(utterance, 16000);
          try {
            const result = await transcribeAndTranslateStream({
              audioBase64: wavBase64,
              mimeType: "audio/wav",
              sourceLangCode: sourceCode,
              targetLangCode: targetLang.code,
              bidirectional: isBidirectional,
              onPartial: (partial) => {
                if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
                if (partial.sourceTranscript) {
                  clientWs.send(
                    JSON.stringify({
                      type: "inputTranscript",
                      text: partial.sourceTranscript,
                      replace: true,
                    })
                  );
                }
                if (partial.translatedText) {
                  clientWs.send(
                    JSON.stringify({
                      type: "outputTranscript",
                      text: partial.translatedText,
                      replace: true,
                    })
                  );
                }
              },
            });
            if (isClosed || clientWs.readyState !== WebSocket.OPEN) break;
            if (!result.sourceTranscript && !result.translatedText) continue;

            if (result.sourceTranscript) {
              clientWs.send(
                JSON.stringify({
                  type: "inputTranscript",
                  text: result.sourceTranscript,
                  replace: true,
                })
              );
            }
            if (result.translatedText) {
              clientWs.send(
                JSON.stringify({
                  type: "outputTranscript",
                  text: result.translatedText,
                  replace: true,
                })
              );
              let leftoverLow: Buffer | null = null;
              const streamed = await synthesizeSpeechStream(
                result.translatedText,
                voiceName,
                (pcm24kBuf) => {
                  if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
                  if (lowBandwidth) {
                    const combined = leftoverLow
                      ? Buffer.concat([leftoverLow, pcm24kBuf])
                      : pcm24kBuf;
                    const usableBytes = Math.floor(combined.length / 4) * 4;
                    leftoverLow =
                      combined.length > usableBytes ? combined.subarray(usableBytes) : null;
                    if (usableBytes >= 4) {
                      const down12k = downsample24kTo12kPcm16(combined.subarray(0, usableBytes));
                      clientWs.send(
                        JSON.stringify({
                          type: "audio",
                          sampleRate: 12000,
                          data: down12k.toString("base64"),
                        })
                      );
                    }
                  } else {
                    clientWs.send(
                      JSON.stringify({
                        type: "audio",
                        sampleRate: 24000,
                        data: pcm24kBuf.toString("base64"),
                      })
                    );
                  }
                }
              ).catch(() => false);

              if (!streamed && !isClosed && clientWs.readyState === WebSocket.OPEN) {
                clientWs.send(
                  JSON.stringify({
                    type: "fallbackSpeak",
                    text: result.translatedText,
                    lang: result.resolvedTargetLang || targetLang.code,
                  })
                );
              }
            }
            if (!isClosed && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(
                JSON.stringify({
                  type: "turnComplete",
                  resolvedSourceLang: result.resolvedSourceLang,
                  resolvedTargetLang: result.resolvedTargetLang,
                })
              );
            }
          } catch (err) {
            const formatted = formatApiError(err);
            if (formatted.status === 403 && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
              clientWs.close();
              return;
            }
            if (
              (formatted.status === 429 || formatted.status === 503) &&
              clientWs.readyState === WebSocket.OPEN
            ) {
              clientWs.send(
                JSON.stringify({
                  type: "notice",
                  level: "traffic",
                  message: formatted.message,
                })
              );
              await new Promise((r) => setTimeout(r, 900));
            }
          }
        }
      } finally {
        vadDraining = false;
      }
    };

    const feedVadFrame = (base64Pcm: string) => {
      const pcmBuf = Buffer.from(base64Pcm, "base64");
      if (pcmBuf.length < 4) return;
      const rms = computePcm16Rms(pcmBuf);
      const now = Date.now();
      const minFloor = whisperBoost ? 0.0024 : 0.0040;
      const multiplier = whisperBoost ? 1.85 : 2.15;
      const SPEECH_THRESHOLD = Math.max(minFloor, vadNoiseFloor * multiplier);
      const SILENCE_HANG_MS = 450;
      const MAX_UTTERANCE_MS = 6500;
      const MIN_UTTERANCE_BYTES = 16000 * 2 * 0.24; // 240ms preserves crisp short words

      if (!vadSpeechActive) {
        if (rms < SPEECH_THRESHOLD) {
          vadNoiseFloor = vadNoiseFloor * 0.92 + rms * 0.08;
        }
        vadPreRoll.push(pcmBuf);
        if (vadPreRoll.length > 5) vadPreRoll.shift();
        if (rms >= SPEECH_THRESHOLD) {
          vadSpeechActive = true;
          vadSpeechStartAt = now;
          vadLastVoiceAt = now;
          vadChunks = [...vadPreRoll, pcmBuf];
          vadPreRoll = [];
        }
        return;
      }

      vadChunks.push(pcmBuf);
      if (rms >= SPEECH_THRESHOLD) {
        vadLastVoiceAt = now;
      }

      const silenceElapsed = now - vadLastVoiceAt;
      const speechElapsed = now - vadSpeechStartAt;
      if (silenceElapsed >= SILENCE_HANG_MS || speechElapsed >= MAX_UTTERANCE_MS) {
        const totalBytes = vadChunks.reduce((acc, b) => acc + b.length, 0);
        const utterance = vadChunks;
        vadChunks = [];
        vadSpeechActive = false;
        if (totalBytes >= MIN_UTTERANCE_BYTES) {
          vadQueue.push(utterance);
          void drainVadQueue();
        }
      }
    };

    const flushVad = () => {
      if (!vadSpeechActive || vadChunks.length === 0) return;
      const totalBytes = vadChunks.reduce((acc, b) => acc + b.length, 0);
      const utterance = vadChunks;
      vadChunks = [];
      vadSpeechActive = false;
      if (totalBytes >= 16000 * 2 * 0.24) {
        vadQueue.push(utterance);
        void drainVadQueue();
      }
    };

    const activateContinuousVadFallback = () => {
      if (isClosed) return;
      mode = "vad_stream";
      try {
        activeSession?.close();
      } catch {
        // ignore
      }
      activeSession = null;
      sendConnectedStatus("precision-stream");
    };

    // Register client message & close listeners immediately to prevent missing early frames
    clientWs.on("message", (raw) => {
      if (isClosed) return;
      try {
        const parsed = JSON.parse(raw.toString()) as { type?: string; data?: string };
        if (parsed.type === "audio" && typeof parsed.data === "string" && parsed.data.length > 0) {
          if (mode === "vad_stream") {
            feedVadFrame(parsed.data);
          } else if (activeSession) {
            activeSession.sendRealtimeInput({
              audio: { data: parsed.data, mimeType: "audio/pcm;rate=16000" },
            });
          }
        } else if (parsed.type === "vad_flush" && mode === "vad_stream") {
          flushVad();
        }
      } catch {
        // Ignore malformed frames
      }
    });

    clientWs.on("close", () => {
      isClosed = true;
      sessionGeneration += 1;
      try {
        activeSession?.close();
      } catch {
        // ignore
      }
      activeSession = null;
    });

    const connectLiveSession = async (
      useTranslateModel: boolean,
      withCompression: boolean
    ): Promise<LiveSessionHandle> => {
      const ai = getAiClient("v1beta");
      const model = useTranslateModel ? "gemini-3.5-live-translate-preview" : "gemini-3.8-live";
      const gen = ++sessionGeneration;

      const systemInstruction =
        isBidirectional && sourceLang
          ? `You are an ultra-fast, high-precision two-way simultaneous voice interpreter between ${sourceLang.name} (${sourceLang.nativeName}) and ${targetLang.name} (${targetLang.nativeName}). Listen carefully even to soft, whispered, or distant speech: if it is ${sourceLang.name}, immediately speak ONLY the complete, word-for-word accurate translation in ${targetLang.name}; if it is ${targetLang.name}, immediately speak ONLY the complete translation in ${sourceLang.name}. Pronounce every word clearly with zero omissions or commentary.`
          : `You are an ultra-fast, high-precision simultaneous voice interpreter. Listen carefully even to soft, whispered, or distant speech (${sourceLangLabel}) and immediately speak ONLY the complete, word-for-word accurate translation in ${targetLang.name} (${targetLang.nativeName}), pronouncing every word clearly with zero omissions or commentary.`;

      const baseConfig = useTranslateModel
        ? {
            responseModalities: [Modality.AUDIO],
            inputAudioTranscription: {},
            outputAudioTranscription: {},
            translationConfig: {
              targetLanguageCode: targetLang.code,
              echoTargetLanguage: false,
            },
          }
        : {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName } },
            },
            inputAudioTranscription: {},
            outputAudioTranscription: {},
            systemInstruction,
          };

      const config = withCompression
        ? {
            ...baseConfig,
            contextWindowCompression: { slidingWindow: {} },
            sessionResumption: resumeHandle ? { handle: resumeHandle } : {},
          }
        : baseConfig;

      const session = await ai.live.connect({
        model,
        config,
        callbacks: {
          onopen: () => {
            if (!isClosed && gen === sessionGeneration) {
              sendConnectedStatus(model);
            }
          },
          onmessage: (message: LiveServerMessage) => {
            if (isClosed || gen !== sessionGeneration || clientWs.readyState !== WebSocket.OPEN) {
              return;
            }
            if (
              message.sessionResumptionUpdate?.resumable &&
              message.sessionResumptionUpdate?.newHandle
            ) {
              resumeHandle = message.sessionResumptionUpdate.newHandle;
            }
            if (message.goAway) {
              void reconnectLiveSession(useTranslateModel);
              return;
            }
            const content = message.serverContent;
            if (content?.inputTranscription?.text) {
              clientWs.send(
                JSON.stringify({
                  type: "inputTranscript",
                  text: content.inputTranscription.text,
                })
              );
            }
            if (content?.outputTranscription?.text) {
              clientWs.send(
                JSON.stringify({
                  type: "outputTranscript",
                  text: content.outputTranscription.text,
                })
              );
            }
            for (const part of content?.modelTurn?.parts ?? []) {
              if (part.inlineData?.data) {
                clientWs.send(
                  JSON.stringify({
                    type: "audio",
                    data: part.inlineData.data,
                  })
                );
              }
            }
            if (content?.interrupted) {
              clientWs.send(JSON.stringify({ type: "interrupted" }));
            }
            if (content?.turnComplete) {
              clientWs.send(JSON.stringify({ type: "turnComplete" }));
            }
          },
          onerror: (err: unknown) => {
            if (isClosed || gen !== sessionGeneration) return;
            const msg = extractErrorMessage(err);
            if (isFatalAuthError(msg)) {
              const formatted = formatApiError(err);
              if (clientWs.readyState === WebSocket.OPEN) {
                clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
                clientWs.close();
              }
            }
          },
          onclose: async () => {
            if (isClosed || gen !== sessionGeneration) return;
            await reconnectLiveSession(useTranslateModel);
          },
        },
      });

      return session;
    };

    const reconnectLiveSession = async (useTranslateModel: boolean) => {
      if (isClosed || mode === "vad_stream") return;
      if (reconnectAttempts >= 4) {
        activateContinuousVadFallback();
        return;
      }
      reconnectAttempts += 1;
      try {
        activeSession?.close();
      } catch {
        // ignore
      }
      try {
        activeSession = await connectLiveSession(useTranslateModel, true);
        reconnectAttempts = 0;
        if (!statusSent) sendConnectedStatus("gemini-live");
      } catch (err) {
        const msg = extractErrorMessage(err);
        if (isFatalAuthError(msg)) {
          const formatted = formatApiError(err);
          if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
            clientWs.close();
          }
          return;
        }
        if (useTranslateModel && !usedFallbackModel) {
          usedFallbackModel = true;
          try {
            activeSession = await connectLiveSession(false, false);
            reconnectAttempts = 0;
            if (!statusSent) sendConnectedStatus("gemini-3.8-live");
            return;
          } catch {
            // Fall through to Continuous VAD fallback
          }
        }
        activateContinuousVadFallback();
      }
    };

    if (usePrecisionEngineInitially) {
      sendConnectedStatus("precision-stream");
      return;
    }

    const useTranslateInitially = !isBidirectional && targetLang.liveTranslateSupported;
    try {
      activeSession = await connectLiveSession(useTranslateInitially, true);
    } catch (err) {
      const msg = extractErrorMessage(err);
      if (isFatalAuthError(msg)) {
        const formatted = formatApiError(err);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
          clientWs.close();
        }
        return;
      }
      if (useTranslateInitially && !usedFallbackModel) {
        usedFallbackModel = true;
        try {
          activeSession = await connectLiveSession(false, true);
        } catch {
          activateContinuousVadFallback();
        }
      } else {
        // Try without compression/resumption once, else activate unlimited VAD fallback
        try {
          activeSession = await connectLiveSession(false, false);
        } catch {
          activateContinuousVadFallback();
        }
      }
    }
  });

  // --------------------------------------------------------------------------
  // Multi-Device Room WebSocket (/ws/room)
  // --------------------------------------------------------------------------
  roomWss.on("connection", (ws, req) => {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const roomCode = (url.searchParams.get("roomCode") || "").toUpperCase().trim();
    const participantId = (url.searchParams.get("participantId") || "").trim();
    const name = (url.searchParams.get("name") || "Guest").slice(0, 40).trim() || "Guest";
    const langCode = sanitizeLangCode(url.searchParams.get("lang") || "en", false);
    const voiceName = sanitizeVoice(url.searchParams.get("voice") || "Zephyr");
    const lowBandwidth = url.searchParams.get("lowBandwidth") === "true";
    const whisperBoost = url.searchParams.get("whisperBoost") !== "false";

    if (!/^[A-Z0-9]{4,10}$/.test(roomCode) || !participantId) {
      ws.send(JSON.stringify({ type: "error", message: "Invalid room or participant ID." }));
      ws.close();
      return;
    }

    let room = rooms.get(roomCode);
    if (!room) {
      room = {
        code: roomCode,
        participants: new Map(),
        messages: [],
        processedMessageIds: new Set(),
      };
      rooms.set(roomCode, room);
    }

    if (room.cleanupTimer) {
      clearTimeout(room.cleanupTimer);
      room.cleanupTimer = undefined;
    }

    const participant: RoomParticipant = {
      id: participantId,
      name,
      langCode,
      voiceName,
      liveCallActive: false,
      lowBandwidth,
      whisperBoost,
      liveTranslators: new Map(),
      ws,
      joinedAt: Date.now(),
    };
    room.participants.set(participantId, participant);

    // Send authoritative initial state to joining participant
    ws.send(
      JSON.stringify({
        type: "room:init",
        roomCode,
        participants: serializeParticipants(room),
        messages: room.messages.map((m) => ({
          id: m.id,
          speakerId: m.speakerId,
          speakerName: m.speakerName,
          sourceLang: m.sourceLang,
          sourceText: m.sourceText,
          targetLang: langCode,
          translatedText: m.translations[langCode] ?? m.sourceText,
          timestamp: m.timestamp,
        })),
      })
    );

    broadcastRoomPresence(room);
    reconcileAllRoomTranslators(room);

    ws.on("message", async (raw) => {
      try {
        const msg = JSON.parse(raw.toString()) as {
          type?: string;
          langCode?: string;
          voiceName?: string;
          name?: string;
          lowBandwidth?: boolean;
          whisperBoost?: boolean;
          messageId?: string;
          audioBase64?: string;
          mimeType?: string;
          text?: string;
          data?: string;
        };

        if (msg.type === "participant:update") {
          const current = room!.participants.get(participantId);
          if (current) {
            const langChanged =
              msg.langCode && sanitizeLangCode(msg.langCode, false) !== current.langCode;
            const bwChanged =
              typeof msg.lowBandwidth === "boolean" && msg.lowBandwidth !== current.lowBandwidth;
            if (msg.langCode) current.langCode = sanitizeLangCode(msg.langCode, false);
            if (msg.voiceName) current.voiceName = sanitizeVoice(msg.voiceName);
            if (msg.name) current.name = msg.name.slice(0, 40).trim() || current.name;
            if (typeof msg.lowBandwidth === "boolean") current.lowBandwidth = msg.lowBandwidth;
            if (typeof msg.whisperBoost === "boolean") current.whisperBoost = msg.whisperBoost;
            broadcastRoomPresence(room!);
            if (langChanged || bwChanged) {
              closeParticipantTranslators(current);
              reconcileAllRoomTranslators(room!);
            }
          }
          return;
        }

        if (msg.type === "room:live_start") {
          const current = room!.participants.get(participantId);
          if (current) {
            current.liveCallActive = true;
            reconcileParticipantTranslators(room!, current);
            broadcastRoomPresence(room!);
          }
          return;
        }

        if (msg.type === "room:live_stop") {
          const current = room!.participants.get(participantId);
          if (current) {
            current.liveCallActive = false;
            closeParticipantTranslators(current);
            broadcastRoomPresence(room!);
          }
          return;
        }

        if (msg.type === "room:live_audio") {
          const current = room!.participants.get(participantId);
          if (
            current &&
            current.liveCallActive &&
            typeof msg.data === "string" &&
            msg.data.length > 0
          ) {
            for (const ch of current.liveTranslators.values()) {
              if (ch.closed) continue;
              if (ch.mode === "vad_stream") {
                feedRoomChannelVadFrame(room!, current, ch, msg.data);
              } else if (ch.session) {
                ch.session.sendRealtimeInput({
                  audio: { data: msg.data, mimeType: "audio/pcm;rate=16000" },
                });
              }
            }
          }
          return;
        }

        if (msg.type === "room:live_flush") {
          const current = room!.participants.get(participantId);
          if (current && current.liveCallActive) {
            for (const ch of current.liveTranslators.values()) {
              if (ch.closed || ch.mode !== "vad_stream") continue;
              if (ch.vadSpeechActive && ch.vadChunks.length > 0) {
                const totalBytes = ch.vadChunks.reduce((acc, b) => acc + b.length, 0);
                const utterance = ch.vadChunks;
                ch.vadChunks = [];
                ch.vadSpeechActive = false;
                if (totalBytes >= 16000 * 2 * 0.24) {
                  ch.vadQueue.push(utterance);
                  void drainRoomChannelVadQueue(room!, current, ch);
                }
              }
            }
          }
          return;
        }

        if (msg.type === "room:speak") {
          const messageId = msg.messageId || `${participantId}-${Date.now()}`;
          if (room!.processedMessageIds.has(messageId)) return;
          room!.processedMessageIds.add(messageId);

          const speaker = room!.participants.get(participantId);
          if (!speaker) return;

          const hasAudio = typeof msg.audioBase64 === "string" && msg.audioBase64.length > 0;
          const hasText = typeof msg.text === "string" && msg.text.trim().length > 0;
          if (!hasAudio && !hasText) return;

          // Collect all unique target languages needed by connected participants
          const uniqueTargetLangs = new Set<string>();
          for (const p of room!.participants.values()) {
            uniqueTargetLangs.add(p.langCode);
          }
          const targetLangArray = Array.from(uniqueTargetLangs);
          const primaryTarget =
            targetLangArray.find((c) => c !== speaker.langCode) ?? speaker.langCode;

          // Step 1: Transcribe speaker once (and get first translation simultaneously)
          const firstResult = await transcribeAndTranslate({
            audioBase64: hasAudio ? msg.audioBase64 : undefined,
            mimeType: msg.mimeType || "audio/webm",
            text: hasText ? msg.text!.trim() : undefined,
            sourceLangCode: speaker.langCode,
            targetLangCode: primaryTarget,
          });

          if (!firstResult.sourceTranscript) {
            ws.send(JSON.stringify({ type: "room:empty_speech" }));
            return;
          }

          const translationsMap: Record<string, string> = {
            [speaker.langCode]: firstResult.sourceTranscript,
            [primaryTarget]: firstResult.translatedText,
          };

          // Step 2: Translate into any additional participant languages in parallel
          const remainingLangs = targetLangArray.filter(
            (c) => c !== speaker.langCode && c !== primaryTarget
          );
          await Promise.all(
            remainingLangs.map(async (lang) => {
              const t = await translateTextOnly({
                text: firstResult.sourceTranscript,
                sourceLangName: getLanguageByCode(speaker.langCode).name,
                targetLangCode: lang,
              });
              translationsMap[lang] = t;
            })
          );

          // Step 3: Synthesize WAV audio per listener language+voice pair
          const audioCache = new Map<string, string | null>();
          const listenerPairs: Array<{ langCode: string; voiceName: string }> = [];
          const pairSeen = new Set<string>();

          for (const p of room!.participants.values()) {
            // Only synthesize audio for listeners (or different language) to avoid echoing speaker's own voice
            if (p.id === speaker.id) continue;
            const key = `${p.langCode}:${p.voiceName}`;
            if (!pairSeen.has(key)) {
              pairSeen.add(key);
              listenerPairs.push({ langCode: p.langCode, voiceName: p.voiceName });
            }
          }

          await Promise.all(
            listenerPairs.map(async ({ langCode: lc, voiceName: vn }) => {
              const textForAudio = translationsMap[lc] || firstResult.sourceTranscript;
              try {
                const wav = await synthesizeSpeechWav(textForAudio, vn);
                audioCache.set(`${lc}:${vn}`, wav);
              } catch {
                audioCache.set(`${lc}:${vn}`, null);
              }
            })
          );

          const timestamp = Date.now();
          const storedMsg: StoredRoomMessage = {
            id: messageId,
            speakerId: speaker.id,
            speakerName: speaker.name,
            sourceLang: speaker.langCode,
            sourceText: firstResult.sourceTranscript,
            translations: translationsMap,
            timestamp,
          };
          room!.messages.push(storedMsg);
          if (room!.messages.length > 100) {
            room!.messages.shift();
          }

          // Deliver personalized message + target audio to each connected participant
          for (const p of room!.participants.values()) {
            if (p.ws.readyState !== WebSocket.OPEN) continue;
            const isSpeaker = p.id === speaker.id;
            const targetText = translationsMap[p.langCode] ?? firstResult.sourceTranscript;
            const audioWavBase64 = isSpeaker
              ? null
              : audioCache.get(`${p.langCode}:${p.voiceName}`) ?? null;

            p.ws.send(
              JSON.stringify({
                type: "room:message",
                message: {
                  id: messageId,
                  speakerId: speaker.id,
                  speakerName: speaker.name,
                  sourceLang: speaker.langCode,
                  sourceText: firstResult.sourceTranscript,
                  targetLang: p.langCode,
                  translatedText: targetText,
                  audioWavBase64,
                  timestamp,
                },
              })
            );
          }
        }
      } catch (err) {
        const formatted = formatApiError(err);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "error", message: formatted.message }));
        }
      }
    });

    ws.on("close", () => {
      if (!room) return;
      const leaving = room.participants.get(participantId);
      if (leaving) {
        closeParticipantTranslators(leaving);
      }
      room.participants.delete(participantId);
      broadcastRoomPresence(room);
      reconcileAllRoomTranslators(room);
      if (room.participants.size === 0) {
        room.cleanupTimer = setTimeout(() => {
          if (room && room.participants.size === 0) {
            rooms.delete(room.code);
          }
        }, 10 * 60 * 1000);
      }
    });
  });

  server.on("upgrade", (req, socket, head) => {
    const pathname = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`).pathname;
    if (pathname === "/ws/live") {
      liveWss.handleUpgrade(req, socket, head, (ws) => {
        liveWss.emit("connection", ws, req);
      });
    } else if (pathname === "/ws/room") {
      roomWss.handleUpgrade(req, socket, head, (ws) => {
        roomWss.emit("connection", ws, req);
      });
    } else {
      socket.destroy();
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  // Keep WebSockets alive across international mobile networks & reverse proxies
  const heartbeatInterval = setInterval(() => {
    for (const client of liveWss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.ping();
      }
    }
    for (const client of roomWss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.ping();
      }
    }
  }, 25000);

  server.on("close", () => {
    clearInterval(heartbeatInterval);
  });

  server.listen(PORT, "0.0.0.0", () => {
    console.log(`Lissan Live Voice Translation Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
