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
import { WORLD_LANGUAGES, VOICE_PROFILES, getLanguageByCode } from "./src/data/languages.ts";

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

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured on the server.");
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function formatApiError(err: unknown): { status: number; message: string } {
  const msg = err instanceof Error ? err.message : String(err);
  if (msg.includes("403") || msg.includes("PERMISSION_DENIED") || msg.includes("API_KEY_INVALID")) {
    return {
      status: 403,
      message: "Gemini API key permission or configuration error. Please check Settings > Secrets.",
    };
  }
  if (msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED")) {
    return {
      status: 429,
      message: "Rate limit reached. Please wait a moment and try again.",
    };
  }
  return { status: 500, message: msg || "Translation request failed." };
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

  const sourceHint = isBidirectional
    ? `This is a two-way conversation between ${sourceLang.name} (${sourceLang.nativeName}) [code: ${sourceLang.code}] and ${targetLang.name} (${targetLang.nativeName}) [code: ${targetLang.code}]. Detect which of these two languages was used: if it is ${sourceLang.name}, translate into ${targetLang.name}; if it is ${targetLang.name}, translate into ${sourceLang.name}.`
    : sourceLang === null
    ? `Auto-detect the spoken language and translate into ${targetLang.name} (${targetLang.nativeName}).`
    : `The speaker is speaking ${sourceLang.name} (${sourceLang.nativeName}). Translate into ${targetLang.name} (${targetLang.nativeName}).`;

  const promptText = params.audioBase64
    ? `Listen carefully to the audio. ${sourceHint}
1) Transcribe what was said accurately in its original language and native script (e.g., Ge'ez/Ethiopic script for Amharic or Tigrinya, Arabic script for Arabic, Chinese characters for Chinese). If the audio is silent or contains no intelligible speech, return empty strings for all fields.
2) Translate the meaning naturally and idiomatically into the destination language.
Return strict JSON.`
    : `${sourceHint}
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

  const response = await ai.models.generateContent({
    model: "gemini-4-hydrogen",
    contents: { parts },
    config: {
      thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          sourceTranscript: {
            type: Type.STRING,
            description: "Accurate transcription of the spoken input in its original script.",
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
            description: "Natural, faithful translation in the destination language.",
          },
        },
        required: ["sourceTranscript", "detectedLanguage", "translatedText"],
      },
    },
  });

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

async function translateTextOnly(params: {
  text: string;
  sourceLangName: string;
  targetLangCode: string;
}): Promise<string> {
  const ai = getAiClient();
  const targetLang = getLanguageByCode(params.targetLangCode);
  const response = await ai.models.generateContent({
    model: "gemini-4-hydrogen",
    contents: `Translate this ${params.sourceLangName} text naturally and directly into ${targetLang.name} (${targetLang.nativeName}). Return ONLY the translated text with no extra commentary or quotation marks.\n\nText: ${params.text}`,
    config: {
      thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
    },
  });
  return (response.text ?? "").trim();
}

async function synthesizeSpeechWav(text: string, voiceName: string): Promise<string | null> {
  if (!text.trim()) return null;
  const ai = getAiClient();
  const safeVoice = sanitizeVoice(voiceName);
  const response = await ai.models.generateContent({
    model: "gemini-3.8-flash-lite-tts",
    contents: [
      {
        role: "user",
        parts: [
          {
            text,
            speechMetadata: {
              style: "Clear, warm, natural simultaneous voice interpreter",
            },
          },
        ],
      },
    ],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: safeVoice },
        },
      },
    },
  });

  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  return base64Audio ?? null;
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
  inputBuf: string;
  outputBuf: string;
}

interface RoomParticipant {
  id: string;
  name: string;
  langCode: string;
  voiceName: string;
  liveCallActive: boolean;
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

async function openRoomTranslatorChannel(
  room: RoomState,
  speaker: RoomParticipant,
  targetLangCode: string
): Promise<void> {
  const channel = speaker.liveTranslators.get(targetLangCode);
  if (!channel || channel.closed) return;

  const targetLang = getLanguageByCode(targetLangCode);
  const sourceLang = getLanguageByCode(speaker.langCode);
  let usedFallback = false;

  const connectWithModel = async (useTranslateModel: boolean): Promise<LiveSessionHandle> => {
    const ai = getAiClient();
    const model = useTranslateModel ? "gemini-3.5-live-translate-preview" : "gemini-3.8-live";
    const config = useTranslateModel
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
          systemInstruction: `You are a simultaneous voice interpreter on a live phone call. Listen to ${speaker.name} speaking ${sourceLang.name} (${sourceLang.nativeName}) and immediately speak ONLY the natural, accurate translation in ${targetLang.name} (${targetLang.nativeName}). Never reply to the speaker or add commentary.`,
        };

    return ai.live.connect({
      model,
      config,
      callbacks: {
        onmessage: (message: LiveServerMessage) => {
          if (channel.closed) return;
          const content = message.serverContent;

          if (content?.inputTranscription?.text) {
            channel.inputBuf += content.inputTranscription.text;
            // Send live source caption back to speaker and target listeners
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
        onerror: async () => {
          if (channel.closed) return;
          if (useTranslateModel && !usedFallback) {
            usedFallback = true;
            try {
              channel.session = await connectWithModel(false);
            } catch {
              // ignore
            }
          }
        },
        onclose: async (event: unknown) => {
          if (channel.closed) return;
          const reason =
            typeof event === "object" && event && "reason" in event
              ? String((event as { reason?: string }).reason || "")
              : "";
          if (useTranslateModel && !usedFallback && reason.toLowerCase().includes("language")) {
            usedFallback = true;
            try {
              channel.session = await connectWithModel(false);
            } catch {
              // ignore
            }
          }
        },
      },
    });
  };

  try {
    channel.session = await connectWithModel(targetLang.liveTranslateSupported);
  } catch {
    if (targetLang.liveTranslateSupported && !usedFallback) {
      usedFallback = true;
      try {
        channel.session = await connectWithModel(false);
      } catch (err) {
        const formatted = formatApiError(err);
        if (speaker.ws.readyState === WebSocket.OPEN) {
          speaker.ws.send(JSON.stringify({ type: "error", message: formatted.message }));
        }
      }
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
        inputBuf: "",
        outputBuf: "",
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

async function startServer() {
  const app = express();
  app.set("trust proxy", 1);
  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    next();
  });
  app.use(express.json({ limit: "15mb" }));

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
        ttsWarning,
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
  // --------------------------------------------------------------------------
  liveWss.on("connection", async (clientWs, req) => {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const targetCode = sanitizeLangCode(url.searchParams.get("targetLang") || "en", false);
    const sourceCode = sanitizeLangCode(url.searchParams.get("sourceLang") || "auto", true);
    const voiceName = sanitizeVoice(url.searchParams.get("voiceName") || "Zephyr");
    const bidirectionalParam = url.searchParams.get("bidirectional") === "true";

    const targetLang = getLanguageByCode(targetCode);
    const sourceLang = sourceCode === "auto" ? null : getLanguageByCode(sourceCode);
    const isBidirectional =
      bidirectionalParam && sourceLang !== null && sourceLang.code !== targetLang.code;
    const sourceLangLabel = sourceLang ? sourceLang.name : "any spoken language";

    let activeSession: Awaited<ReturnType<GoogleGenAI["live"]["connect"]>> | null = null;
    let isClosed = false;
    let usedFallback = false;

    const connectLiveSession = async (useTranslateModel: boolean) => {
      const ai = getAiClient();
      const model = useTranslateModel ? "gemini-3.5-live-translate-preview" : "gemini-3.8-live";

      const systemInstruction = isBidirectional && sourceLang
        ? `You are a seamless two-way simultaneous voice interpreter between ${sourceLang.name} (${sourceLang.nativeName}) and ${targetLang.name} (${targetLang.nativeName}). Listen to each spoken phrase: if it is in ${sourceLang.name}, immediately speak ONLY the natural translation in ${targetLang.name}. If it is in ${targetLang.name}, immediately speak ONLY the natural translation in ${sourceLang.name}. Never reply to the content or add commentary.`
        : `You are a simultaneous voice interpreter. Listen to the speaker (${sourceLangLabel}) and immediately translate everything they say aloud into ${targetLang.name} (${targetLang.nativeName}). Speak ONLY the direct translation naturally—do not answer questions, do not explain, and do not add commentary.`;

      const config = useTranslateModel
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

      const session = await ai.live.connect({
        model,
        config,
        callbacks: {
          onopen: () => {
            if (!isClosed && clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(
                JSON.stringify({
                  type: "status",
                  connected: true,
                  model,
                  targetLang: targetLang.code,
                })
              );
            }
          },
          onmessage: (message: LiveServerMessage) => {
            if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
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
          onerror: async (err: unknown) => {
            if (isClosed) return;
            if (useTranslateModel && !usedFallback) {
              usedFallback = true;
              try {
                activeSession = await connectLiveSession(false);
                return;
              } catch (fallbackErr) {
                const formatted = formatApiError(fallbackErr);
                if (clientWs.readyState === WebSocket.OPEN) {
                  clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
                }
              }
            } else {
              const formatted = formatApiError(err);
              if (clientWs.readyState === WebSocket.OPEN) {
                clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
              }
            }
          },
          onclose: async (event: unknown) => {
            if (isClosed) return;
            const reason =
              typeof event === "object" && event && "reason" in event
                ? String((event as { reason?: string }).reason || "")
                : "";
            if (
              useTranslateModel &&
              !usedFallback &&
              reason.toLowerCase().includes("language")
            ) {
              usedFallback = true;
              try {
                activeSession = await connectLiveSession(false);
                return;
              } catch {
                // fall through to notify client
              }
            }
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ type: "status", connected: false, reason }));
            }
          },
        },
      });

      return session;
    };

    const useTranslateInitially = !isBidirectional && targetLang.liveTranslateSupported;
    try {
      activeSession = await connectLiveSession(useTranslateInitially);
    } catch (err) {
      if (useTranslateInitially && !usedFallback) {
        usedFallback = true;
        try {
          activeSession = await connectLiveSession(false);
        } catch (fallbackErr) {
          const formatted = formatApiError(fallbackErr);
          if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
            clientWs.close();
          }
          return;
        }
      } else {
        const formatted = formatApiError(err);
        if (clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(JSON.stringify({ type: "error", message: formatted.message }));
          clientWs.close();
        }
        return;
      }
    }

    clientWs.on("message", (raw) => {
      if (isClosed || !activeSession) return;
      try {
        const parsed = JSON.parse(raw.toString()) as { type?: string; data?: string };
        if (parsed.type === "audio" && typeof parsed.data === "string" && parsed.data.length > 0) {
          activeSession.sendRealtimeInput({
            audio: { data: parsed.data, mimeType: "audio/pcm;rate=16000" },
          });
        }
      } catch {
        // Ignore malformed frames
      }
    });

    clientWs.on("close", () => {
      isClosed = true;
      try {
        activeSession?.close();
      } catch {
        // ignore
      }
    });
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
            if (msg.langCode) current.langCode = sanitizeLangCode(msg.langCode, false);
            if (msg.voiceName) current.voiceName = sanitizeVoice(msg.voiceName);
            if (msg.name) current.name = msg.name.slice(0, 40).trim() || current.name;
            broadcastRoomPresence(room!);
            if (langChanged) {
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
              if (!ch.closed && ch.session) {
                ch.session.sendRealtimeInput({
                  audio: { data: msg.data, mimeType: "audio/pcm;rate=16000" },
                });
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
