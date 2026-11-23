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

function isServerBusyError(err: unknown): boolean {
  const msg = extractErrorMessage(err).toUpperCase();
  return (
    msg.includes("503") ||
    msg.includes("UNAVAILABLE") ||
    msg.includes("HIGH DEMAND")
  );
}

function isRateLimitError(err: unknown): boolean {
  const msg = extractErrorMessage(err).toUpperCase();
  return (
    msg.includes("429") ||
    msg.includes("RESOURCE_EXHAUSTED") ||
    msg.includes("RATE_LIMIT") ||
    msg.includes("QUOTA")
  );
}

function isRetryableQuotaOrLoadError(err: unknown): boolean {
  return isServerBusyError(err) || isRateLimitError(err);
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

  // Never amplify silent room hiss/static (< 240 peak); cap gain at 3.2x so noise stays quiet
  if (peak < 240 || peak >= 7500) {
    return pcmBuffer;
  }

  const gain = Math.min(3.2, 16000 / peak);
  const out = Buffer.allocUnsafe(pcmBuffer.length);
  for (let i = 0; i < sampleCount; i++) {
    const boosted = Math.round(pcmBuffer.readInt16LE(i * 2) * gain);
    const clamped = Math.max(-32768, Math.min(32767, boosted));
    out.writeInt16LE(clamped, i * 2);
  }
  return out;
}

/**
 * Acoustic gate before invoking Gemini on a queued utterance:
 * rejects desk taps, fan hum, breathing, or faint static so Gemini only sees real speech.
 */
function hasValidSpeechEnergy(
  chunks: Buffer[],
  noiseFloor: number,
  whisperBoost: boolean
): boolean {
  let totalBytes = 0;
  let peakAbs = 0;
  let maxChunkRms = 0;
  let activeFrames = 0;
  const minRms = Math.max(whisperBoost ? 0.0035 : 0.0058, noiseFloor * 1.9);
  const minChunkPeak = whisperBoost ? 260 : 450;

  for (const chunk of chunks) {
    totalBytes += chunk.length;
    const chunkRms = computePcm16Rms(chunk);
    if (chunkRms > maxChunkRms) maxChunkRms = chunkRms;

    let chunkPeak = 0;
    const sampleCount = Math.floor(chunk.length / 2);
    for (let i = 0; i < sampleCount; i++) {
      const abs = Math.abs(chunk.readInt16LE(i * 2));
      if (abs > chunkPeak) chunkPeak = abs;
    }
    if (chunkPeak > peakAbs) peakAbs = chunkPeak;
    if (chunkRms >= minRms && chunkPeak >= minChunkPeak) {
      activeFrames += 1;
    }
  }

  if (totalBytes < 16000 * 2 * 0.18) return false;
  if (peakAbs < minChunkPeak) return false;
  if (maxChunkRms < minRms) return false;
  if (activeFrames < 1) return false;
  return true;
}

function isNoiseOrHallucination(sourceText: string, translatedText: string): boolean {
  const src = sourceText.trim();
  const tr = translatedText.trim();
  if (!src && !tr) return true;

  // Reject bracketed sound descriptions such as [inaudible], [noise], (music), *cough*
  const stripBrackets = (s: string) =>
    s
      .replace(/\[[^\]]*\]/g, "")
      .replace(/\([^)]*\)/g, "")
      .replace(/\*[^*]*\*/g, "")
      .replace(/[\s.,!?;:'"…\-–—።፣፤፥]+/g, "")
      .trim();

  const srcCore = stripBrackets(src);
  const trCore = stripBrackets(tr);
  if (srcCore.length < 2 || trCore.length < 1) return true;

  const noiseTagPattern =
    /^(none|inaudible|unintelligible|silence|no speech|noise|music|applause|breathing|static|background noise|cough|laughter)$/i;
  if (noiseTagPattern.test(srcCore) || noiseTagPattern.test(trCore)) return true;

  return false;
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
    ? `STRICT HEARING-ONLY RULE: Listen strictly to the actual audible speech in this audio clip. ${sourceHint}${ethiopicRules}
1) Transcribe ONLY words that are genuinely and clearly spoken in the audio—even if quiet/low-tone. NEVER guess, hallucinate, complete unspoken sentences, or invent phantom greetings/phrases (like "Thank you", "Hello", "You", "ሰላም", "እሺ", or "[inaudible]") from background noise, breathing, room hiss, clicks, or silence.
2) If there are NO clearly audible human spoken words, return empty strings ("") for ALL fields immediately.
3) Translate ONLY what was actually spoken into the destination language cleanly and naturally.
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
              description: "Exact word-for-word transcription in original script, or empty string if silent/noise.",
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
              description: "Complete translation of spoken words only, or empty string if silent/noise.",
            },
          },
          required: ["sourceTranscript", "detectedLanguage", "translatedText"],
        },
      },
    });
  };

  let response;
  try {
    response = await callModel("gemini-3.1-flash-lite", ThinkingLevel.MINIMAL);
  } catch (err) {
    if (isServerBusyError(err)) {
      try {
        response = await callModel("gemini-4-hydrogen", ThinkingLevel.LOW);
      } catch (err2) {
        if (isServerBusyError(err2)) {
          await new Promise((r) => setTimeout(r, 350));
          response = await callModel("gemini-4-hydrogen", ThinkingLevel.LOW);
        } else {
          throw err2;
        }
      }
    } else if (isRateLimitError(err)) {
      response = await callModel("gemini-4-hydrogen", ThinkingLevel.LOW);
    } else {
      throw err;
    }
  }

  const raw = (response.text ?? "{}").trim();
  const parsed = JSON.parse(raw) as Partial<TranslatedTurnResult>;
  const sourceTranscript = (parsed.sourceTranscript ?? "").trim();
  const translatedText = (parsed.translatedText ?? "").trim();
  if (params.audioBase64 && isNoiseOrHallucination(sourceTranscript, translatedText)) {
    return { sourceTranscript: "", detectedLanguage: "", translatedText: "" };
  }
  return {
    sourceTranscript,
    detectedLanguage: (parsed.detectedLanguage ?? "").trim(),
    translatedText,
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

  const trMatch = clean.match(/(?:^|\n)TR:\s*([\s\S]*?)(?=\nSRC:|$)/);
  const srcMatch = clean.match(/(?:^|\n)SRC:\s*([\s\S]*?)(?=\nTR:|$)/);

  let sourceTranscript = srcMatch ? srcMatch[1].trim() : "";
  let translatedText = trMatch ? trMatch[1].trim() : "";

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
  onTranslationComplete?: (translatedText: string, resolvedTargetLang?: string) => void;
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

  const promptText = `STRICT HEARING-ONLY RULE: Listen strictly to what is actually spoken in this audio clip. ${directionPrompt}${ethiopicRules}
Transcribe ONLY clearly audible human spoken words in native script, and translate only what was heard.
NEVER guess, hallucinate, or invent words, greetings ("Hello", "Thank you", "ሰላም", "እሺ"), sound tags ("[inaudible]", "[noise]"), or phrases from background noise, breathing, mic clicks, or silence.
If you do NOT clearly hear real human speech, output ONLY: NONE
Otherwise, output ONLY this exact 3-line format (no markdown, no extra text):
LANG: <detected BCP-47 source code>|<target BCP-47 code>|<Language English Name>
TR: <complete natural translation>
SRC: <exact word-for-word transcript>`;

  const parts = [
    {
      inlineData: {
        mimeType: params.mimeType || "audio/wav",
        data: params.audioBase64,
      },
    },
    { text: promptText },
  ];

  const runStreamModel = async (
    modelName: string,
    thinkingLevel: ThinkingLevel
  ): Promise<TranslatedTurnResult> => {
    const stream = await ai.models.generateContentStream({
      model: modelName,
      contents: { parts },
      config: {
        thinkingConfig: { thinkingLevel },
      },
    });
    let fullText = "";
    let lastSrcSent = "";
    let lastTrSent = "";
    let trFired = false;

    for await (const chunk of stream) {
      if (chunk.text) {
        fullText += chunk.text;
        const parsed = parseStreamingTaggedOutput(fullText);

        // Fire TTS immediately as soon as TR line completes (when SRC: starts arriving)
        if (
          !trFired &&
          params.onTranslationComplete &&
          fullText.includes("\nSRC:") &&
          parsed.translatedText.trim().length >= 1 &&
          !isNoiseOrHallucination(
            parsed.translatedText,
            parsed.translatedText
          )
        ) {
          trFired = true;
          params.onTranslationComplete(
            parsed.translatedText,
            parsed.resolvedTargetLang
          );
        }

        if (params.onPartial) {
          const srcLen = parsed.sourceTranscript.trim().length;
          const trLen = parsed.translatedText.trim().length;
          const hasValidPartial =
            (srcLen >= 2 || trLen >= 2) &&
            !isNoiseOrHallucination(
              parsed.sourceTranscript || parsed.translatedText,
              parsed.translatedText || parsed.sourceTranscript
            );
          if (
            hasValidPartial &&
            (parsed.sourceTranscript !== lastSrcSent ||
              parsed.translatedText !== lastTrSent)
          ) {
            lastSrcSent = parsed.sourceTranscript;
            lastTrSent = parsed.translatedText;
            params.onPartial(parsed);
          }
        }
      }
    }

    const finalParsed = parseStreamingTaggedOutput(fullText);
    if (isNoiseOrHallucination(finalParsed.sourceTranscript, finalParsed.translatedText)) {
      return { sourceTranscript: "", detectedLanguage: "", translatedText: "" };
    }
    if (!trFired && params.onTranslationComplete && finalParsed.translatedText.trim()) {
      params.onTranslationComplete(
        finalParsed.translatedText,
        finalParsed.resolvedTargetLang
      );
    }
    return finalParsed;
  };

  try {
    return await runStreamModel("gemini-3.1-flash-lite", ThinkingLevel.MINIMAL);
  } catch (err) {
    if (isServerBusyError(err)) {
      try {
        return await runStreamModel("gemini-4-hydrogen", ThinkingLevel.LOW);
      } catch (err2) {
        if (isServerBusyError(err2)) {
          // Dedicated STT model fallback when both general multimodal models see 503 load
          const sttResp = await ai.models.generateContent({
            model: "gemini-3.5-transcribe",
            contents: {
              parts: [
                {
                  inlineData: {
                    mimeType: params.mimeType || "audio/wav",
                    data: params.audioBase64,
                  },
                },
                {
                  text: `Transcribe only clearly audible human speech word-for-word in its native script. Output empty string if only noise or silence.`,
                },
              ],
            },
          });
          const transcript = (sttResp.text ?? "").trim();
          if (!transcript || isNoiseOrHallucination(transcript, transcript)) {
            return { sourceTranscript: "", detectedLanguage: "", translatedText: "" };
          }
          const translated = await translateTextOnly({
            text: transcript,
            sourceLangName: sourceLang ? sourceLang.name : "spoken",
            targetLangCode: targetLang.code,
          });
          const res: TranslatedTurnResult = {
            sourceTranscript: transcript,
            detectedLanguage: sourceLang ? sourceLang.name : "Auto",
            translatedText: translated,
            resolvedSourceLang: sourceLang?.code,
            resolvedTargetLang: targetLang.code,
          };
          if (params.onPartial) params.onPartial(res);
          if (params.onTranslationComplete && translated) {
            params.onTranslationComplete(translated, targetLang.code);
          }
          return res;
        }
        throw err2;
      }
    }
    if (isRateLimitError(err)) {
      // Single non-amplifying hop to alternate model bucket without multi-step generateContent storm
      return await runStreamModel("gemini-4-hydrogen", ThinkingLevel.LOW);
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
      model: "gemini-3.1-flash-lite",
      contents: prompt,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
      },
    });
  } catch (err) {
    if (isRetryableQuotaOrLoadError(err)) {
      await new Promise((r) => setTimeout(r, 250));
      response = await ai.models.generateContent({
        model: "gemini-4-hydrogen",
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
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

const ttsModelCooldownUntil = new Map<string, number>();

function isTtsModelCoolingDown(model: string): boolean {
  const until = ttsModelCooldownUntil.get(model);
  return typeof until === "number" && Date.now() < until;
}

function recordTtsModelFailure(model: string, err: unknown): void {
  const msg = extractErrorMessage(err);
  if (isRateLimitError(err)) {
    const isDaily = /PerDay|GenerateRequestsPerDay/i.test(msg);
    // Skip daily-exhausted free-tier TTS model for 30 minutes (or 60s for RPM) so live turns have zero 429 stalls
    const cooldownMs = isDaily ? 30 * 60 * 1000 : 60 * 1000;
    ttsModelCooldownUntil.set(model, Date.now() + cooldownMs);
  } else if (isServerBusyError(err)) {
    ttsModelCooldownUntil.set(model, Date.now() + 15 * 1000);
  }
}

function pcm24kChunksToWavBase64(chunks: Buffer[]): string | null {
  if (chunks.length === 0) return null;
  const pcmData = Buffer.concat(chunks);
  if (pcmData.length === 0) return null;
  const header = Buffer.alloc(44);
  const sampleRate = 24000;
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

function isSchemaFieldError(err: unknown): boolean {
  const msg = extractErrorMessage(err);
  return (
    msg.includes("INVALID_ARGUMENT") ||
    msg.includes("speechMetadata") ||
    msg.includes("speechConfig")
  );
}

/**
 * Real-time Live API streaming TTS fallback using gemini-3.8-live.
 * Ensures crisp 24kHz Gemini voice output even when Free Tier generateContent TTS (10 RPD) hits 429.
 */
async function synthesizeViaLiveAudio(
  text: string,
  voiceName: string,
  onPcm24kChunk: (pcm24kBuffer: Buffer) => void
): Promise<boolean> {
  const cleaned = text.trim();
  if (!cleaned) return false;
  const ai = getAiClient("v1beta");
  const safeVoice = sanitizeVoice(voiceName);

  const attemptLiveTts = (): Promise<boolean> =>
    new Promise<boolean>((resolve) => {
      let settled = false;
      let emitted = false;
      let session: LiveSessionHandle | null = null;
      let inactivityTimer: ReturnType<typeof setTimeout> | null = null;

      const finish = (ok: boolean) => {
        if (settled) return;
        settled = true;
        if (inactivityTimer) clearTimeout(inactivityTimer);
        clearTimeout(hardTimeout);
        try {
          session?.close();
        } catch {
          // ignore
        }
        resolve(ok);
      };

      const bumpInactivity = () => {
        if (inactivityTimer) clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(() => finish(emitted), emitted ? 1500 : 3500);
      };

      const hardTimeout = setTimeout(() => finish(emitted), 9500);

      ai.live
        .connect({
          model: "gemini-3.8-live",
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: safeVoice } },
            },
            systemInstruction:
              "You are a pure text-to-speech voice engine. Immediately read aloud ONLY the exact phrase enclosed in <read> tags, verbatim, clearly, and naturally in the phrase's own language. Never translate, answer, or add any extra words.",
          },
          callbacks: {
            onopen: () => {
              // Wait until text is transmitted in .then() before starting initial wait timer
            },
            onmessage: (message: LiveServerMessage) => {
              if (settled) return;
              const content = message.serverContent;
              for (const part of content?.modelTurn?.parts ?? []) {
                if (part.inlineData?.data) {
                  emitted = true;
                  bumpInactivity();
                  onPcm24kChunk(Buffer.from(part.inlineData.data, "base64"));
                }
              }
              if (content?.turnComplete) {
                finish(emitted);
              }
            },
            onerror: () => finish(emitted),
            onclose: () => finish(emitted),
          },
        })
        .then((connected) => {
          session = connected;
          if (settled) {
            try {
              connected.close();
            } catch {
              // ignore
            }
            return;
          }
          bumpInactivity();
          connected.sendRealtimeInput({
            text: `<read>${cleaned}</read>`,
          });
        })
        .catch(() => finish(false));
    });

  const first = await attemptLiveTts();
  if (first) return true;
  await new Promise((r) => setTimeout(r, 150));
  return await attemptLiveTts();
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

  for (const model of ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"]) {
    if (isTtsModelCoolingDown(model)) continue;
    try {
      const response = await executeWithFallbacks(model);
      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) return base64Audio;
    } catch (err) {
      recordTtsModelFailure(model, err);
      if (!isRetryableQuotaOrLoadError(err) && !isSchemaFieldError(err)) {
        break;
      }
    }
  }

  // Fallback to Live Audio synthesis so user still hears natural 24kHz Gemini voice even on TTS quota limits
  const liveChunks: Buffer[] = [];
  const liveOk = await synthesizeViaLiveAudio(text, safeVoice, (buf) => {
    liveChunks.push(buf);
  });
  if (liveOk) {
    return pcm24kChunksToWavBase64(liveChunks);
  }
  return null;
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

  for (const model of ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"]) {
    if (isTtsModelCoolingDown(model)) continue;
    try {
      const ok = await executeStream(model);
      if (ok) return true;
    } catch (err) {
      recordTtsModelFailure(model, err);
      if (!isRetryableQuotaOrLoadError(err) && !isSchemaFieldError(err)) {
        break;
      }
    }
  }

  // Fallback directly to gemini-3.8-live WebSocket streaming TTS (no 10 RPD limit, <600ms first audio)
  return await synthesizeViaLiveAudio(text, safeVoice, onPcm24kChunk);
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
  vadConfirmCount: number;
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
      if (!hasValidSpeechEnergy(utterance, channel.vadNoiseFloor, speaker.whisperBoost)) {
        continue;
      }
      const wavBase64 = pcm16ChunksToWavBase64(utterance, 16000);
      let ttsPromise: Promise<boolean> | null = null;
      let ttsTextStarted = "";

      const startRoomEarlyTts = (textToSpeak: string) => {
        const cleaned = textToSpeak.trim();
        if (!cleaned || ttsPromise || channel.closed) return;
        ttsTextStarted = cleaned;
        let leftoverLow: Buffer | null = null;
        ttsPromise = synthesizeSpeechStream(cleaned, speaker.voiceName, (pcm24kBuf) => {
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
        }).catch(() => false);
      };

      const runRoomTurn = () =>
        transcribeAndTranslateStream({
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
          onTranslationComplete: (trText) => {
            if (!channel.closed) {
              startRoomEarlyTts(trText);
            }
          },
        });

      try {
        let result: TranslatedTurnResult;
        try {
          result = await runRoomTurn();
        } catch (firstErr) {
          if (isServerBusyError(firstErr)) {
            await new Promise((r) => setTimeout(r, 250));
            result = await runRoomTurn();
          } else {
            throw firstErr;
          }
        }

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

        if (!ttsPromise) {
          startRoomEarlyTts(result.translatedText);
        }
        const streamed = ttsPromise ? await ttsPromise : false;

        if (ttsTextStarted && !streamed && !channel.closed) {
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
  const minFloor = speaker.whisperBoost ? 0.0035 : 0.0058;
  const multiplier = speaker.whisperBoost ? 2.1 : 2.4;
  const SPEECH_THRESHOLD = Math.max(minFloor, channel.vadNoiseFloor * multiplier);
  const SILENCE_HANG_MS = 280;
  const MAX_UTTERANCE_MS = 3800;
  const MIN_UTTERANCE_BYTES = 16000 * 2 * 0.18;

  if (!channel.vadSpeechActive) {
    if (rms < SPEECH_THRESHOLD) {
      channel.vadConfirmCount = 0;
      if (rms < minFloor * 1.6) {
        const updated = channel.vadNoiseFloor * 0.94 + rms * 0.06;
        channel.vadNoiseFloor = Math.max(0.002, Math.min(0.035, updated));
      }
      channel.vadPreRoll.push(pcmBuf);
      if (channel.vadPreRoll.length > 5) channel.vadPreRoll.shift();
    } else {
      channel.vadConfirmCount += 1;
      channel.vadPreRoll.push(pcmBuf);
      if (channel.vadPreRoll.length > 6) channel.vadPreRoll.shift();
      if (channel.vadConfirmCount >= 2) {
        channel.vadSpeechActive = true;
        channel.vadConfirmCount = 0;
        channel.vadSpeechStartAt = now;
        channel.vadLastVoiceAt = now;
        channel.vadChunks = [...channel.vadPreRoll];
        channel.vadPreRoll = [];
      }
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
    channel.vadConfirmCount = 0;
    if (
      totalBytes >= MIN_UTTERANCE_BYTES &&
      hasValidSpeechEnergy(utterance, channel.vadNoiseFloor, speaker.whisperBoost)
    ) {
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
        vadConfirmCount: 0,
        vadSpeechStartAt: 0,
        vadLastVoiceAt: 0,
        vadNoiseFloor: 0.005,
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
    const engineParam = url.searchParams.get("engine") || "auto";

    const targetLang = getLanguageByCode(targetCode);
    const sourceLang = sourceCode === "auto" ? null : getLanguageByCode(sourceCode);
    const isBidirectional =
      bidirectionalParam && sourceLang !== null && sourceLang.code !== targetLang.code;
    const sourceLangLabel = sourceLang ? sourceLang.name : "any spoken language";

    const usePrecisionEngineInitially =
      engineParam === "vad"
        ? true
        : engineParam === "live38" || engineParam === "translate35"
        ? false
        : isBidirectional ||
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
    let liveInAcc = "";
    let liveOutAcc = "";

    // Continuous Precision Streaming VAD state
    let vadPreRoll: Buffer[] = [];
    let vadChunks: Buffer[] = [];
    let vadSpeechActive = false;
    let vadConfirmCount = 0;
    let vadSpeechStartAt = 0;
    let vadLastVoiceAt = 0;
    let vadNoiseFloor = 0.005;
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
          if (!hasValidSpeechEnergy(utterance, vadNoiseFloor, whisperBoost)) {
            continue;
          }
          const wavBase64 = pcm16ChunksToWavBase64(utterance, 16000);
          let emittedPartial = false;
          let ttsPromise: Promise<boolean> | null = null;
          let ttsTextStarted = "";

          const startEarlyTts = (textToSpeak: string) => {
            const cleaned = textToSpeak.trim();
            if (!cleaned || ttsPromise) return;
            ttsTextStarted = cleaned;
            let leftoverLow: Buffer | null = null;
            ttsPromise = synthesizeSpeechStream(cleaned, voiceName, (pcm24kBuf) => {
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
            }).catch(() => false);
          };

          const processTurnOnce = () =>
            transcribeAndTranslateStream({
              audioBase64: wavBase64,
              mimeType: "audio/wav",
              sourceLangCode: sourceCode,
              targetLangCode: targetLang.code,
              bidirectional: isBidirectional,
              onPartial: (partial) => {
                if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
                if (partial.sourceTranscript) {
                  emittedPartial = true;
                  clientWs.send(
                    JSON.stringify({
                      type: "inputTranscript",
                      text: partial.sourceTranscript,
                      replace: true,
                    })
                  );
                }
                if (partial.translatedText) {
                  emittedPartial = true;
                  clientWs.send(
                    JSON.stringify({
                      type: "outputTranscript",
                      text: partial.translatedText,
                      replace: true,
                    })
                  );
                }
              },
              onTranslationComplete: (trText) => {
                if (isClosed || clientWs.readyState !== WebSocket.OPEN) return;
                startEarlyTts(trText);
              },
            });

          try {
            let result: TranslatedTurnResult;
            try {
              result = await processTurnOnce();
            } catch (firstErr) {
              if (isServerBusyError(firstErr)) {
                await new Promise((r) => setTimeout(r, 250));
                result = await processTurnOnce();
              } else {
                throw firstErr;
              }
            }

            if (isClosed || clientWs.readyState !== WebSocket.OPEN) break;
            if (!result.sourceTranscript && !result.translatedText) {
              if (emittedPartial) {
                clientWs.send(JSON.stringify({ type: "turnDiscarded" }));
              }
              continue;
            }

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
              if (!ttsPromise) {
                startEarlyTts(result.translatedText);
              }
              const streamed = ttsPromise ? await ttsPromise : false;
              if (ttsTextStarted && !streamed && !isClosed && clientWs.readyState === WebSocket.OPEN) {
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
              // Automatically fail over to persistent Gemini Live WebSocket when generateContent hits 429!
              if (formatted.status === 429 && mode === "vad_stream" && !isClosed) {
                try {
                  mode = "gemini_live";
                  const useTranslateModel = !isBidirectional && targetLang.liveTranslateSupported;
                  activeSession = await connectLiveSession(useTranslateModel, true);
                  sendConnectedStatus(
                    useTranslateModel ? "gemini-3.5-live-failover" : "gemini-3.8-live-failover"
                  );
                  for (const chunk of utterance) {
                    if (isClosed) break;
                    activeSession.sendRealtimeInput({
                      audio: { data: chunk.toString("base64"), mimeType: "audio/pcm;rate=16000" },
                    });
                    await new Promise((r) => setTimeout(r, 20));
                  }
                  const zeroB64 = Buffer.alloc(2048).toString("base64");
                  for (let i = 0; i < 10; i++) {
                    if (isClosed) break;
                    activeSession.sendRealtimeInput({
                      audio: { data: zeroB64, mimeType: "audio/pcm;rate=16000" },
                    });
                    await new Promise((r) => setTimeout(r, 20));
                  }
                  continue;
                } catch {
                  mode = "vad_stream";
                }
              }
              clientWs.send(
                JSON.stringify({
                  type: "notice",
                  level: "traffic",
                  message: formatted.message,
                })
              );
              await new Promise((r) => setTimeout(r, 700));
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
      const minFloor = whisperBoost ? 0.0035 : 0.0058;
      const multiplier = whisperBoost ? 2.1 : 2.4;
      const SPEECH_THRESHOLD = Math.max(minFloor, vadNoiseFloor * multiplier);
      const SILENCE_HANG_MS = 280;
      const MAX_UTTERANCE_MS = 3800;
      const MIN_UTTERANCE_BYTES = 16000 * 2 * 0.18;

      if (!vadSpeechActive) {
        if (rms < SPEECH_THRESHOLD) {
          vadConfirmCount = 0;
          if (rms < minFloor * 1.6) {
            const updated = vadNoiseFloor * 0.94 + rms * 0.06;
            vadNoiseFloor = Math.max(0.002, Math.min(0.035, updated));
          }
          vadPreRoll.push(pcmBuf);
          if (vadPreRoll.length > 5) vadPreRoll.shift();
        } else {
          vadConfirmCount += 1;
          vadPreRoll.push(pcmBuf);
          if (vadPreRoll.length > 6) vadPreRoll.shift();
          if (vadConfirmCount >= 2) {
            vadSpeechActive = true;
            vadConfirmCount = 0;
            vadSpeechStartAt = now;
            vadLastVoiceAt = now;
            vadChunks = [...vadPreRoll];
            vadPreRoll = [];
          }
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
        vadConfirmCount = 0;
        if (
          totalBytes >= MIN_UTTERANCE_BYTES &&
          hasValidSpeechEnergy(utterance, vadNoiseFloor, whisperBoost)
        ) {
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
      vadConfirmCount = 0;
      if (
        totalBytes >= 16000 * 2 * 0.18 &&
        hasValidSpeechEnergy(utterance, vadNoiseFloor, whisperBoost)
      ) {
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

    // Pure digital silence buffer (64ms @ 16kHz) used to signal clean end-of-speech to Gemini Live
    const DIGITAL_SILENCE_B64 = Buffer.alloc(2048).toString("base64");

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
        } else if (parsed.type === "vad_flush") {
          if (mode === "vad_stream") {
            flushVad();
          } else if (activeSession) {
            // Feed clean digital zeros (~512ms) so Live VAD finishes turn promptly without ambient noise
            for (let i = 0; i < 8; i++) {
              activeSession.sendRealtimeInput({
                audio: { data: DIGITAL_SILENCE_B64, mimeType: "audio/pcm;rate=16000" },
              });
            }
          }
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
          ? `STRICT HEARING-ONLY INTERPRETER: Listen strictly to ${sourceLang.name} (${sourceLang.nativeName}) and ${targetLang.name} (${targetLang.nativeName}). Translate ONLY clearly audible spoken words; NEVER invent or hallucinate words during silence, breathing, clicks, or background noise.`
          : `STRICT HEARING-ONLY INTERPRETER: Listen strictly to ${sourceLangLabel} and translate ONLY clearly audible spoken words into ${targetLang.name} (${targetLang.nativeName}). NEVER invent, guess, or speak anything during silence, breathing, clicks, or noise.`;

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
              liveInAcc += content.inputTranscription.text;
              clientWs.send(
                JSON.stringify({
                  type: "inputTranscript",
                  text: content.inputTranscription.text,
                })
              );
            }
            if (content?.outputTranscription?.text) {
              liveOutAcc += content.outputTranscription.text;
              clientWs.send(
                JSON.stringify({
                  type: "outputTranscript",
                  text: content.outputTranscription.text,
                })
              );
            }
            for (const part of content?.modelTurn?.parts ?? []) {
              if (part.inlineData?.data) {
                if (lowBandwidth) {
                  const down = downsample24kTo12kPcm16(Buffer.from(part.inlineData.data, "base64"));
                  if (down.length > 0) {
                    clientWs.send(
                      JSON.stringify({
                        type: "audio",
                        sampleRate: 12000,
                        data: down.toString("base64"),
                      })
                    );
                  }
                } else {
                  clientWs.send(
                    JSON.stringify({
                      type: "audio",
                      sampleRate: 24000,
                      data: part.inlineData.data,
                    })
                  );
                }
              }
            }
            if (content?.interrupted) {
              liveInAcc = "";
              liveOutAcc = "";
              clientWs.send(JSON.stringify({ type: "interrupted" }));
            }
            if (content?.turnComplete) {
              const trimmedIn = liveInAcc.trim();
              const trimmedOut = liveOutAcc.trim();
              liveInAcc = "";
              liveOutAcc = "";
              if (!trimmedIn && !trimmedOut) {
                clientWs.send(JSON.stringify({ type: "turnDiscarded" }));
              } else if (isNoiseOrHallucination(trimmedIn || trimmedOut, trimmedOut || trimmedIn)) {
                clientWs.send(JSON.stringify({ type: "turnDiscarded" }));
              } else {
                clientWs.send(JSON.stringify({ type: "turnComplete" }));
              }
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

    const useTranslateInitially =
      engineParam === "translate35"
        ? true
        : engineParam === "live38"
        ? false
        : !isBidirectional && targetLang.liveTranslateSupported;
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
            const digitalZeroB64 = Buffer.alloc(2048).toString("base64");
            for (const ch of current.liveTranslators.values()) {
              if (ch.closed) continue;
              if (ch.mode !== "vad_stream") {
                if (ch.session) {
                  for (let i = 0; i < 8; i++) {
                    ch.session.sendRealtimeInput({
                      audio: { data: digitalZeroB64, mimeType: "audio/pcm;rate=16000" },
                    });
                  }
                }
                continue;
              }
              ch.vadConfirmCount = 0;
              if (ch.vadSpeechActive && ch.vadChunks.length > 0) {
                const totalBytes = ch.vadChunks.reduce((acc, b) => acc + b.length, 0);
                const utterance = ch.vadChunks;
                ch.vadChunks = [];
                ch.vadSpeechActive = false;
                if (
                  totalBytes >= 16000 * 2 * 0.18 &&
                  hasValidSpeechEnergy(utterance, ch.vadNoiseFloor, current.whisperBoost)
                ) {
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
