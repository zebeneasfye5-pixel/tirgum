export function downsampleTo16k(
  input: Float32Array,
  inputSampleRate: number
): Float32Array {
  if (inputSampleRate === 16000) return input;
  const ratio = inputSampleRate / 16000;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j++) {
      sum += input[j];
      count++;
    }
    output[i] = count > 0 ? sum / count : 0;
  }
  return output;
}

export function float32ToPcm16Base64(float32: Float32Array): string {
  const pcm16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, float32[i]));
    pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  const bytes = new Uint8Array(pcm16.buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const sub = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...sub);
  }
  return btoa(binary);
}

export function base64Pcm16ToFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const byteLength = binary.length;
  const bytes = new Uint8Array(byteLength);
  for (let i = 0; i < byteLength; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const view = new DataView(bytes.buffer);
  const sampleCount = Math.floor(byteLength / 2);
  const float32 = new Float32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    float32[i] = view.getInt16(i * 2, true) / 32768;
  }
  return float32;
}

export async function createInputAudioContext(): Promise<AudioContext> {
  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ sampleRate: 16000 });
  } catch {
    ctx = new AudioContext();
  }
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
  return ctx;
}

export interface BoostedMicPipeline {
  ctx: AudioContext;
  analyser: AnalyserNode;
  processor: ScriptProcessorNode;
  setBoost: (enabled: boolean) => void;
  cleanup: () => void;
}

/**
 * Studio voice pre-conditioning chain:
 * High-pass filter (85Hz) strips room rumble/wind, followed by high-sensitivity
 * pre-gain + DynamicsCompressor + makeup gain so whispers and distant voices
 * are lifted clearly without clipping loud speech.
 */
export async function createBoostedMicPipeline(
  stream: MediaStream,
  boostEnabled: boolean
): Promise<BoostedMicPipeline> {
  const ctx = await createInputAudioContext();
  const source = ctx.createMediaStreamSource(stream);

  const highpass = ctx.createBiquadFilter();
  highpass.type = "highpass";
  highpass.frequency.value = 85;
  highpass.Q.value = 0.7;

  const preGain = ctx.createGain();
  const compressor = ctx.createDynamicsCompressor();
  compressor.knee.value = 18;
  compressor.attack.value = 0.003;
  compressor.release.value = 0.14;
  const makeupGain = ctx.createGain();

  const applyBoost = (enabled: boolean) => {
    const now = ctx.currentTime;
    preGain.gain.setTargetAtTime(enabled ? 3.8 : 1.35, now, 0.03);
    compressor.threshold.setTargetAtTime(enabled ? -38 : -22, now, 0.03);
    compressor.ratio.setTargetAtTime(enabled ? 8 : 3, now, 0.03);
    makeupGain.gain.setTargetAtTime(enabled ? 1.65 : 1.05, now, 0.03);
  };
  applyBoost(boostEnabled);

  const analyser = ctx.createAnalyser();
  analyser.fftSize = 256;

  // 2048-sample frames halve capture latency vs 4096 while keeping WS smooth
  const processor = ctx.createScriptProcessor(2048, 1, 1);
  const silentGain = ctx.createGain();
  silentGain.gain.value = 0;

  source.connect(highpass);
  highpass.connect(preGain);
  preGain.connect(compressor);
  compressor.connect(makeupGain);
  makeupGain.connect(analyser);
  analyser.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(ctx.destination);

  return {
    ctx,
    analyser,
    processor,
    setBoost: applyBoost,
    cleanup: () => {
      try {
        processor.disconnect();
        processor.onaudioprocess = null;
        analyser.disconnect();
        makeupGain.disconnect();
        compressor.disconnect();
        preGain.disconnect();
        highpass.disconnect();
        source.disconnect();
        silentGain.disconnect();
      } catch {
        // ignore cleanup errors
      }
      void ctx.close().catch(() => undefined);
    },
  };
}

export class GaplessPcmPlayer {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private gainNode: GainNode | null = null;
  private nextStartTime = 0;
  private activeSources = new Set<AudioBufferSourceNode>();
  private muted = false;

  public ensureContext(): { ctx: AudioContext; analyser: AnalyserNode } {
    if (!this.ctx) {
      try {
        this.ctx = new AudioContext({ sampleRate: 24000 });
      } catch {
        this.ctx = new AudioContext();
      }
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = this.muted ? 0 : 1.25;
      this.analyser.connect(this.gainNode);
      this.gainNode.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") {
      void this.ctx.resume();
    }
    return { ctx: this.ctx, analyser: this.analyser! };
  }

  public getAnalyser(): AnalyserNode | null {
    return this.analyser;
  }

  public setMuted(muted: boolean) {
    this.muted = muted;
    if (this.gainNode) {
      this.gainNode.gain.value = muted ? 0 : 1.25;
    }
  }

  public enqueuePcm24kBase64(base64: string) {
    const { ctx, analyser } = this.ensureContext();
    const samples = base64Pcm16ToFloat32(base64);
    if (samples.length === 0) return;

    const buffer = ctx.createBuffer(1, samples.length, 24000);
    buffer.getChannelData(0).set(samples);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyser);

    const now = ctx.currentTime;
    if (this.nextStartTime < now + 0.015) {
      this.nextStartTime = now + 0.025;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += buffer.duration;

    this.activeSources.add(source);
    source.onended = () => {
      this.activeSources.delete(source);
    };
  }

  public async enqueueWavBase64(base64Wav: string) {
    try {
      const { ctx, analyser } = this.ensureContext();
      const binary = atob(base64Wav);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const decoded = await ctx.decodeAudioData(bytes.buffer.slice(0));
      if (!this.ctx || !this.analyser) return;

      const source = ctx.createBufferSource();
      source.buffer = decoded;
      source.connect(analyser);

      const now = ctx.currentTime;
      if (this.nextStartTime < now + 0.015) {
        this.nextStartTime = now + 0.025;
      }

      source.start(this.nextStartTime);
      this.nextStartTime += decoded.duration;

      this.activeSources.add(source);
      source.onended = () => {
        this.activeSources.delete(source);
      };
    } catch {
      if (!this.muted) {
        await playWavBase64(base64Wav);
      }
    }
  }

  public interrupt() {
    for (const src of this.activeSources) {
      try {
        src.stop();
      } catch {
        // ignore
      }
    }
    this.activeSources.clear();
    if (this.ctx) {
      this.nextStartTime = this.ctx.currentTime;
    }
  }

  public close() {
    this.interrupt();
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
      this.analyser = null;
      this.gainNode = null;
    }
  }
}

export async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const commaIdx = result.indexOf(",");
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

export function playWavBase64(base64Wav: string): Promise<void> {
  return new Promise((resolve) => {
    const audio = new Audio(`data:audio/wav;base64,${base64Wav}`);
    audio.onended = () => resolve();
    audio.onerror = () => resolve();
    void audio.play().catch(() => resolve());
  });
}

export function speakBrowserFallback(text: string, langCode: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window) || !text.trim()) {
    return;
  }
  try {
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = langCode;
    utter.rate = 1.02;
    window.speechSynthesis.speak(utter);
  } catch {
    // ignore browser speech errors
  }
}
