import { useEffect, useRef } from "react";

interface WaveformCanvasProps {
  analyser: AnalyserNode | null;
  active: boolean;
  accentColor?: string;
}

export function WaveformCanvas({
  analyser,
  active,
  accentColor = "#1D4ED8",
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    const barCount = 36;
    const dataArray = analyser ? new Uint8Array(analyser.frequencyBinCount) : null;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      const gap = 3;
      const totalGap = gap * (barCount - 1);
      const barWidth = Math.max(2, (width - totalGap) / barCount);

      if (active && analyser && dataArray) {
        analyser.getByteFrequencyData(dataArray);
      }

      for (let i = 0; i < barCount; i++) {
        let energy = 0.08;
        if (active && dataArray && dataArray.length > 0) {
          const binIndex = Math.floor((i / barCount) * (dataArray.length * 0.6));
          energy = Math.max(0.1, dataArray[binIndex] / 255);
        }
        const barHeight = Math.max(3, energy * (height - 6));
        const x = i * (barWidth + gap);
        const y = (height - barHeight) / 2;

        ctx.fillStyle = active ? accentColor : "#CBD5E1";
        ctx.globalAlpha = active ? 0.35 + energy * 0.65 : 0.7;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }

      ctx.restore();
      if (active) {
        rafId = requestAnimationFrame(render);
      }
    };

    render();
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [analyser, active, accentColor]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="h-10 w-full block"
    />
  );
}
