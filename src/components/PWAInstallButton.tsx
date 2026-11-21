import { useState } from "react";
import { Download, Smartphone, WifiOff, X } from "lucide-react";
import { UiLang } from "../data/i18n";
import { useOnlineStatus, usePWAInstall } from "../hooks/usePWAInstall";

interface PWAInstallButtonProps {
  uiLang: UiLang;
}

export function PWAInstallButton({ uiLang }: PWAInstallButtonProps) {
  const { isInstallable, isInstalled, isIOS, install } = usePWAInstall();
  const [showGuide, setShowGuide] = useState(false);

  if (isInstalled) return null;

  if (isInstallable) {
    return (
      <button
        type="button"
        onClick={() => void install()}
        className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-blue-200 bg-blue-50 px-3 text-xs font-semibold text-blue-900 transition-colors hover:bg-blue-100 whitespace-nowrap shrink-0"
      >
        <Download className="h-3.5 w-3.5" />
        <span>{uiLang === "am" ? "ወደ ስልክ ጫን" : "Install App"}</span>
      </button>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setShowGuide(true)}
        className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-semibold text-slate-700 transition-colors hover:bg-slate-50 whitespace-nowrap shrink-0"
      >
        <Smartphone className="h-3.5 w-3.5 text-blue-700" />
        <span>{uiLang === "am" ? "ወደ ስልክ ጫን" : "Install App"}</span>
      </button>

      {showGuide && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 p-4 backdrop-blur-[2px]"
          role="dialog"
          aria-modal="true"
        >
          <div className="w-full max-w-sm rounded-xl border border-slate-200 bg-white p-6 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="font-display text-base font-semibold text-slate-900">
                {uiLang === "am" ? "ወደ ስልክዎ እንደ አፕ ይጫኑ" : "Install Lissan on Your Phone"}
              </h3>
              <button
                type="button"
                onClick={() => setShowGuide(false)}
                aria-label={uiLang === "am" ? "ዝጋ" : "Close"}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {isIOS ? (
              <ol className="mt-3 space-y-2 text-sm leading-relaxed text-slate-600">
                <li>
                  {uiLang === "am"
                    ? "1. በSafari አሳሽ ከታች ወይም ከላይ ያለውን Share (መላኪያ) ምልክት ይጫኑ።"
                    : "1. Tap the Share icon in the Safari toolbar."}
                </li>
                <li>
                  {uiLang === "am"
                    ? "2. ወደ ታች አውርደው «Add to Home Screen» የሚለውን ይምረጡ።"
                    : "2. Scroll down and tap Add to Home Screen."}
                </li>
              </ol>
            ) : (
              <ol className="mt-3 space-y-2 text-sm leading-relaxed text-slate-600">
                <li>
                  {uiLang === "am"
                    ? "1. በChrome ወይም Edge አሳሽ የላይኛውን ⋮ (Menu) ይጫኑ።"
                    : "1. Open the browser menu (⋮) in Chrome or Edge."}
                </li>
                <li>
                  {uiLang === "am"
                    ? "2. «Install app» ወይም «Add to Home screen» የሚለውን ይጫኑ።"
                    : "2. Tap Install app or Add to Home screen."}
                </li>
              </ol>
            )}

            <button
              type="button"
              onClick={() => setShowGuide(false)}
              className="mt-5 h-10 w-full rounded-lg bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800"
            >
              {uiLang === "am" ? "እሺ" : "Got it"}
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export function OfflineIndicator({ uiLang }: { uiLang: UiLang }) {
  const isOnline = useOnlineStatus();

  if (isOnline) return null;

  return (
    <div
      role="status"
      className="fixed bottom-4 left-4 z-50 flex items-center gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3.5 py-2 text-xs font-semibold text-amber-900 shadow-md"
    >
      <WifiOff className="h-3.5 w-3.5" />
      <span>
        {uiLang === "am"
          ? "ኢንተርኔት ተቋርጧል — የቆየው መዝገብ ብቻ ይታያል"
          : "Offline Mode — Saved transcripts remain available"}
      </span>
    </div>
  );
}
