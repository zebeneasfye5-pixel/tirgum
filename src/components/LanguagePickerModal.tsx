import { useMemo, useState } from "react";
import { Search, X, Check } from "lucide-react";
import { WORLD_LANGUAGES, LanguageOption } from "../data/languages";
import { UI_COPY, UiLang } from "../data/i18n";

interface LanguagePickerModalProps {
  open: boolean;
  uiLang: UiLang;
  selectedCode: string;
  allowAuto?: boolean;
  onSelect: (code: string) => void;
  onClose: () => void;
}

const REGIONS = ["All", "Africa", "Europe", "Asia", "Middle East", "Americas"] as const;

export function LanguagePickerModal({
  open,
  uiLang,
  selectedCode,
  allowAuto = false,
  onSelect,
  onClose,
}: LanguagePickerModalProps) {
  const [query, setQuery] = useState("");
  const [region, setRegion] = useState<(typeof REGIONS)[number]>("All");
  const t = UI_COPY[uiLang];

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return WORLD_LANGUAGES.filter((lang) => {
      if (region !== "All" && lang.region !== region) return false;
      if (!q) return true;
      return (
        lang.name.toLowerCase().includes(q) ||
        lang.nativeName.toLowerCase().includes(q) ||
        lang.amharicName.toLowerCase().includes(q) ||
        lang.code.toLowerCase().includes(q)
      );
    });
  }, [query, region]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 p-4 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-label={t.selectLanguageTitle}
    >
      <div className="flex max-h-[84vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-slate-300 bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <h2 className="font-display text-xl font-extrabold text-slate-950">
            {t.selectLanguageTitle}
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label={t.close}
            className="flex h-10 w-10 items-center justify-center rounded-lg text-slate-700 transition-colors hover:bg-slate-100 hover:text-slate-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-3 border-b border-slate-200 px-6 py-4">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t.searchLanguage}
              autoFocus
              className="h-11 w-full rounded-lg border border-slate-300 bg-slate-50 pl-10 pr-4 text-base font-semibold text-slate-950 placeholder:text-slate-500 focus:border-blue-600 focus:bg-white focus:outline-none"
            />
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {REGIONS.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRegion(r)}
                className={`whitespace-nowrap shrink-0 rounded-md px-3.5 py-2 text-sm font-bold transition-colors ${
                  region === r
                    ? "bg-slate-900 text-white"
                    : "bg-slate-100 text-slate-800 hover:bg-slate-200 hover:text-slate-950"
                }`}
              >
                {r === "All" ? t.allRegions : r}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-y-auto p-4 sm:p-6">
          {allowAuto && (
            <button
              type="button"
              onClick={() => {
                onSelect("auto");
                onClose();
              }}
              className={`mb-3 flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition-colors ${
                selectedCode === "auto"
                  ? "border-blue-600 bg-blue-50 text-blue-950"
                  : "border-slate-300 bg-white text-slate-950 hover:border-slate-400 hover:bg-slate-50"
              }`}
            >
              <div>
                <div className="text-base font-extrabold">{t.autoDetect}</div>
                <div className="text-sm font-semibold text-slate-700">
                  {uiLang === "am"
                    ? "የተናጋሪውን ቋንቋ በራሱ ይለያል"
                    : "Automatically identifies the spoken language"}
                </div>
              </div>
              {selectedCode === "auto" && <Check className="h-5 w-5 text-blue-700" />}
            </button>
          )}

          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            {filtered.map((lang: LanguageOption) => {
              const isSelected = lang.code === selectedCode;
              return (
                <button
                  key={lang.code}
                  type="button"
                  onClick={() => {
                    onSelect(lang.code);
                    onClose();
                  }}
                  className={`flex items-center justify-between rounded-lg border px-4 py-3 text-left transition-colors ${
                    isSelected
                      ? "border-blue-600 bg-blue-50"
                      : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50"
                  }`}
                >
                  <div className="min-w-0 pr-2">
                    <div className="truncate text-base font-extrabold text-slate-950">
                      {lang.nativeName}
                    </div>
                    <div className="truncate text-sm font-semibold text-slate-700">
                      {uiLang === "am" ? `${lang.amharicName} · ${lang.name}` : lang.name}
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className="font-mono text-xs font-bold uppercase text-slate-600">
                      {lang.code}
                    </span>
                    {isSelected && <Check className="h-4 w-4 text-blue-700" />}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
