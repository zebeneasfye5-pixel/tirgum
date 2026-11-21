export interface LanguageOption {
  code: string;
  name: string;
  nativeName: string;
  amharicName: string;
  region: 'Africa' | 'Americas' | 'Asia' | 'Europe' | 'Middle East';
  liveTranslateSupported: boolean;
}

export const WORLD_LANGUAGES: LanguageOption[] = [
  { code: 'am', name: 'Amharic', nativeName: 'አማርኛ', amharicName: 'አማርኛ', region: 'Africa', liveTranslateSupported: true },
  { code: 'en', name: 'English', nativeName: 'English', amharicName: 'እንግሊዝኛ', region: 'Americas', liveTranslateSupported: true },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية', amharicName: 'አረብኛ', region: 'Middle East', liveTranslateSupported: true },
  { code: 'es', name: 'Spanish', nativeName: 'Español', amharicName: 'ስፓኒሽ', region: 'Europe', liveTranslateSupported: true },
  { code: 'fr', name: 'French', nativeName: 'Français', amharicName: 'ፈረንሳይኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'zh', name: 'Chinese (Mandarin)', nativeName: '中文 (普通话)', amharicName: 'ቻይንኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'hi', name: 'Hindi', nativeName: 'हिन्दी', amharicName: 'ሂንዲ', region: 'Asia', liveTranslateSupported: true },
  { code: 'de', name: 'German', nativeName: 'Deutsch', amharicName: 'ጀርመንኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'pt', name: 'Portuguese', nativeName: 'Português', amharicName: 'ፖርቱጋልኛ', region: 'Americas', liveTranslateSupported: true },
  { code: 'ru', name: 'Russian', nativeName: 'Русский', amharicName: 'ሩሲያኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', amharicName: 'ጃፓንኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'ko', name: 'Korean', nativeName: '한국어', amharicName: 'ኮሪያኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'sw', name: 'Swahili', nativeName: 'Kiswahili', amharicName: 'ስዋሂሊ', region: 'Africa', liveTranslateSupported: true },
  { code: 'ti', name: 'Tigrinya', nativeName: 'ትግርኛ', amharicName: 'ትግርኛ', region: 'Africa', liveTranslateSupported: false },
  { code: 'om', name: 'Oromo', nativeName: 'Afaan Oromoo', amharicName: 'አፋን ኦሮሞ', region: 'Africa', liveTranslateSupported: false },
  { code: 'so', name: 'Somali', nativeName: 'Soomaali', amharicName: 'ሶማልኛ', region: 'Africa', liveTranslateSupported: false },
  { code: 'it', name: 'Italian', nativeName: 'Italiano', amharicName: 'ጣሊያንኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'tr', name: 'Turkish', nativeName: 'Türkçe', amharicName: 'ቱርክኛ', region: 'Middle East', liveTranslateSupported: true },
  { code: 'nl', name: 'Dutch', nativeName: 'Nederlands', amharicName: 'ደች', region: 'Europe', liveTranslateSupported: true },
  { code: 'pl', name: 'Polish', nativeName: 'Polski', amharicName: 'ፖላንድኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'uk', name: 'Ukrainian', nativeName: 'Українська', amharicName: 'ዩክሬንኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'vi', name: 'Vietnamese', nativeName: 'Tiếng Việt', amharicName: 'ቬትናምኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'th', name: 'Thai', nativeName: 'ไทย', amharicName: 'ታይኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'id', name: 'Indonesian', nativeName: 'Bahasa Indonesia', amharicName: 'ኢንዶኔዥያኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'ms', name: 'Malay', nativeName: 'Bahasa Melayu', amharicName: 'ማላይኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'fa', name: 'Persian', nativeName: 'فارسی', amharicName: 'ፋርሲ', region: 'Middle East', liveTranslateSupported: true },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو', amharicName: 'ኡርዱ', region: 'Asia', liveTranslateSupported: true },
  { code: 'he', name: 'Hebrew', nativeName: 'עברית', amharicName: 'ዕብራይስጥ', region: 'Middle East', liveTranslateSupported: true },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', amharicName: 'ቤንጋሊ', region: 'Asia', liveTranslateSupported: true },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', amharicName: 'ታሚልኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', amharicName: 'ቴሉጉ', region: 'Asia', liveTranslateSupported: true },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', amharicName: 'ማራቲ', region: 'Asia', liveTranslateSupported: true },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', amharicName: 'ፑንጃቢ', region: 'Asia', liveTranslateSupported: true },
  { code: 'tl', name: 'Filipino', nativeName: 'Filipino', amharicName: 'ፊሊፒኖ', region: 'Asia', liveTranslateSupported: true },
  { code: 'el', name: 'Greek', nativeName: 'Ελληνικά', amharicName: 'ግሪክኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'sv', name: 'Swedish', nativeName: 'Svenska', amharicName: 'ስዊድንኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'no', name: 'Norwegian', nativeName: 'Norsk', amharicName: 'ኖርዌይኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'da', name: 'Danish', nativeName: 'Dansk', amharicName: 'ዴንማርክኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'fi', name: 'Finnish', nativeName: 'Suomi', amharicName: 'ፊንላንድኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'cs', name: 'Czech', nativeName: 'Čeština', amharicName: 'ቼክኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'ro', name: 'Romanian', nativeName: 'Română', amharicName: 'ሮማንያኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'hu', name: 'Hungarian', nativeName: 'Magyar', amharicName: 'ሃንጋሪኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'bg', name: 'Bulgarian', nativeName: 'Български', amharicName: 'ቡልጋሪኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'hr', name: 'Croatian', nativeName: 'Hrvatski', amharicName: 'ክሮኤሽያኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'sr', name: 'Serbian', nativeName: 'Српски', amharicName: 'ሰርቢኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'sk', name: 'Slovak', nativeName: 'Slovenčina', amharicName: 'ስሎቫክኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'ha', name: 'Hausa', nativeName: 'Hausa', amharicName: 'ሃውሳ', region: 'Africa', liveTranslateSupported: false },
  { code: 'yo', name: 'Yoruba', nativeName: 'Yorùbá', amharicName: 'ዮሩባ', region: 'Africa', liveTranslateSupported: false },
  { code: 'ig', name: 'Igbo', nativeName: 'Igbo', amharicName: 'ኢግቦ', region: 'Africa', liveTranslateSupported: false },
  { code: 'zu', name: 'Zulu', nativeName: 'isiZulu', amharicName: 'ዙሉ', region: 'Africa', liveTranslateSupported: true },
  { code: 'af', name: 'Afrikaans', nativeName: 'Afrikaans', amharicName: 'አፍሪካንስ', region: 'Africa', liveTranslateSupported: true },
  { code: 'my', name: 'Burmese', nativeName: 'မြန်မာဘာသာ', amharicName: 'በርማኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'km', name: 'Khmer', nativeName: 'ខ្មែរ', amharicName: 'ክመር', region: 'Asia', liveTranslateSupported: true },
  { code: 'ne', name: 'Nepali', nativeName: 'नेपाली', amharicName: 'ኔፓሊ', region: 'Asia', liveTranslateSupported: true },
  { code: 'si', name: 'Sinhala', nativeName: 'සිංහල', amharicName: 'ሲንሃላ', region: 'Asia', liveTranslateSupported: true },
  { code: 'ka', name: 'Georgian', nativeName: 'ქართული', amharicName: 'ጆርጂያኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'hy', name: 'Armenian', nativeName: 'Հայերեն', amharicName: 'አርሜንያኛ', region: 'Europe', liveTranslateSupported: true },
  { code: 'az', name: 'Azerbaijani', nativeName: 'Azərbaycan', amharicName: 'አዘርባጃንኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'kk', name: 'Kazakh', nativeName: 'Қазақша', amharicName: 'ካዛክኛ', region: 'Asia', liveTranslateSupported: true },
  { code: 'uz', name: 'Uzbek', nativeName: "O'zbekcha", amharicName: 'ኡዝቤክኛ', region: 'Asia', liveTranslateSupported: true },
];

export const VOICE_PROFILES = [
  { id: 'Zephyr', label: 'Zephyr', labelAm: 'ዜፊር (ተፈጥሯዊ)', tone: 'Warm & balanced' },
  { id: 'Kore', label: 'Kore', labelAm: 'ኮሬ (ግልፅ)', tone: 'Crisp & articulate' },
  { id: 'Puck', label: 'Puck', labelAm: 'ፓክ (ንቁ)', tone: 'Energetic & bright' },
  { id: 'Charon', label: 'Charon', labelAm: 'ካሮን (ረጋ ያለ)', tone: 'Deep & calm' },
  { id: 'Fenrir', label: 'Fenrir', labelAm: 'ፌንሪር (ጠንካራ)', tone: 'Resonant & bold' },
] as const;

export type VoiceId = (typeof VOICE_PROFILES)[number]['id'];

export function getLanguageByCode(code: string): LanguageOption {
  return WORLD_LANGUAGES.find((l) => l.code === code) ?? WORLD_LANGUAGES[1];
}
