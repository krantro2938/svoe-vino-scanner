'use client';

import {
  ArrowLeft,
  BadgeCheck,
  BookOpen,
  Camera,
  ChevronRight,
  Clock3,
  Grape,
  Heart,
  ImagePlus,
  LoaderCircle,
  MapPin,
  RefreshCw,
  ScanLine,
  ShieldCheck,
  Sparkles,
  ThermometerSun,
  UtensilsCrossed,
  Wine,
  X,
} from 'lucide-react';
import { ChangeEvent, useEffect, useRef, useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const CONFIGURED_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '');
const SAMPLE_IMAGE = '/images/massandra_muskatel_belyy_belye_sorta_vinograda_beloe_sladkoe_16_f7e0314033.webp';

type WineRecord = {
  slug: string;
  name: string;
  winery: string;
  category?: string;
  color?: string;
  region?: string;
  grapes?: string[] | string;
  description?: string;
  image_url?: string;
  serving_temperature?: string;
  alcohol?: string;
  rating?: number;
};

type RankedWine = { rank: number; probability: number; wine: WineRecord };
type AnalogWine = { wine: WineRecord; reasons: string[] };

type SearchResponse = {
  status?: 'found' | 'uncertain' | 'not_found' | 'low_quality';
  wine?: WineRecord;
  top5?: RankedWine[];
  analogs?: AnalogWine[];
  confidence_top5?: number;
  slug?: string;
  confidence?: number;
  top1_top2_margin?: number;
  latency_ms?: number;
  message?: string;
  error?: { message?: string };
  quality_hint?: string;
  alternatives?: WineRecord[];
};

type SearchMeta = Pick<SearchResponse, 'confidence' | 'confidence_top5' | 'top1_top2_margin' | 'latency_ms' | 'status'>;

let activeApiBase: string | null = null;

function withApiImages(wine: WineRecord): WineRecord {
  if (wine.image_url?.startsWith('/v1/') && activeApiBase) return { ...wine, image_url: `${activeApiBase}${wine.image_url}` };
  return wine;
}

async function apiGet<T>(path: string): Promise<T> {
  const bases = activeApiBase ? [activeApiBase, ...apiBaseCandidates()] : apiBaseCandidates();
  for (const baseUrl of Array.from(new Set(bases))) {
    try {
      const response = await fetch(`${baseUrl}${path}`);
      if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) continue;
      activeApiBase = baseUrl;
      return (await response.json()) as T;
    } catch {
      continue;
    }
  }
  throw new Error('Каталог сейчас недоступен.');
}

function apiBaseCandidates() {
  const hostname = typeof window === 'undefined' ? '127.0.0.1' : window.location.hostname;
  const pagePort = typeof window === 'undefined' ? '' : window.location.port;
  const inferredPorts = pagePort === '3003' ? ['8081', '8080'] : ['8080', '8081'];
  return Array.from(new Set([
    CONFIGURED_API_BASE_URL,
    ...inferredPorts.map((port) => `http://${hostname}:${port}`),
  ].filter((value): value is string => Boolean(value))));
}

// Phone photos are 3-5 MB; recognition never uses more than ~1300 px, so send a
// 2560 px JPEG (~0.8 MB) instead; 2048 px flipped one borderline real photo. EXIF orientation is applied by createImageBitmap.
async function downscaleForUpload(file: File, maxSide = 2560): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && file.size < 1_500_000) return file;
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d')?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    return blob ? new File([blob], 'label.jpg', { type: 'image/jpeg' }) : file;
  } catch {
    return file;
  }
}

async function searchWine(original: File): Promise<SearchResponse> {
  const file = await downscaleForUpload(original);
  let connectionFailure = false;

  for (const baseUrl of apiBaseCandidates()) {
    const form = new FormData();
    form.append('image', file);

    let response: Response;
    try {
      response = await fetch(`${baseUrl}/v1/search`, { method: 'POST', body: form });
    } catch {
      connectionFailure = true;
      continue;
    }

    const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
    if (!contentType.includes('application/json')) {
      connectionFailure = true;
      continue;
    }

    let payload: SearchResponse;
    try {
      payload = (await response.json()) as SearchResponse;
    } catch {
      connectionFailure = true;
      continue;
    }

    if (!response.ok) {
      throw new Error(payload.error?.message || payload.message || 'Сервис не смог обработать фотографию.');
    }
    activeApiBase = baseUrl;
    return {
      ...payload,
      wine: payload.wine ? withApiImages(payload.wine) : undefined,
      top5: payload.top5?.map((item) => ({ ...item, wine: withApiImages(item.wine) })),
      analogs: payload.analogs?.map((item) => ({ ...item, wine: withApiImages(item.wine) })),
    };
  }

  throw new Error(
    connectionFailure
      ? 'Сервис распознавания сейчас недоступен. Попробуйте ещё раз через несколько секунд.'
      : 'Не удалось обработать фотографию.',
  );
}

type PairingResponse = {
  fit?: 'great' | 'good' | 'better_alternative';
  verdict?: 'excellent' | 'good' | 'acceptable' | 'not_ideal';
  title?: string;
  explanation?: string;
  serving_tip?: string;
  serving_temperature?: string;
  pair_with?: string[];
  glass?: string;
  alternatives?: { slug: string; name: string; winery?: string; category?: string; reason: string }[];
};

async function requestPairing(slug: string, dish: string, occasion: string, preference: string): Promise<PairingResponse> {
  for (const baseUrl of apiBaseCandidates()) {
    try {
      const response = await fetch(`${baseUrl}/v1/sommelier/pairing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug, dish, occasion: occasion || null, preference: preference || null }),
      });
      const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
      if (!contentType.includes('application/json')) continue;
      const payload = (await response.json()) as PairingResponse & { error?: { message?: string } };
      if (!response.ok) throw new Error(payload.error?.message || 'Не удалось получить рекомендацию.');
      return payload;
    } catch {
      continue;
    }
  }
  throw new Error('Сервис рекомендаций сейчас недоступен.');
}

const sampleWine: WineRecord = {
  slug: 'massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16',
  name: 'Мускатель белый',
  winery: 'Массандра',
  category: 'Белое сладкое',
  color: 'Золотисто-янтарный',
  region: 'Крым',
  grapes: ['Белые сорта винограда'],
  description:
    'Мягкое десертное вино с цветочно-медовыми нотами, ароматом спелых фруктов и долгим гармоничным послевкусием.',
  image_url: SAMPLE_IMAGE,
  serving_temperature: '10–12 °C',
  alcohol: '16%',
  rating: 5,
};

const dishOptions = [
  { id: 'fish', label: 'Рыба', icon: '🐟' },
  { id: 'meat', label: 'Мясо', icon: '🥩' },
  { id: 'cheese', label: 'Сыр', icon: '🧀' },
  { id: 'dessert', label: 'Десерт', icon: '🍰' },
  { id: 'vegetables', label: 'Овощи', icon: '🥬' },
] as const;

function normalizeWine(response: SearchResponse): WineRecord | null {
  if (response.wine?.slug) return response.wine;
  if (response.slug) {
    return {
      ...sampleWine,
      slug: response.slug,
      name: response.slug
        .split('-')
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' '),
      winery: 'Каталог «Своё Вино»',
      image_url: undefined,
    };
  }
  return null;
}

function grapeList(grapes: WineRecord['grapes']) {
  if (Array.isArray(grapes)) return grapes.join(', ');
  return grapes || 'Не указан';
}

function localPairing(wine: WineRecord, dish: string): PairingResponse {
  const category = `${wine.category ?? ''} ${wine.color ?? ''}`.toLowerCase();
  const preferred = category.includes('красн')
    ? ['meat', 'cheese']
    : category.includes('слад')
      ? ['dessert', 'cheese']
      : category.includes('бел')
        ? ['fish', 'vegetables', 'cheese']
        : ['vegetables', 'cheese'];
  const isMatch = preferred.includes(dish);
  const dishLabel = dishOptions.find((item) => item.id === dish)?.label.toLowerCase() ?? 'блюдом';

  return {
    fit: isMatch ? 'great' : 'good',
    title: isMatch ? 'Отличное сочетание' : 'Сочетание возможно',
    explanation: isMatch
      ? `${wine.name} поддержит вкус блюда «${dishLabel}» и не потеряется рядом с ним.`
      : `Для блюда «${dishLabel}» это вино лучше подать с более деликатным соусом. Баланс блюда важнее насыщенности.`,
    serving_tip: `Охладите до ${wine.serving_temperature ?? '10–12 °C'} и откройте за 10 минут до подачи.`,
  };
}

export default function Home() {
  const [phase, setPhase] = useState<'scan' | 'loading' | 'result' | 'not_found' | 'error'>('scan');
  const [candidates, setCandidates] = useState<RankedWine[]>([]);
  const [analogs, setAnalogs] = useState<AnalogWine[]>([]);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [wine, setWine] = useState<WineRecord | null>(null);
  const [searchMeta, setSearchMeta] = useState<SearchMeta>({});
  const [errorMessage, setErrorMessage] = useState('');
  const [dish, setDish] = useState('');
  const [occasion, setOccasion] = useState('');
  const [taste, setTaste] = useState('');
  const [pairing, setPairing] = useState<PairingResponse | null>(null);
  const [pairingLoading, setPairingLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [recent, setRecent] = useState<WineRecord[]>([]);
  const uploadRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('svoe-wine-history') ?? '[]') as WineRecord[];
      queueMicrotask(() => setRecent(stored));
    } catch {
      queueMicrotask(() => setRecent([]));
    }
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function recognize(file: File) {
    if (!file.type.startsWith('image/') || file.size > 20 * 1024 * 1024) {
      setErrorMessage('Выберите изображение JPEG, PNG или WebP размером до 20 МБ.');
      setPhase('error');
      return;
    }

    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
    const localPreview = URL.createObjectURL(file);
    setPreviewUrl(localPreview);
    setPhase('loading');
    setPairing(null);
    setDish('');
    setSaved(false);

    try {
      const payload = await searchWine(file);

      const matchedWine = normalizeWine(payload);
      setCandidates(payload.top5 ?? []);
      setAnalogs(payload.analogs ?? []);
      setSearchMeta({
        confidence: payload.confidence,
        confidence_top5: payload.confidence_top5,
        top1_top2_margin: payload.top1_top2_margin,
        latency_ms: payload.latency_ms,
        status: payload.status,
      });
      if (!matchedWine || payload.status === 'not_found') {
        setPhase('not_found');
        return;
      }
      if (payload.status === 'low_quality') {
        setErrorMessage(payload.quality_hint || 'Фото удалось прочитать лишь частично. Снимите этикетку ближе и без блика.');
        setPhase('error');
        return;
      }

      setWine(matchedWine);
      setPhase('result');
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Не удалось связаться с сервисом распознавания.');
      setPhase('error');
    }
  }

  function openWine(next: WineRecord, status: SearchMeta['status'] = 'found') {
    setWine(next);
    setSearchMeta((meta) => ({ ...meta, status }));
    setPairing(null);
    setDish('');
    setSaved(false);
    setPhase('result');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async function openSlug(slug: string) {
    try {
      openWine(withApiImages(await apiGet<WineRecord>(`/v1/wines/${encodeURIComponent(slug)}`)), 'found');
      setCandidates([]);
    } catch {
      /* the list stays visible; nothing else to do */
    }
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void recognize(file);
    event.target.value = '';
  }

  function showSample() {
    setPreviewUrl(null);
    setWine(sampleWine);
    setSearchMeta({ confidence: 0.75, top1_top2_margin: 0.021, latency_ms: 2071, status: 'found' });
    setPairing(null);
    setDish('');
    setSaved(false);
    setPhase('result');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function reset() {
    setPhase('scan');
    setWine(null);
    setSearchMeta({});
    setPreviewUrl(null);
    setErrorMessage('');
    setPairing(null);
    setDish('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function saveWine() {
    if (!wine) return;
    const next = [wine, ...recent.filter((item) => item.slug !== wine.slug)].slice(0, 5);
    setRecent(next);
    setSaved(true);
    localStorage.setItem('svoe-wine-history', JSON.stringify(next));
  }

  async function getPairing() {
    if (!wine || !dish) return;
    setPairingLoading(true);
    try {
      const result = await requestPairing(wine.slug, dish, occasion, taste);
      const titles = {
        excellent: 'Отличное сочетание',
        good: 'Хорошее сочетание',
        acceptable: 'Сочетание возможно',
        not_ideal: 'Лучше выбрать альтернативу',
      };
      setPairing({
        ...result,
        title: result.title || (result.verdict ? titles[result.verdict] : 'Рекомендация готова'),
        serving_tip:
          result.serving_tip ||
          [result.serving_temperature ? `Подавайте при ${result.serving_temperature}.` : '', result.pair_with?.length ? `Также подойдёт: ${result.pair_with.join(', ')}.` : '']
            .filter(Boolean)
            .join(' '),
      });
    } catch {
      setPairing(localPairing(wine, dish));
    } finally {
      setPairingLoading(false);
    }
  }

  return (
    <main className="min-h-dvh overflow-hidden bg-background text-foreground">
      <SiteHeader compact={phase !== 'scan'} onBack={phase !== 'scan' ? reset : undefined} />

      {phase === 'scan' && (
        <>
          <section className="relative mx-auto grid w-full max-w-6xl grid-cols-1 gap-10 px-5 pb-16 pt-4 sm:px-8 lg:grid-cols-[1.06fr_0.94fr] lg:items-center lg:pb-24 lg:pt-14">
            <div className="pointer-events-none absolute -left-52 top-20 size-[32rem] rounded-full bg-[radial-gradient(circle,var(--wine-glow),transparent_68%)]" />
            <div className="relative z-10 max-w-xl">
              <Badge className="mb-5 bg-accent text-accent-foreground">
                <Sparkles /> Узнать вино за несколько секунд
              </Badge>
              <h1 className="font-heading text-[clamp(2.65rem,8vw,5.8rem)] font-semibold leading-[0.9] tracking-[-0.055em] text-balance">
                Наведите камеру. Найдите своё.
              </h1>
              <p className="mt-6 max-w-lg text-base leading-7 text-muted-foreground sm:text-lg">
                Сфотографируйте этикетку российского вина — мы найдём точную карточку, расскажем о вкусе и подскажем сочетание с вашим блюдом.
              </p>
              <div className="mt-7 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
                <span className="flex items-center gap-2"><span className="size-1.5 rounded-full bg-primary" /> Одна точная карточка</span>
                <span className="flex items-center gap-2"><span className="size-1.5 rounded-full bg-primary" /> Без лишнего выбора</span>
              </div>
            </div>

            <ScannerCard cameraRef={cameraRef} uploadRef={uploadRef} onFile={onFile} onSample={showSample} />
          </section>

          <section className="border-y border-primary/10 bg-card/55">
            <div className="mx-auto grid max-w-6xl grid-cols-3 divide-x divide-primary/10 px-5 sm:px-8">
              <Metric value="2 182" label="вина в каталоге" />
              <Metric value="≈ 1,5 с" label="от фото до карточки" />
              <Metric value="Top-1" label="одна карточка, без выбора" />
            </div>
          </section>

          <section className="mx-auto grid w-full max-w-6xl grid-cols-1 gap-8 px-5 py-16 sm:px-8 lg:grid-cols-[0.8fr_1.2fr] lg:items-center lg:py-24">
            <div className="relative mx-auto grid h-[410px] w-full max-w-sm place-items-center overflow-hidden rounded-[2rem] bg-[#f9f1f1]">
              <div className="absolute inset-x-10 top-8 h-40 rounded-full bg-white/50 blur-3xl" />
              <img src={SAMPLE_IMAGE} alt="Бутылка вина Мускатель белый Массандра" className="relative h-[360px] w-auto object-contain drop-shadow-[0_24px_28px_rgb(44_42_40/22%)]" />
              <Badge className="absolute bottom-5 left-5 bg-card text-foreground shadow-sm"><BadgeCheck className="text-emerald-700" /> Карточка найдена</Badge>
            </div>
            <div className="lg:pl-10">
              <span className="text-xs font-bold uppercase tracking-[0.18em] text-primary">После сканирования</span>
              <h2 className="mt-4 max-w-xl font-heading text-4xl leading-tight tracking-tight sm:text-5xl">Не просто название — понятный следующий шаг</h2>
              <div className="mt-8 grid gap-5 sm:grid-cols-2">
                <Feature icon={<BookOpen />} title="Полная карточка" copy="Регион, сорт, описание и точные данные из каталога." />
                <Feature icon={<UtensilsCrossed />} title="Сомелье для блюда" copy="Проверит сочетание и предложит подходящие альтернативы." />
                <Feature icon={<ShieldCheck />} title="Честная уверенность" copy="Сервис отличает точное совпадение от неизвестной этикетки." />
                <Feature icon={<Clock3 />} title="Всегда под рукой" copy="Сохранённые находки остаются на этом устройстве." />
              </div>
              <Button size="lg" className="mt-9 h-12 rounded-xl px-5" onClick={showSample}>Посмотреть пример <ChevronRight /></Button>
            </div>
          </section>

          {recent.length > 0 && <RecentScans wines={recent} onOpen={(item) => { setWine(item); setSearchMeta({}); setPhase('result'); window.scrollTo(0, 0); }} />}
        </>
      )}

      {phase === 'loading' && <LoadingState previewUrl={previewUrl} />}

      {phase === 'not_found' && (
        <NotFoundState
          previewUrl={previewUrl}
          candidates={candidates}
          analogs={analogs}
          onOpen={(item) => openWine(item, 'uncertain')}
          onRetry={reset}
        />
      )}

      {phase === 'error' && (
        <ErrorState
          previewUrl={previewUrl}
          message={errorMessage}
          alternatives={candidates.map((item) => item.wine)}
          onRetry={reset}
          onSample={showSample}
        />
      )}

      {phase === 'result' && wine && (
        <WineResult
          wine={wine}
          previewUrl={previewUrl}
          meta={searchMeta}
          saved={saved}
          onSave={saveWine}
          onReset={reset}
          dish={dish}
          setDish={(value) => { setDish(value); setPairing(null); }}
          pairing={pairing}
          pairingLoading={pairingLoading}
          onPairing={() => void getPairing()}
          candidates={candidates}
          onOpenWine={(item) => openWine(item, 'found')}
          onOpenSlug={(slug) => void openSlug(slug)}
          occasion={occasion}
          setOccasion={setOccasion}
          taste={taste}
          setTaste={setTaste}
        />
      )}

      <footer className="border-t border-primary/10 px-5 py-8 text-center text-xs leading-5 text-muted-foreground">
        <p>© 2026 Своё Вино · Демонстрационный модуль сканера</p>
        <p className="mt-2 font-semibold text-foreground/65">Чрезмерное употребление алкоголя вредит вашему здоровью</p>
      </footer>
    </main>
  );
}

function SiteHeader({ compact, onBack }: { compact?: boolean; onBack?: () => void }) {
  return (
    <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-5 sm:px-8">
      <div className="flex items-center gap-3">
        {onBack && <Button variant="ghost" size="icon" className="-ml-2 rounded-full" onClick={onBack} aria-label="Вернуться к сканеру"><ArrowLeft /></Button>}
        <button className="flex items-center gap-3 text-left" onClick={onBack} aria-label="Своё Вино — сканер">
          <span className="grid size-10 place-items-center rounded-full bg-primary text-sm font-bold tracking-tight text-primary-foreground">СВ</span>
          <span>
            <span className="block font-heading text-base font-semibold leading-none">Своё Вино</span>
            <span className="mt-1 block text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Сканер этикеток</span>
          </span>
        </button>
      </div>
      {!compact && <Badge variant="outline" className="hidden border-primary/20 bg-white/70 text-primary sm:flex"><ShieldCheck /> Российские вина</Badge>}
      {compact && <Badge variant="outline" className="border-primary/20 bg-white/70 text-primary"><ScanLine /> Результат</Badge>}
    </header>
  );
}

function ScannerCard({ cameraRef, uploadRef, onFile, onSample }: {
  cameraRef: React.RefObject<HTMLInputElement | null>;
  uploadRef: React.RefObject<HTMLInputElement | null>;
  onFile: (event: ChangeEvent<HTMLInputElement>) => void;
  onSample: () => void;
}) {
  return (
    <Card id="scanner" className="relative z-10 gap-0 rounded-[2rem] border-0 bg-card py-0 shadow-[0_28px_90px_rgb(114_49_53/14%)] ring-1 ring-primary/10">
      <CardContent className="p-4 sm:p-5">
        <div className="relative min-h-[448px] overflow-hidden rounded-[1.55rem] bg-[linear-gradient(155deg,var(--scanner-dark),var(--scanner-mid))] p-5 text-white sm:p-7">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-white/60">Новая фотография</span>
            <span className="flex items-center gap-2 text-xs text-white/70"><span className="size-2 rounded-full bg-emerald-400" /> Готов</span>
          </div>
          <div className="mx-auto mt-8 grid w-full max-w-[290px] place-items-center rounded-[1.6rem] border border-white/15 bg-white/8 px-8 py-8 text-center backdrop-blur-sm">
            <div className="grid size-16 place-items-center rounded-full bg-white text-primary shadow-xl"><Camera className="size-7" strokeWidth={1.8} /></div>
            <h2 className="mt-5 font-heading text-2xl font-semibold">Этикетка целиком</h2>
            <p className="mt-2 text-sm leading-6 text-white/65">Держите бутылку прямо и постарайтесь избежать бликов</p>
          </div>
          <div className="mt-8 grid gap-3 sm:grid-cols-2">
            <Button size="lg" className="h-13 rounded-xl bg-white px-5 text-primary hover:bg-white/90" onClick={() => cameraRef.current?.click()}><Camera /> Сканировать</Button>
            <Button size="lg" variant="outline" className="h-13 rounded-xl border-white/20 bg-white/8 px-5 text-white hover:bg-white/15 hover:text-white" onClick={() => uploadRef.current?.click()}><ImagePlus /> Загрузить фото</Button>
          </div>
          <button className="mx-auto mt-4 block text-xs text-white/55 underline decoration-white/25 underline-offset-4 transition hover:text-white" onClick={onSample}>или открыть готовый пример</button>
          <input ref={cameraRef} className="sr-only" type="file" accept="image/*" capture="environment" onChange={onFile} aria-label="Сфотографировать этикетку" />
          <input ref={uploadRef} className="sr-only" type="file" accept="image/*" onChange={onFile} aria-label="Загрузить фотографию этикетки" />
        </div>
      </CardContent>
    </Card>
  );
}

function LoadingState({ previewUrl }: { previewUrl: string | null }) {
  return (
    <section className="mx-auto min-h-[calc(100dvh-190px)] w-full max-w-2xl px-5 py-8 sm:px-8 sm:py-16">
      <Card className="rounded-[2rem] border-0 py-0 shadow-[0_24px_80px_rgb(114_49_53/12%)] ring-1 ring-primary/10">
        <CardContent className="p-5 sm:p-7">
          <div className="relative aspect-[4/5] max-h-[540px] overflow-hidden rounded-[1.5rem] bg-muted">
            {previewUrl && <img src={previewUrl} alt="Загруженная этикетка" className="size-full object-cover" />}
            <div className="absolute inset-0 grid place-items-center bg-primary/42 backdrop-blur-[2px]">
              <div className="rounded-2xl bg-card/95 px-7 py-6 text-center text-foreground shadow-2xl">
                <LoaderCircle className="mx-auto size-7 animate-spin text-primary" />
                <h1 className="mt-4 font-heading text-2xl">Изучаем этикетку</h1>
                <p className="mt-2 text-sm text-muted-foreground">Сверяем форму, детали и надписи с каталогом</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
}

function ErrorState({ previewUrl, message, alternatives, onRetry, onSample }: {
  previewUrl: string | null;
  message: string;
  alternatives: WineRecord[];
  onRetry: () => void;
  onSample: () => void;
}) {
  return (
    <section className="mx-auto min-h-[calc(100dvh-190px)] w-full max-w-3xl px-5 py-10 sm:px-8 sm:py-20">
      <Card className="rounded-[2rem] border-0 py-0 shadow-[0_24px_80px_rgb(114_49_53/10%)] ring-1 ring-primary/10">
        <CardContent className="grid gap-7 p-5 sm:grid-cols-[180px_1fr] sm:p-8">
          <div className="aspect-square overflow-hidden rounded-2xl bg-muted">
            {previewUrl ? <img src={previewUrl} alt="Загруженная фотография" className="size-full object-cover" /> : <div className="grid size-full place-items-center"><X className="size-9 text-muted-foreground" /></div>}
          </div>
          <div className="self-center">
            <Badge variant="outline" className="border-amber-600/20 bg-amber-50 text-amber-800">Нужна помощь</Badge>
            <h1 className="mt-4 font-heading text-3xl leading-tight">Не удалось уверенно найти вино</h1>
            <p className="mt-3 leading-7 text-muted-foreground">{message}</p>
            <p className="mt-3 text-sm text-muted-foreground">Попробуйте снять переднюю этикетку ближе, прямо и без яркого блика.</p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Button className="h-11 rounded-xl" onClick={onRetry}><RefreshCw /> Попробовать снова</Button>
              <Button variant="outline" className="h-11 rounded-xl" onClick={onSample}>Открыть пример</Button>
            </div>
          </div>
        </CardContent>
      </Card>
      {alternatives.length > 0 && <Alternatives wines={alternatives} />}
    </section>
  );
}

const occasionOptions = [
  { id: 'dinner', label: 'Ужин' },
  { id: 'party', label: 'Компания' },
  { id: 'date', label: 'Свидание' },
  { id: 'gift', label: 'Подарок' },
] as const;

const tasteOptions = [
  { id: 'dry', label: 'Сухое' },
  { id: 'semi', label: 'Полусухое / полусладкое' },
  { id: 'sweet', label: 'Сладкое' },
  { id: '', label: 'Не важно' },
] as const;

function WineResult({ wine, previewUrl, meta, saved, onSave, onReset, dish, setDish, pairing, pairingLoading, onPairing, candidates, onOpenWine, onOpenSlug, occasion, setOccasion, taste, setTaste }: {
  wine: WineRecord;
  previewUrl: string | null;
  meta: SearchMeta;
  saved: boolean;
  onSave: () => void;
  onReset: () => void;
  dish: string;
  setDish: (dish: string) => void;
  pairing: PairingResponse | null;
  pairingLoading: boolean;
  onPairing: () => void;
  candidates: RankedWine[];
  onOpenWine: (wine: WineRecord) => void;
  onOpenSlug: (slug: string) => void;
  occasion: string;
  setOccasion: (value: string) => void;
  taste: string;
  setTaste: (value: string) => void;
}) {
  const image = wine.image_url || previewUrl || SAMPLE_IMAGE;
  const isUncertain = meta.status === 'uncertain';
  const others = candidates.filter((item) => item.wine.slug !== wine.slug);
  return (
    <div className="mx-auto w-full max-w-6xl px-5 pb-16 pt-3 sm:px-8 sm:pt-8">
      <section className="grid grid-cols-1 overflow-hidden rounded-[2rem] bg-card shadow-[0_24px_80px_rgb(114_49_53/10%)] ring-1 ring-primary/10 lg:grid-cols-[0.88fr_1.12fr]">
        <div className="relative grid min-h-[430px] place-items-center overflow-hidden bg-[#f9f1f1] p-8 sm:min-h-[560px]">
          <div className="absolute inset-x-16 top-10 h-56 rounded-full bg-white/60 blur-3xl" />
          <img src={image} alt={`${wine.name}, ${wine.winery}`} className={`relative max-h-[490px] max-w-full object-contain ${wine.image_url ? 'drop-shadow-[0_28px_32px_rgb(44_42_40/22%)]' : 'rounded-2xl'}`} />
          <Badge className={`absolute left-5 top-5 bg-card shadow-sm ${isUncertain ? 'text-amber-800' : 'text-emerald-800'}`}>
            {isUncertain ? <ShieldCheck /> : <BadgeCheck />} {isUncertain ? 'Возможное совпадение' : 'Совпадение найдено'}
          </Badge>
          {previewUrl && wine.image_url && <img src={previewUrl} alt="Ваше фото" className="absolute bottom-5 left-5 size-16 rounded-xl object-cover shadow-lg ring-2 ring-white" />}
          {meta.confidence !== undefined && <span className="absolute bottom-5 right-5 rounded-full bg-black/60 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur">Уверенность {Math.round(meta.confidence * 100)}%</span>}
        </div>

        <div className="p-6 sm:p-10 lg:p-12">
          <div className="flex items-start justify-between gap-5">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.17em] text-primary">{wine.winery}</p>
              <h1 className="mt-3 font-heading text-4xl leading-[1.05] tracking-tight sm:text-5xl">{wine.name}</h1>
              <p className="mt-3 text-base text-muted-foreground">{wine.category || 'Российское вино'}{wine.color ? ` · ${wine.color}` : ''}</p>
            </div>
            <Button variant="outline" size="icon" className="size-11 rounded-full" onClick={onSave} aria-label="Сохранить вино"><Heart className={saved ? 'fill-primary text-primary' : ''} /></Button>
          </div>

          {isUncertain && (
            <Alert className="mt-6 border-amber-700/20 bg-amber-50/70 px-4 py-3 text-amber-950">
              <ShieldCheck />
              <AlertTitle>Проверьте название и производителя</AlertTitle>
              <AlertDescription className="text-amber-900/75">
                Этикетка похожа на эту карточку, но у серии есть близкие варианты. Если это не то вино — выберите ниже.
              </AlertDescription>
            </Alert>
          )}

          <div className="mt-8 grid grid-cols-3 gap-px overflow-hidden rounded-2xl bg-border ring-1 ring-border">
            <WineFact icon={<MapPin />} label="Регион" value={wine.region || 'Не указан'} />
            <WineFact icon={<Grape />} label="Сорт" value={grapeList(wine.grapes)} />
            <WineFact icon={<ThermometerSun />} label="Подача" value={pairing?.serving_temperature || wine.serving_temperature || servingHint(wine)} />
          </div>

          <p className="mt-8 text-[15px] leading-7 text-muted-foreground">{wine.description || 'Описание вкуса и аромата уточняется в каталоге.'}</p>

          <div className="mt-8 flex flex-wrap gap-3">
            {meta.latency_ms !== undefined && <Badge variant="outline" className="h-7 px-3"><Clock3 /> {(meta.latency_ms / 1000).toFixed(1)} с</Badge>}
            {meta.top1_top2_margin !== undefined && <Badge variant="outline" className="h-7 px-3">Отрыв от 2-го места {(meta.top1_top2_margin * 100).toFixed(0)} п.п.</Badge>}
          </div>

          <div className="mt-9 grid gap-3 sm:grid-cols-2">
            <Button size="lg" className="h-12 rounded-xl" onClick={() => document.getElementById('sommelier')?.scrollIntoView({ behavior: 'smooth' })}><UtensilsCrossed /> Подойдёт к блюду?</Button>
            <Button size="lg" variant="outline" className="h-12 rounded-xl" onClick={onReset}><ScanLine /> Сканировать ещё</Button>
          </div>
          {saved && <p className="mt-3 text-center text-xs font-semibold text-emerald-700">Сохранено на этом устройстве</p>}
        </div>
      </section>

      {isUncertain && others.length > 0 && (
        <section className="mt-8">
          <h2 className="font-heading text-2xl">Это одно из них?</h2>
          <p className="mt-1 text-sm text-muted-foreground">Близкие варианты той же серии — вероятность по оценке сканера.</p>
          <WineStrip items={others.map((item) => ({ wine: item.wine, note: `${Math.round(item.probability * 100)}%` }))} onOpen={onOpenWine} />
        </section>
      )}

      <section id="sommelier" className="mt-10 scroll-mt-6 overflow-hidden rounded-[2rem] bg-[linear-gradient(145deg,#5e2529,#8f3d42)] p-6 text-white sm:p-10 lg:p-12">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[0.72fr_1.28fr] lg:gap-14">
          <div>
            <Badge className="bg-white/10 text-white ring-1 ring-white/15"><Sparkles /> Цифровой сомелье</Badge>
            <h2 className="mt-5 font-heading text-3xl leading-tight sm:text-4xl">Подойдёт ли это вино?</h2>
            <p className="mt-3 text-sm leading-6 text-white/65">Три коротких вопроса — и сомелье оценит сочетание, подскажет подачу и предложит вино из каталога, если найдётся вариант точнее.</p>
          </div>
          <div className="space-y-5">
            <SommelierStep index={1} title="Какой повод?">
              {occasionOptions.map((option) => <Chip key={option.id} active={occasion === option.id} onClick={() => setOccasion(occasion === option.id ? '' : option.id)}>{option.label}</Chip>)}
            </SommelierStep>
            <SommelierStep index={2} title="Что на столе?">
              {dishOptions.map((option) => <Chip key={option.id} active={dish === option.id} onClick={() => setDish(option.id)}><span aria-hidden="true">{option.icon}</span> {option.label}</Chip>)}
            </SommelierStep>
            <SommelierStep index={3} title="Какой вкус вам ближе?">
              {tasteOptions.map((option) => <Chip key={option.label} active={taste === option.id} onClick={() => setTaste(option.id)}>{option.label}</Chip>)}
            </SommelierStep>
            <Button size="lg" disabled={!dish || pairingLoading} onClick={onPairing} className="h-12 w-full rounded-xl bg-[#fefdfa] text-[#8f3d42] hover:bg-white">
              {pairingLoading ? <><LoaderCircle className="animate-spin" /> Подбираем сочетание</> : <><UtensilsCrossed /> {dish ? 'Спросить сомелье' : 'Выберите блюдо'}</>}
            </Button>
            {pairing && (
              <div className="rounded-2xl bg-white p-5 text-foreground shadow-xl">
                <div className={`flex items-center gap-2 ${pairing.verdict === 'not_ideal' ? 'text-amber-800' : 'text-emerald-800'}`}><BadgeCheck className="size-5" /><span className="text-xs font-bold uppercase tracking-[0.12em]">{pairing.title || 'Рекомендация готова'}</span></div>
                <p className="mt-3 text-sm leading-6">{pairing.explanation}</p>
                <div className="mt-4 grid grid-cols-1 gap-2 text-xs sm:grid-cols-2">
                  {pairing.serving_temperature && <p className="flex items-center gap-2 rounded-xl bg-muted px-3 py-2.5"><ThermometerSun className="size-4 shrink-0 text-primary" /> Подавать при {pairing.serving_temperature}</p>}
                  {pairing.glass && <p className="flex items-center gap-2 rounded-xl bg-muted px-3 py-2.5"><Wine className="size-4 shrink-0 text-primary" /> Бокал: {pairing.glass}</p>}
                </div>
                {pairing.pair_with && pairing.pair_with.length > 0 && <p className="mt-3 text-xs leading-5 text-muted-foreground">Также хорошо сочетается: {pairing.pair_with.join(', ')}.</p>}
                {pairing.alternatives && pairing.alternatives.length > 0 && (
                  <div className="mt-4 border-t border-border pt-4">
                    <p className="text-xs font-bold uppercase tracking-[0.12em] text-primary">Сомелье предлагает</p>
                    <div className="mt-3 grid grid-cols-1 gap-2">
                      {pairing.alternatives.map((item) => (
                        <button key={item.slug} onClick={() => onOpenSlug(item.slug)} className="flex min-w-0 items-center gap-3 rounded-xl bg-muted/60 px-3 py-3 text-left transition hover:bg-muted">
                          <span className="min-w-0 flex-1"><strong className="block truncate text-sm">{item.name}</strong><span className="block truncate text-xs text-muted-foreground">{item.winery} · {item.reason}</span></span>
                          <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      <AnalogsSection slug={wine.slug} onOpen={onOpenWine} />
    </div>
  );
}

function servingHint(wine: WineRecord) {
  const text = `${wine.name} ${wine.category ?? ''}`.toLowerCase();
  if (text.includes('брют') || text.includes('игрист')) return '6–8 °C';
  if (text.includes('красн')) return '16–18 °C';
  return '8–12 °C';
}

function SommelierStep({ index, title, children }: { index: number; title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-white/55">{index}. {title}</p>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} aria-pressed={active} className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${active ? 'border-white bg-white text-primary' : 'border-white/15 bg-white/7 text-white hover:bg-white/12'}`}>
      {children}
    </button>
  );
}

function WineStrip({ items, onOpen }: { items: { wine: WineRecord; note?: string }[]; onOpen: (wine: WineRecord) => void }) {
  return (
    <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map(({ wine, note }) => (
        <button key={wine.slug} onClick={() => onOpen(wine)} className="group flex flex-col rounded-2xl bg-card p-3 text-left ring-1 ring-primary/10 transition hover:-translate-y-0.5 hover:shadow-lg">
          <span className="grid h-36 place-items-center rounded-xl bg-[#f9f1f1]">
            {wine.image_url ? <img src={wine.image_url} alt="" loading="lazy" className="max-h-32 max-w-full object-contain" /> : <Wine className="size-8 text-primary/50" />}
          </span>
          <span className="mt-3 block truncate text-[10px] font-bold uppercase tracking-[0.12em] text-primary">{wine.winery}</span>
          <strong className="mt-1 line-clamp-2 text-sm leading-5">{wine.name}</strong>
          {note && <span className="mt-2 text-xs text-muted-foreground">{note}</span>}
        </button>
      ))}
    </div>
  );
}

function AnalogsSection({ slug, onOpen }: { slug: string; onOpen: (wine: WineRecord) => void }) {
  const [items, setItems] = useState<AnalogWine[]>([]);
  useEffect(() => {
    let cancelled = false;
    apiGet<AnalogWine[]>(`/v1/wines/${encodeURIComponent(slug)}/analogs?limit=4`)
      .then((result) => { if (!cancelled) setItems(result.map((item) => ({ ...item, wine: withApiImages(item.wine) }))); })
      .catch(() => { if (!cancelled) setItems([]); });
    return () => { cancelled = true; };
  }, [slug]);
  if (!items.length) return null;
  return (
    <section className="mt-10">
      <span className="text-xs font-bold uppercase tracking-[0.16em] text-primary">Аналоги</span>
      <h2 className="mt-2 font-heading text-2xl sm:text-3xl">Похожие вина других виноделен</h2>
      <WineStrip items={items.map((item) => ({ wine: item.wine, note: item.reasons.slice(0, 2).join(' · ') }))} onOpen={onOpen} />
    </section>
  );
}

function NotFoundState({ previewUrl, candidates, analogs, onOpen, onRetry }: {
  previewUrl: string | null;
  candidates: RankedWine[];
  analogs: AnalogWine[];
  onOpen: (wine: WineRecord) => void;
  onRetry: () => void;
}) {
  return (
    <section className="mx-auto w-full max-w-6xl px-5 pb-16 pt-3 sm:px-8 sm:pt-8">
      <Card className="rounded-[2rem] border-0 py-0 shadow-[0_24px_80px_rgb(114_49_53/10%)] ring-1 ring-primary/10">
        <CardContent className="grid gap-7 p-5 sm:grid-cols-[180px_1fr] sm:p-8">
          <div className="aspect-square overflow-hidden rounded-2xl bg-muted">
            {previewUrl ? <img src={previewUrl} alt="Ваша фотография" className="size-full object-cover" /> : <div className="grid size-full place-items-center"><Wine className="size-9 text-muted-foreground" /></div>}
          </div>
          <div className="self-center">
            <Badge variant="outline" className="border-primary/20 bg-accent text-accent-foreground">Нет в каталоге</Badge>
            <h1 className="mt-4 font-heading text-3xl leading-tight">Этого вина пока нет в каталоге «Своё Вино»</h1>
            <p className="mt-3 leading-7 text-muted-foreground">Мы не нашли точной карточки и не стали выдавать похожую за найденную. Посмотрите близкие по этикетке вина и аналоги по стилю — или снимите этикетку ещё раз, ближе и без блика.</p>
            <Button className="mt-6 h-11 rounded-xl" onClick={onRetry}><RefreshCw /> Сканировать снова</Button>
          </div>
        </CardContent>
      </Card>
      {candidates.length > 0 && (
        <div className="mt-10">
          <h2 className="font-heading text-2xl">Похожие по этикетке</h2>
          <WineStrip items={candidates.slice(0, 4).map((item) => ({ wine: item.wine }))} onOpen={onOpen} />
        </div>
      )}
      {analogs.length > 0 && (
        <div className="mt-10">
          <h2 className="font-heading text-2xl">Аналоги по стилю</h2>
          <WineStrip items={analogs.map((item) => ({ wine: item.wine, note: item.reasons.slice(0, 2).join(' · ') }))} onOpen={onOpen} />
        </div>
      )}
    </section>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return <div className="px-2 py-7 text-center sm:py-9"><strong className="block font-heading text-xl text-primary sm:text-3xl">{value}</strong><span className="mt-1 block text-[10px] text-muted-foreground sm:text-xs">{label}</span></div>;
}

function Feature({ icon, title, copy }: { icon: React.ReactNode; title: string; copy: string }) {
  return <div className="flex gap-3"><span className="grid size-10 shrink-0 place-items-center rounded-xl bg-accent text-accent-foreground [&_svg]:size-4">{icon}</span><span><strong className="block text-sm">{title}</strong><span className="mt-1 block text-xs leading-5 text-muted-foreground">{copy}</span></span></div>;
}

function WineFact({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="min-h-28 bg-card p-4"><span className="text-primary [&_svg]:size-4">{icon}</span><span className="mt-3 block text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">{label}</span><strong className="mt-1 block text-xs leading-5">{value}</strong></div>;
}

function Alternatives({ wines }: { wines: WineRecord[] }) {
  return (
    <section className="mt-10">
      <h2 className="font-heading text-2xl">Похожие вина из каталога</h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {wines.slice(0, 3).map((item) => <Card key={item.slug} size="sm"><CardContent><p className="text-xs font-bold uppercase tracking-[0.12em] text-primary">{item.winery}</p><h3 className="mt-2 font-heading text-lg">{item.name}</h3><p className="mt-2 text-xs text-muted-foreground">{item.category || item.region}</p></CardContent></Card>)}
      </div>
    </section>
  );
}

function RecentScans({ wines, onOpen }: { wines: WineRecord[]; onOpen: (wine: WineRecord) => void }) {
  return (
    <section className="mx-auto w-full max-w-6xl px-5 pb-16 sm:px-8">
      <div className="flex items-end justify-between"><div><span className="text-xs font-bold uppercase tracking-[0.16em] text-primary">На этом устройстве</span><h2 className="mt-2 font-heading text-3xl">Недавние находки</h2></div></div>
      <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {wines.slice(0, 3).map((item) => <button key={item.slug} onClick={() => onOpen(item)} className="flex items-center gap-4 rounded-2xl bg-card p-4 text-left ring-1 ring-primary/10 transition hover:-translate-y-0.5 hover:shadow-lg"><span className="grid size-12 place-items-center rounded-xl bg-muted text-primary"><Wine /></span><span className="min-w-0 flex-1"><strong className="block truncate text-sm">{item.name}</strong><span className="mt-1 block truncate text-xs text-muted-foreground">{item.winery}</span></span><ChevronRight className="size-4 text-muted-foreground" /></button>)}
      </div>
    </section>
  );
}
