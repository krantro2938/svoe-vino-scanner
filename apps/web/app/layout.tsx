import type { Metadata } from 'next';
import { Playfair_Display } from 'next/font/google';

import './globals.css';

// «Своё Вино» sets headings in Playfair Display and body copy in the system UI font.
const playfair = Playfair_Display({
  variable: '--font-playfair',
  weight: ['400', '500', '600', '700'],
  subsets: ['cyrillic', 'latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.SITE_ORIGIN ?? 'http://localhost:3000'),
  title: 'Сканер российских вин — Своё Вино',
  description: 'Сфотографируйте этикетку и откройте точную карточку российского вина.',
  openGraph: {
    title: 'Сканер российских вин',
    description: 'Наведите камеру. Найдите своё.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Сканер российских вин' }],
    locale: 'ru_RU',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Сканер российских вин',
    description: 'Наведите камеру. Найдите своё.',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ru">
      <body className={`${playfair.variable} antialiased`}>{children}</body>
    </html>
  );
}
