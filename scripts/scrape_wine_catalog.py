#!/usr/bin/env python3
"""Resume a provenance-preserving snapshot of vino-svoe.ru public SSR pages.

No private/API endpoints. HTML and image responses are cached on disk. Catalogue
metadata is never used to overwrite the official competition catalogue.
"""
from __future__ import annotations
import argparse
import binascii
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

BASE = 'https://vino-svoe.ru'
USER_AGENT = 'WineCatalogResearch/1.0 (public catalogue dataset; low-rate resumable)'


def parse_listing(html: str) -> dict:
    match = re.search(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        raise ValueError('Missing Nuxt SSR catalogue payload')
    values = json.loads(match.group(1))
    def resolve(index):
        if index < 0:
            return None
        value = values[index]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                return resolve(value[1]) if len(value) > 1 else []
            return [resolve(item) for item in value]
        return value
    for index, value in enumerate(values):
        if isinstance(value, dict) and {'currentPage', 'totalItems', 'totalPages', 'items'} <= value.keys():
            return resolve(index)
    raise ValueError('Missing paginated catalogue object')


def normalize(item: dict, page: int, observed: str) -> dict:
    slug = item['slug']
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', slug):
        raise ValueError(f'Unsafe slug {slug!r}')
    image = item.get('image') or {}
    image_path = image.get('url', '')
    # This is the exact rendition linked by the public catalogue HTML.
    image_url = 'https://api.vino-svoe.ru/v1/img/str-api/910/910/resize' + image_path if image_path.startswith('/uploads/') else None
    return dict(slug=slug, name=item.get('title', ''), category=item.get('category', ''),
                color=item.get('color', ''), winery=item.get('manufacturer', ''),
                region=item.get('region', ''), image_url=image_url,
                photo_name=Path(image_path).name, source_image_path=image_path,
                source_url=BASE + '/wines/' + slug, source_page=page,
                observed_at=observed, source='vino-svoe.ru public listing',
                metadata_scope='listing', image_rendition='public 910px rendition',
                index_ready=False, media_status='pending')


class Fetcher:
    def __init__(self, delay: float):
        self.delay = delay
        self.lock = threading.Lock()
        self.last = 0.0
    def get(self, url: str) -> bytes:
        for attempt in range(4):
            with self.lock:
                time.sleep(max(0, self.last + self.delay - time.monotonic()))
                self.last = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.read()
            except (OSError, urllib.error.URLError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


def validate_image_signature(payload: bytes) -> None:
    webp = payload.startswith(b'RIFF') and payload[8:12] == b'WEBP'
    if not (webp or payload.startswith(b'\xff\xd8') or payload.startswith(b'\x89PNG\r\n\x1a\n')):
        raise ValueError('Image response is not WebP/JPEG/PNG')


def dump_jsonl(path, rows):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/external/vino-svoe'))
    parser.add_argument('--delay', type=float, default=0.5, help='Minimum global seconds between requests')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-images', action='store_true')
    parser.add_argument('--max-pages', type=int, default=0)
    args = parser.parse_args()
    if args.delay < 0.25 or not 1 <= args.workers <= 8:
        parser.error('delay must be >=0.25s and workers 1..8')
    out = args.output
    (out / 'pages').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    fetcher = Fetcher(args.delay)
    robots_path = out / 'robots.txt'
    if not robots_path.exists():
        robots_path.write_bytes(fetcher.get(BASE + '/robots.txt'))
    robots = robots_path.read_text()
    # The live site's wildcard query exclusion explicitly allows pagination.
    if not re.search(r'^Allow:\s*\*page=', robots, re.M):
        raise RuntimeError('Robots policy changed; review before pagination')
    image_robots_path = out / 'robots-images.txt'
    if not image_robots_path.exists():
        image_robots_path.write_bytes(fetcher.get('https://api.vino-svoe.ru/robots.txt'))
    if not re.search(r'^Allow:\s*\*/img/\*', image_robots_path.read_text(), re.M):
        raise RuntimeError('Image-host robots policy changed; review before image download')
    observed = datetime.now(timezone.utc).isoformat()
    def listing(page):
        path = out / 'pages' / f'{page:04d}.html'
        if not path.exists():
            payload = fetcher.get(BASE + '/wines' + (f'?page={page}' if page > 1 else ''))
            parsed = parse_listing(payload.decode())
            if parsed['currentPage'] != page:
                raise ValueError(f'Unexpected currentPage for page {page}')
            path.write_bytes(payload)
        return parse_listing(path.read_text())
    first = listing(1)
    total_pages = min(first['totalPages'], args.max_pages or first['totalPages'])
    pages = {1: first}
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(listing, page): page for page in range(2, total_pages + 1)}
        for future in as_completed(futures):
            page = futures[future]
            try:
                pages[page] = future.result()
            except Exception as exc:
                errors.append({'page': page, 'error': str(exc)})
            if len(pages) % 10 == 0:
                print(f'Pages {len(pages)}/{total_pages}', flush=True)
    records = {}
    duplicates = []
    for page, listing_data in sorted(pages.items()):
        for item in listing_data['items']:
            row = normalize(item, page, observed)
            if row['slug'] in records:
                duplicates.append(row['slug'])
            records[row['slug']] = row
    rows = sorted(records.values(), key=lambda row: row['slug'])
    dump_jsonl(out / 'catalog.jsonl', rows)
    organizer = {}
    manifest = Path('data/generated/catalog/extracted_media.jsonl')
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            if entry.get('index_ready') and len(entry.get('slugs', [])) == 1:
                key = (entry['slugs'][0], Path(entry['source_archive_path']).name)
                organizer[key] = entry
    def download(row):
        row = dict(row)
        if not row['image_url']:
            row['media_status'] = 'missing_url'
            return row
        path = out / 'images' / (row['slug'] + Path(urlsplit(row['image_url']).path).suffix)
        try:
            entry = organizer.get((row['slug'], Path(row['source_image_path']).name))
            reused = False
            if entry:
                source = Path('data') / entry['images_dir'] / entry['local_path']
                if source.is_file():
                    original = source.read_bytes()
                    crc = f'{binascii.crc32(original) & 0xffffffff:08X}'
                    if len(original) == entry['size'] and crc == entry['crc32'].upper():
                        shutil.copyfile(source, path)
                        reused = True
                        row.update(organizer_source_path=str(source), organizer_crc32=crc,
                                   organizer_archive_path=entry['source_archive_path'],
                                   image_rendition='organizer original same public upload')
            if not path.exists():
                payload = fetcher.get(row['image_url'])
                validate_image_signature(payload)
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
            payload = path.read_bytes()
            validate_image_signature(payload)
            row.update(image_path=str(path.relative_to(out)), image_sha256=hashlib.sha256(payload).hexdigest(),
                       media_path=str(path), media_sha256=hashlib.sha256(payload).hexdigest(),
                       media_bytes=len(payload), media_status='reused_organizer' if reused else 'downloaded', index_ready=True)
        except Exception as exc:
            row.update(media_status='failed', media_error=str(exc))
        return row
    if not args.skip_images:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, completed in enumerate(pool.map(download, rows), 1):
                rows[count - 1] = completed
                if count % 100 == 0:
                    dump_jsonl(out / 'catalog.jsonl', rows)
                    print(f'Images {count}/{len(rows)}', flush=True)
    dump_jsonl(out / 'catalog.jsonl', rows)
    report = dict(observed_at=observed, source=BASE + '/wines', expected_items=first['totalItems'],
                  expected_pages=first['totalPages'], fetched_pages=len(pages), unique_slugs=len(rows),
                  duplicate_slugs=duplicates, downloaded_images=sum(r['media_status']=='downloaded' for r in rows),
                  reused_organizer_images=sum(r['media_status']=='reused_organizer' for r in rows),
                  failed_images=[r['slug'] for r in rows if r['media_status']=='failed'], errors=errors,
                  complete=len(pages)==first['totalPages'] and len(rows)==first['totalItems'] and not errors,
                  license_status='No redistribution/training license asserted; public source retained for review',
                  note='Official competition catalogue remains authoritative; snapshot uses public listing metadata and rendered images.')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
