const CACHE_NAME = 'decideflight-static-v44';
const DATA_CACHE = 'decideflight-map-data';
const OFFLINE_API = ['/api/v1/airspace', '/api/v1/map-layers/airports'];
const STATIC_ASSETS = [
  '/',
  '/static/index.html',
  '/static/theme.css',
  '/static/airspace.css',
  '/static/airspace.js',
  '/static/radar.js',
  '/static/wind.js',
  '/static/airports.js',
  '/static/sigmet.js',
  '/static/route.js',
  '/static/region.js',
  '/static/manifest.json',
  '/static/decideflight-icon.svg',
  '/static/data/cities.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== CACHE_NAME && key !== DATA_CACHE).map((key) => caches.delete(key))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const requestUrl = new URL(event.request.url);
  const isOfflineApi = event.request.method === 'GET'
    && requestUrl.origin === self.location.origin
    && OFFLINE_API.some((path) => requestUrl.pathname === path || requestUrl.pathname + requestUrl.search === path);
  if (isOfflineApi) {
    event.respondWith(
      fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(DATA_CACHE).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match(event.request, { cacheName: DATA_CACHE }).then(
        (cached) => cached || new Response('{"detail":"Çevrimdışı ve önbellekte veri yok."}', {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })
      ))
    );
    return;
  }
  if (
    event.request.method !== 'GET'
    || requestUrl.origin !== self.location.origin
    || requestUrl.pathname.startsWith('/api/')
  ) return;

  event.respondWith(
    fetch(event.request).then((response) => {
      if (response.ok) {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
      }
      return response;
    }).catch((error) => {
      console.error('Service worker fetch failed:', error);
      return caches.match(event.request).then((cached) => cached || caches.match('/static/index.html')).then(
        (fallback) => fallback || new Response('Çevrimdışı içerik kullanılamıyor.', {
          status: 503,
          statusText: 'Offline',
          headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        })
      );
    })
  );
});
