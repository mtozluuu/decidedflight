(function () {
  function esc(value) {
    const el = document.createElement('div');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
  }

  function levelFor(ft) {
    if (ft < 2500) return 'surface';
    if (ft < 7500) return '850';
    if (ft < 14000) return '700';
    return '500';
  }

  function altitudeFt() {
    const select = document.getElementById('altitudeSelect');
    return select ? parseFloat(select.value) : null;
  }

  function sample(b) {
    const pts = [];
    for (let i = 0; i <= 4; i += 1) {
      for (let j = 0; j <= 4; j += 1) {
        pts.push([b.getSouth() + ((b.getNorth() - b.getSouth()) * i) / 4, b.getWest() + ((b.getEast() - b.getWest()) * j) / 4]);
      }
    }
    return pts;
  }

  function touches(feature, bounds, pts) {
    const geom = feature.geometry;
    if (!geom || geom.type !== 'Polygon') return false;
    const ring = geom.coordinates[0];
    if (ring.some((c) => bounds.contains([c[1], c[0]]))) return true;
    return pts.some((p) => RouteGeometry.inFeature(p, feature));
  }

  function attach(map) {
    let bounds = null;
    let bodyEl = null;
    let statusEl = null;
    let panel = null;
    let airspaces = null;
    let requestId = 0;

    async function getAirspaces() {
      if (airspaces) return airspaces;
      const res = await fetch('/api/v1/airspace?categories=CTR,TMA');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Hava sahası alınamadı');
      airspaces = data.features;
      return airspaces;
    }

    async function getSigmets() {
      try {
        const res = await fetch('/api/v1/map-layers/sigmet');
        const data = await res.json();
        return res.ok ? data.sigmets : null;
      } catch (err) {
        return null;
      }
    }

    async function update() {
      if (!bounds) return;
      const current = ++requestId;
      const ft = altitudeFt();
      MapUtil.setStatus(statusEl, 'Hesaplanıyor…');
      bodyEl.innerHTML = '';
      try {
        const [features, sigmets] = await Promise.all([getAirspaces(), getSigmets()]);
        if (current !== requestId) return;
        const pts = sample(bounds);
        const hits = features.filter((f) => {
          const p = f.properties;
          const inAlt = ft == null || ((p.lower_ft == null || p.lower_ft <= ft) && (p.upper_ft == null || p.upper_ft >= ft));
          return inAlt && touches(f, bounds, pts);
        });
        const alerts = (sigmets || []).filter((s) => {
          const inAlt = ft == null || ((s.base_ft == null || s.base_ft <= ft) && (s.top_ft == null || s.top_ft >= ft));
          return inAlt && s.coords && s.coords.length && L.latLngBounds(s.coords).intersects(bounds);
        });
        const list = hits.slice(0, 8).map((f) => '<li>' + esc(f.properties.name) + ' <span class="region-limits">' + esc(f.properties.lower) + '–' + esc(f.properties.upper) + '</span></li>').join('');
        bodyEl.innerHTML =
          '<p><b>' + hits.length + '</b> TMA/CTR' + (ft != null ? ' (' + ft + ' ft)' : '') + ' bölgeyle kesişiyor</p>' +
          (list ? '<ul>' + list + '</ul>' : '') + (hits.length > 8 ? '<p>+' + (hits.length - 8) + ' tane daha</p>' : '') +
          '<p>' + (sigmets == null ? 'SIGMET verisi alınamadı' : '<b>' + alerts.length + '</b> SIGMET bölgeyi etkiliyor') + '</p>';
        MapUtil.setStatus(statusEl, 'Seçili irtifa katına göre · doğrulama gerekir');
      } catch (err) {
        MapUtil.setStatus(statusEl, err.message, update);
      }
    }

    function openPanel() {
      if (!panel || !panel.classList.contains('collapsed')) return;
      panel.parentElement.querySelectorAll('.airspace-control').forEach((p) => p.classList.add('collapsed'));
      panel.classList.remove('collapsed');
      panel.scrollIntoView({ block: 'nearest' });
    }

    function share() {
      const c = map.getCenter();
      let prefs = {};
      try { prefs = JSON.parse(localStorage.getItem(MapUtil.PREF_KEY)) || {}; } catch (err) { /* ignore */ }
      const state = { c: [Number(c.lat.toFixed(4)), Number(c.lng.toFixed(4))], z: map.getZoom(), s: prefs };
      if (bounds) state.r = [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()].map((v) => Number(v.toFixed(4)));
      const route = map._routeApi && map._routeApi.get();
      if (route && route.from && route.to) state.rt = route;
      const alt = document.getElementById('altitudeSelect');
      if (alt) state.a = alt.value;
      const url = location.origin + location.pathname + '#harita=' + btoa(unescape(encodeURIComponent(JSON.stringify(state))));
      const done = () => MapUtil.setStatus(statusEl, 'Bağlantı kopyalandı');
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(done, () => window.prompt('Bağlantı', url));
      } else {
        window.prompt('Bağlantı', url);
      }
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar region-control');
      box.innerHTML = '<strong>Bölge özeti</strong>';
      bodyEl = L.DomUtil.create('div', 'region-body', box);
      bodyEl.innerHTML = '<p>Dikdörtgen çizince kesişen TMA/CTR ve SIGMET burada özetlenir.</p>';
      const btn = L.DomUtil.create('button', 'airspace-play route-go', box);
      btn.type = 'button';
      btn.textContent = 'Görünümü paylaş';
      btn.addEventListener('click', share);
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      L.DomEvent.disableClickPropagation(box);
      panel = box;
      return box;
    };
    control.addTo(map);

    map.on(L.Draw.Event.CREATED, (e) => {
      bounds = e.layer.getBounds();
      openPanel();
      update();
    });

    function applyAltitude() {
      const ft = altitudeFt();
      if (ft == null) return;
      if (map._windApi) map._windApi.setLevel(levelFor(ft));
      if (map._airspaceApi) map._airspaceApi.setFilter(Math.round(ft / 100));
      update();
    }
    const altSelect = document.getElementById('altitudeSelect');
    if (altSelect) altSelect.addEventListener('change', applyAltitude);
    let timeTimer = null;
    document.addEventListener('df:timechange', (e) => {
      clearTimeout(timeTimer);
      timeTimer = setTimeout(() => { if (map._windApi) map._windApi.setHour(e.detail.hour); }, 400);
    });

    const shared = MapUtil.sharedState();
    if (shared) {
      if (altSelect && shared.a) { altSelect.value = shared.a; applyAltitude(); }
      if (shared.c) map.setView(shared.c, shared.z || 6);
      if (shared.r) {
        const rect = L.rectangle([[shared.r[0], shared.r[1]], [shared.r[2], shared.r[3]]], { color: '#5aa9ff', weight: 2, fillOpacity: 0.08 });
        map.fire(L.Draw.Event.CREATED, { layer: rect, layerType: 'rectangle' });
      }
      if (shared.rt && map._routeApi) map._routeApi.set(shared.rt);
    }
  }

  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
