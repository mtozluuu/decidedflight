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

    function collect(withPrefs) {
      const c = map.getCenter();
      const state = { c: [Number(c.lat.toFixed(4)), Number(c.lng.toFixed(4))], z: map.getZoom() };
      if (withPrefs) {
        try { state.s = JSON.parse(localStorage.getItem(MapUtil.PREF_KEY)) || {}; } catch (err) { state.s = {}; }
      }
      if (bounds) state.r = [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()].map((v) => Number(v.toFixed(4)));
      const route = map._routeApi && map._routeApi.get();
      if (route && route.from && route.to) state.rt = route;
      const alt = document.getElementById('altitudeSelect');
      if (alt) state.a = alt.value;
      return state;
    }

    function restore(state) {
      const altSelect = document.getElementById('altitudeSelect');
      if (altSelect && state.a) { altSelect.value = state.a; applyAltitude(); }
      if (state.c) map.setView(state.c, state.z || 6);
      if (state.r) {
        const rect = L.rectangle([[state.r[0], state.r[1]], [state.r[2], state.r[3]]], { color: '#5aa9ff', weight: 2, fillOpacity: 0.08 });
        map.fire(L.Draw.Event.CREATED, { layer: rect, layerType: 'rectangle' });
      }
      if (state.rt && map._routeApi) map._routeApi.set(state.rt);
    }

    function share() {
      const url = location.origin + location.pathname + '#harita=' + btoa(unescape(encodeURIComponent(JSON.stringify(collect(true)))));
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

    const SAVED_KEY = 'decideflight.savedViews';
    const readSaved = () => { try { return JSON.parse(localStorage.getItem(SAVED_KEY)) || []; } catch (err) { return []; } };

    const savedControl = L.control({ position: 'topright' });
    savedControl.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar region-control');
      box.innerHTML = '<strong>Kayıtlı ve uyarı</strong>';
      const nameInput = L.DomUtil.create('input', 'airspace-select', box);
      nameInput.placeholder = 'Ad (ör. Ege rotası)';
      const saveBtn = L.DomUtil.create('button', 'airspace-play route-go', box);
      saveBtn.type = 'button';
      saveBtn.textContent = 'Bölge/rotayı kaydet';
      const select = L.DomUtil.create('select', 'airspace-select', box);
      const rowBtns = L.DomUtil.create('div', 'route-form', box);
      const openBtn = L.DomUtil.create('button', 'airspace-play route-go', rowBtns);
      openBtn.type = 'button';
      openBtn.textContent = 'Aç';
      const delBtn = L.DomUtil.create('button', 'airspace-play route-go', rowBtns);
      delBtn.type = 'button';
      delBtn.textContent = 'Sil';
      const fill = () => {
        const list = readSaved();
        select.innerHTML = list.length ? list.map((v, i) => '<option value="' + i + '">' + esc(v.name) + '</option>').join('') : '<option value="">Kayıt yok</option>';
      };
      fill();
      saveBtn.addEventListener('click', () => {
        const name = nameInput.value.trim();
        if (!name) { MapUtil.setStatus(savedStatus, 'Önce bir ad girin'); return; }
        const list = readSaved().filter((v) => v.name !== name);
        list.push({ name, state: collect(false) });
        try { localStorage.setItem(SAVED_KEY, JSON.stringify(list)); } catch (err) { /* storage unavailable */ }
        fill();
        select.value = String(list.length - 1);
        MapUtil.setStatus(savedStatus, 'Kaydedildi');
      });
      openBtn.addEventListener('click', () => {
        const item = readSaved()[Number(select.value)];
        if (item) restore(item.state);
      });
      delBtn.addEventListener('click', () => {
        const list = readSaved();
        list.splice(Number(select.value), 1);
        try { localStorage.setItem(SAVED_KEY, JSON.stringify(list)); } catch (err) { /* storage unavailable */ }
        fill();
      });

      const watchRow = L.DomUtil.create('label', 'airspace-row', box);
      const watchBox = L.DomUtil.create('input', '', watchRow);
      watchBox.type = 'checkbox';
      watchRow.appendChild(document.createTextNode(' Uyarıları izle (5 dk)'));
      const alertsEl = L.DomUtil.create('div', 'region-body region-alerts', box);
      const savedStatus = L.DomUtil.create('div', 'airspace-status', box);

      let timer = null;
      let seen = null;
      let categories = null;
      let airportList = null;

      const sigKey = (x) => [x.hazard, x.fir, x.valid_from, x.valid_to].join('|');
      const notify = (text) => {
        const li = document.createElement('p');
        li.textContent = new Date().toTimeString().slice(0, 5) + ' · ' + text;
        alertsEl.prepend(li);
        if (window.Notification && Notification.permission === 'granted') new Notification('DecideFlight', { body: text });
      };

      async function check() {
        const path = map._routeApi && map._routeApi.path();
        if (!bounds && !path) {
          MapUtil.setStatus(savedStatus, 'İzlemek için bölge çizin veya rota analiz edin');
          return;
        }
        try {
          const [sig, st, apt] = await Promise.all([
            fetch('/api/v1/map-layers/sigmet').then((r) => r.json()),
            fetch('/api/v1/map-layers/airports/status').then((r) => r.json()),
            airportList ? Promise.resolve(airportList) : fetch('/api/v1/map-layers/airports').then((r) => r.json()),
          ]);
          airportList = apt;
          const relevant = (sig.sigmets || []).filter((x) => x.coords && x.coords.length > 2 && (
            (bounds && L.latLngBounds(x.coords).intersects(bounds)) ||
            (path && path.along.some((p) => RouteGeometry.inRing(p, x.coords, 1, 0)))));
          const watched = new Set();
          if (path) [path.from, path.to].forEach((a) => a.icao && watched.add(a.icao));
          if (bounds) apt.airports.forEach((a) => bounds.contains([a.lat, a.lon]) && watched.add(a.icao));
          const now = {};
          watched.forEach((icao) => { if (st.stations && st.stations[icao]) now[icao] = st.stations[icao].category; });
          if (seen) {
            relevant.filter((x) => !seen.has(sigKey(x))).forEach((x) => notify('Yeni SIGMET: ' + (x.hazard || '') + ' (' + (x.fir || '') + ')'));
            Object.keys(now).forEach((icao) => {
              if (categories[icao] && categories[icao] !== now[icao] && ['IFR', 'LIFR'].includes(now[icao])) notify(icao + ' ' + categories[icao] + ' → ' + now[icao]);
            });
          }
          seen = new Set(relevant.map(sigKey));
          categories = now;
          MapUtil.setStatus(savedStatus, relevant.length + ' SIGMET, ' + watched.size + ' meydan izleniyor');
        } catch (err) {
          MapUtil.setStatus(savedStatus, 'İzleme verisi alınamadı');
        }
      }

      watchBox.addEventListener('change', () => {
        clearInterval(timer);
        seen = null;
        categories = null;
        if (!watchBox.checked) { MapUtil.setStatus(savedStatus, ''); return; }
        if (window.Notification && Notification.permission === 'default') Notification.requestPermission();
        check();
        timer = setInterval(check, 5 * 60 * 1000);
      });
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    savedControl.addTo(map);

    const LEVEL_OPTIONS = [['surface', 'Yüzey'], ['850', '850 hPa'], ['700', '700 hPa'], ['500', '500 hPa']];
    const HOUR_OPTIONS = [0, 3, 6, 12, 24];
    const compareControl = L.control({ position: 'topright' });
    compareControl.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar region-control');
      box.innerHTML = '<strong>Seviye/saat kıyası</strong>';
      const grid = L.DomUtil.create('div', 'route-form', box);
      const picks = ['A', 'B'].map((name, i) => {
        const level = L.DomUtil.create('select', 'airspace-select', grid);
        level.setAttribute('aria-label', name + ' seviyesi');
        level.innerHTML = LEVEL_OPTIONS.map(([v, t]) => '<option value="' + v + '">' + name + ': ' + t + '</option>').join('');
        level.value = i ? '500' : 'surface';
        const hour = L.DomUtil.create('select', 'airspace-select', grid);
        hour.setAttribute('aria-label', name + ' saati');
        hour.innerHTML = HOUR_OPTIONS.map((h) => '<option value="' + h + '">' + name + ': +' + h + ' sa</option>').join('');
        return { level, hour };
      });
      const go = L.DomUtil.create('button', 'airspace-play route-go', box);
      go.type = 'button';
      go.textContent = 'Seçili bölgeyi karşılaştır';
      const out = L.DomUtil.create('div', 'region-body', box);
      const status = L.DomUtil.create('div', 'airspace-status', box);

      async function stats(pick) {
        const res = await fetch('/api/v1/map-layers/wind-cloud?level=' + pick.level.value + '&hour=' + pick.hour.value);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Veri alınamadı');
        let pts = data.points.filter((p) => bounds.contains([p.lat, p.lon]) && p.wind_speed_knots != null);
        if (!pts.length) {
          const c = bounds.getCenter();
          pts = [data.points.reduce((a, b) => ((b.lat - c.lat) ** 2 + (b.lon - c.lng) ** 2 < (a.lat - c.lat) ** 2 + (a.lon - c.lng) ** 2 ? b : a))];
        }
        const avg = (key) => pts.reduce((sum, p) => sum + (p[key] || 0), 0) / pts.length;
        return {
          label: pick.level.options[pick.level.selectedIndex].text + ' +' + pick.hour.value + ' sa',
          wind: avg('wind_speed_knots'),
          maxWind: Math.max(...pts.map((p) => p.wind_speed_knots)),
          cloud: avg('cloud_cover_pct'),
          n: pts.length,
        };
      }

      go.addEventListener('click', async () => {
        if (!bounds) { MapUtil.setStatus(status, 'Önce haritada dikdörtgen çizin'); return; }
        MapUtil.setStatus(status, 'Hesaplanıyor…');
        try {
          const a = await stats(picks[0]);
          const b = await stats(picks[1]);
          const row = (x) => '<tr><td>' + esc(x.label) + '</td><td>' + Math.round(x.wind) + '/' + Math.round(x.maxWind) + ' kt</td><td>%' + Math.round(x.cloud) + '</td></tr>';
          out.innerHTML = '<table class="route-table"><tr><th>Seçim</th><th>Rüzgâr ort./maks.</th><th>Bulut</th></tr>' + row(a) + row(b) + '</table>' +
            '<p>Fark (B−A): ' + (b.wind - a.wind >= 0 ? '+' : '') + Math.round(b.wind - a.wind) + ' kt rüzgâr, ' + (b.cloud - a.cloud >= 0 ? '+' : '') + Math.round(b.cloud - a.cloud) + ' puan bulut</p>';
          MapUtil.setStatus(status, 'Model ızgarası (~' + a.n + ' nokta) · tahmin');
        } catch (err) {
          MapUtil.setStatus(status, err.message, () => go.click());
        }
      });
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    compareControl.addTo(map);

    const shared = MapUtil.sharedState();
    if (shared) restore(shared);
  }

  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
