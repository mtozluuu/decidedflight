(function () {
  const SAMPLE_KM = 5;

  function esc(value) {
    const el = document.createElement('div');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
  }

  function rad(d) { return (d * Math.PI) / 180; }
  function deg(r) { return (r * 180) / Math.PI; }

  function distanceKm(a, b) {
    const dLat = rad(b[0] - a[0]);
    const dLon = rad(b[1] - a[1]);
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(dLon / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(h));
  }

  // Great-circle interpolation; returns [lat, lon] points spaced about SAMPLE_KM apart.
  function samplePath(a, b) {
    const km = distanceKm(a, b);
    const n = Math.max(2, Math.ceil(km / SAMPLE_KM));
    const d = km / 6371;
    const toVec = (p) => [Math.cos(rad(p[0])) * Math.cos(rad(p[1])), Math.cos(rad(p[0])) * Math.sin(rad(p[1])), Math.sin(rad(p[0]))];
    const va = toVec(a);
    const vb = toVec(b);
    const out = [];
    for (let i = 0; i <= n; i += 1) {
      const f = i / n;
      const k1 = d ? Math.sin((1 - f) * d) / Math.sin(d) : 1 - f;
      const k2 = d ? Math.sin(f * d) / Math.sin(d) : f;
      const x = k1 * va[0] + k2 * vb[0];
      const y = k1 * va[1] + k2 * vb[1];
      const z = k1 * va[2] + k2 * vb[2];
      out.push([deg(Math.atan2(z, Math.hypot(x, y))), deg(Math.atan2(y, x))]);
    }
    return { points: out, km };
  }

  function inRing(pt, ring, lonIndex, latIndex) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][lonIndex], yi = ring[i][latIndex];
      const xj = ring[j][lonIndex], yj = ring[j][latIndex];
      if ((yi > pt[0]) !== (yj > pt[0]) && pt[1] < ((xj - xi) * (pt[0] - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function inFeature(pt, feature) {
    const rings = feature.geometry.coordinates;
    return inRing(pt, rings[0], 0, 1) && !rings.slice(1).some((h) => inRing(pt, h, 0, 1));
  }

  function nearest(points, pt) {
    let best = null;
    let bestD = Infinity;
    points.forEach((p) => {
      const d = (p.lat - pt[0]) ** 2 + (p.lon - pt[1]) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    });
    return best;
  }

  async function getJson(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Veri alınamadı');
    return data;
  }

  function attach(map) {
    let airports = [];
    let line = null;
    let sampleLayer = null;
    let lastPath = null;
    let statusEl = null;
    let resultEl = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function clear() {
      if (line) map.removeLayer(line);
      if (sampleLayer) map.removeLayer(sampleLayer);
      line = null;
      sampleLayer = null;
      lastPath = null;
      resultEl.innerHTML = '';
      setStatus('');
    }

    function find(text) {
      const q = text.trim().toUpperCase();
      const airport = airports.find((a) => a.icao === q);
      if (airport) return airport;
      const m = q.match(/^(-?\d+(?:[.,]\d+)?)\s*[,; ]\s*(-?\d+(?:[.,]\d+)?)$/);
      if (!m) return null;
      const lat = parseFloat(m[1].replace(',', '.'));
      const lon = parseFloat(m[2].replace(',', '.'));
      if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
      return { icao: null, name: lat.toFixed(3) + ', ' + lon.toFixed(3), lat, lon };
    }

    function parseAltitude(text) {
      const q = text.trim().toUpperCase().replace(/\s+/g, '');
      if (!q) return null;
      let m = q.match(/^FL(\d{2,3})$/);
      if (m) return Number(m[1]) * 100;
      m = q.match(/^(\d+(?:[.,]\d+)?)(FT|M)?$/);
      if (!m) return undefined;
      const value = parseFloat(m[1].replace(',', '.'));
      return m[2] === 'M' ? Math.round(value * 3.28084) : Math.round(value);
    }

    function levelFor(ft) {
      if (ft == null || ft < 2500) return 'surface';
      if (ft < 7500) return '850';
      if (ft < 14000) return '700';
      return '500';
    }

    const LEVEL_LABELS = { surface: 'Yüzey (10 m)', 850: '850 hPa (~FL050)', 700: '700 hPa (~FL100)', 500: '500 hPa (~FL180)' };

    function bearing(a, b) {
      const dLon = rad(b[1] - a[1]);
      const y = Math.sin(dLon) * Math.cos(rad(b[0]));
      const x = Math.cos(rad(a[0])) * Math.sin(rad(b[0])) - Math.sin(rad(a[0])) * Math.cos(rad(b[0])) * Math.cos(dLon);
      return (deg(Math.atan2(y, x)) + 360) % 360;
    }

    function label(place) {
      return place.icao || place.name;
    }

    function windStats(data, along, course, km) {
      const rows = along.map((p, i) => {
        const g = nearest(data.points, p);
        if (!g || g.wind_speed_knots == null) return null;
        const head = g.wind_speed_knots * Math.cos(rad(g.wind_direction_deg - course));
        return { km: Math.round((km * i) / (along.length - 1)), g, head, point: p };
      }).filter(Boolean);
      if (!rows.length) return null;
      const clouds = rows.map((r) => r.g.cloud_cover_pct || 0);
      return {
        rows,
        maxWind: Math.max(...rows.map((r) => r.g.wind_speed_knots)),
        avgHead: rows.reduce((sum, r) => sum + r.head, 0) / rows.length,
        avgCloud: clouds.reduce((a, b) => a + b, 0) / clouds.length,
        maxCloud: Math.max(...clouds),
      };
    }

    const headText = (v) => (v >= 0 ? Math.round(v) + ' kt karşı' : Math.round(-v) + ' kt arka');

    async function compareLevels(target) {
      if (!lastPath) return;
      target.innerHTML = '<p>Seviyeler karşılaştırılıyor…</p>';
      const results = [];
      for (const level of Object.keys(LEVEL_LABELS)) {
        try {
          const data = await getJson('/api/v1/map-layers/wind-cloud?level=' + level);
          const st = windStats(data, lastPath.along, lastPath.course, lastPath.km);
          if (st) results.push({ level, st, score: st.avgHead * 0.5 + st.avgCloud / 5 + st.maxWind / 2 });
        } catch (err) { /* level skipped */ }
      }
      if (!results.length) {
        target.innerHTML = '<p>Karşılaştırma için veri alınamadı.</p>';
        return;
      }
      const best = results.reduce((a, b) => (b.score < a.score ? b : a));
      target.innerHTML = '<table class="route-table"><tr><th>Seviye</th><th>Maks. rüzgâr</th><th>Bileşen</th><th>Bulut</th></tr>' +
        results.map((r) => '<tr' + (r === best ? ' class="route-best"' : '') + '><td>' + esc(LEVEL_LABELS[r.level].split(' (')[0]) + '</td><td>' + Math.round(r.st.maxWind) + ' kt</td><td>' + headText(r.st.avgHead) + '</td><td>%' + Math.round(r.st.avgCloud) + '</td></tr>').join('') +
        '</table><p>Önerilen: <b>' + esc(LEVEL_LABELS[best.level]) + '</b> (karşı rüzgâr, en şiddetli rüzgâr ve bulut dengesi). Mevcut 4 seviyeyle sınırlı tahmindir.</p>';
    }

    async function analyse(from, to, altFt) {
      const level = levelFor(altFt);
      setStatus('Hesaplanıyor…');
      const { points, km } = samplePath([from.lat, from.lon], [to.lat, to.lon]);
      if (line) map.removeLayer(line);
      if (sampleLayer) map.removeLayer(sampleLayer);
      line = L.polyline(points, { color: '#ffd166', weight: 3, dashArray: '2 6', interactive: false }).addTo(map);
      map.fitBounds(line.getBounds(), { padding: [60, 60] });

      const [airspace, windData, , sigmet, status] = await Promise.allSettled([
        getJson('/api/v1/airspace?categories=CTR,TMA,FIR'),
        getJson('/api/v1/map-layers/wind-cloud?level=' + level),
        Promise.resolve(null),
        getJson('/api/v1/map-layers/sigmet'),
        getJson('/api/v1/map-layers/airports/status'),
      ]);

      const sections = [];
      sections.push('<p><b>' + esc(label(from)) + ' → ' + esc(label(to)) + '</b> · ' + Math.round(km) + ' km · ' + Math.round(km / 1.852) + ' NM' +
        (altFt != null ? ' · ' + altFt + ' ft' : '') + '</p>');

      if (status.status === 'fulfilled') {
        const st = status.value.stations;
        const cat = (a) => (a.icao ? esc(a.icao) + ': ' + (st[a.icao] ? esc(st[a.icao].category) : 'METAR yok') : null);
        const parts = [cat(from), cat(to)].filter(Boolean);
        if (parts.length) sections.push('<p>Uçuş kategorisi · ' + parts.join(' · ') + '</p>');
      }

      if (airspace.status === 'fulfilled') {
        const hits = [];
        airspace.value.features.forEach((f) => {
          const pr = f.properties;
          const inAlt = altFt == null || ((pr.lower_ft == null || pr.lower_ft <= altFt) && (pr.upper_ft == null || pr.upper_ft >= altFt));
          if (inAlt && points.some((p) => inFeature(p, f))) hits.push(pr);
        });
        const order = { CTR: 0, TMA: 1, FIR: 2 };
        hits.sort((a, b) => order[a.category] - order[b.category]);
        sections.push('<p style="margin-top:6px"><b>Geçilen hava sahaları' + (altFt != null ? ' (' + altFt + ' ft)' : '') + '</b></p>' +
          (hits.length ? '<ul>' + hits.map((h) => '<li>' + esc(h.category) + ' ' + esc(h.name) + ' · ' + esc(h.lower) + '–' + esc(h.upper) + '</li>').join('') + '</ul>' : '<p>—</p>'));
      } else {
        sections.push('<p>Hava sahası alınamadı.</p>');
      }

      const course = bearing([from.lat, from.lon], [to.lat, to.lon]);
      const count = Math.max(4, Math.min(12, Math.ceil(km / 100) + 1));
      const along = Array.from({ length: count }, (_, i) => points[Math.round((i / (count - 1)) * (points.length - 1))]);
      lastPath = { along, course, km, from, to };
      const hitsAt = (p) => (airspace.status !== 'fulfilled' ? [] : airspace.value.features.filter((f) => {
        const pr = f.properties;
        if (pr.category === 'FIR') return false;
        const inAlt = altFt == null || ((pr.lower_ft == null || pr.lower_ft <= altFt) && (pr.upper_ft == null || pr.upper_ft >= altFt));
        return inAlt && inFeature(p, f);
      }).map((f) => f.properties.name));
      let windHtml;
      const st = windData.status === 'fulfilled' ? windStats(windData.value, along, course, km) : null;
      if (windData.status !== 'fulfilled') {
        windHtml = '<p>Rüzgâr/bulut verisi alınamadı.</p>';
      } else if (!st) {
        windHtml = '<p>Veri yok.</p>';
      } else {
        sampleLayer = L.layerGroup().addTo(map);
        st.rows.forEach((r) => {
          r.areas = hitsAt(r.point);
          const tip = r.km + ' km · ' + Math.round(r.g.wind_direction_deg) + '°/' + Math.round(r.g.wind_speed_knots) + ' kt · bulut %' + Math.round(r.g.cloud_cover_pct || 0) + (r.areas.length ? ' · ' + r.areas.join(', ') : '');
          L.circleMarker(r.point, { radius: 5, color: '#14181d', weight: 1, fillColor: '#ffd166', fillOpacity: 1 }).bindTooltip(tip).addTo(sampleLayer);
        });
        windHtml = '<p>Rüzgâr en çok ' + Math.round(st.maxWind) + ' kt · ort. ' + headText(st.avgHead) + ' bileşen (rota ' + Math.round(course) + '°)</p>' +
          '<p>Bulut ort. %' + Math.round(st.avgCloud) + ' · en yoğun %' + Math.round(st.maxCloud) + '</p>' +
          '<table class="route-table"><tr><th>km</th><th>Rüzgâr</th><th>Bulut</th><th>Saha</th></tr>' +
          st.rows.map((r) => '<tr><td>' + r.km + '</td><td>' + Math.round(r.g.wind_direction_deg) + '°/' + Math.round(r.g.wind_speed_knots) + ' kt</td><td>%' + Math.round(r.g.cloud_cover_pct || 0) + '</td><td>' + (r.areas.length ? esc(r.areas.join(', ')) : '—') + '</td></tr>').join('') +
          '</table><button type="button" class="airspace-play route-go route-compare">Seviyeleri karşılaştır</button><div class="route-levels"></div>';
      }
      sections.push('<p style="margin-top:6px"><b>Rota profili · ' + esc(LEVEL_LABELS[level]) + '</b></p>' + windHtml);

      if (sigmet.status === 'fulfilled') {
        const hit = sigmet.value.sigmets.filter((s) => s.coords.length > 2 &&
          points.some((p) => inRing(p, s.coords, 1, 0)));
        sections.push('<p style="margin-top:6px"><b>SIGMET</b>: ' +
          (hit.length ? hit.map((s) => esc(s.hazard) + ' (' + esc(s.fir) + ')').join(', ') : 'rota üzerinde yok') + '</p>');
      }

      sections.push('<p style="opacity:.7;margin-top:6px">Düz hat tahminidir; resmî brifing değildir.</p>');
      resultEl.innerHTML = sections.join('');
      const cmp = resultEl.querySelector('.route-compare');
      if (cmp) cmp.addEventListener('click', () => compareLevels(resultEl.querySelector('.route-levels')));
      if (map._toolsApi) map._toolsApi.show('Rota');
      resultEl.closest('.airspace-control').scrollIntoView({ block: 'nearest' });
      setStatus('');
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar route-control');
      box.innerHTML = '<strong>Rota</strong>';
      const form = L.DomUtil.create('div', 'route-form', box);
      const list = L.DomUtil.create('datalist', '', box);
      list.id = 'routeAirports';
      const mk = (placeholder) => {
        const input = L.DomUtil.create('input', 'airspace-select', form);
        input.placeholder = placeholder;
        input.setAttribute('list', 'routeAirports');
        return input;
      };
      const fromInput = mk('Kalkış: LTFM veya 41.2,28.7');
      const toInput = mk('Varış: LTAC veya 39.9,32.9');
      const altInput = L.DomUtil.create('input', 'airspace-select route-alt', form);
      altInput.placeholder = 'İrtifa: FL100, 8000 ft, 2500 m';
      const pickBtn = (text, input) => {
        const btn = L.DomUtil.create('button', 'airspace-play route-go', form);
        btn.type = 'button';
        btn.textContent = text;
        btn.addEventListener('click', () => {
          map._routePicking = true;
          setStatus('Haritada bir noktaya tıklayın…');
          map.once('click', (e) => {
            input.value = e.latlng.lat.toFixed(3) + ',' + e.latlng.lng.toFixed(3);
            setStatus('');
            setTimeout(() => { map._routePicking = false; }, 0);
          });
        });
      };
      pickBtn('Kalkış ⌖ haritadan', fromInput);
      pickBtn('Varış ⌖ haritadan', toInput);
      const go = L.DomUtil.create('button', 'airspace-play route-go', form);
      go.type = 'button';
      go.textContent = 'Rotayı analiz et';
      const reset = L.DomUtil.create('button', 'airspace-play route-go', form);
      reset.type = 'button';
      reset.textContent = 'Temizle';
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      resultEl = L.DomUtil.create('div', 'route-result', box);

      getJson('/api/v1/map-layers/airports').then((data) => {
        airports = data.airports;
        list.innerHTML = airports.map((a) => '<option value="' + esc(a.icao) + '">' + esc(a.name) + '</option>').join('');
      }).catch(() => setStatus('Havalimanı listesi alınamadı'));

      go.addEventListener('click', async () => {
        const from = find(fromInput.value);
        const to = find(toInput.value);
        const alt = parseAltitude(altInput.value);
        if (!from || !to || (from.lat === to.lat && from.lon === to.lon)) {
          setStatus('Kalkış ve varış için farklı ICAO kodu veya "enlem,boylam" girin.');
          return;
        }
        if (alt === undefined) {
          setStatus('İrtifa: FL100, 8000 ft veya 2500 m biçiminde girin.');
          return;
        }
        try {
          await analyse(from, to, alt);
        } catch (err) {
          setStatus(err.message);
        }
      });
      reset.addEventListener('click', clear);
      map._routeApi = {
        path: () => lastPath,
        get: () => ({ from: fromInput.value, to: toInput.value, alt: altInput.value }),
        set(v) {
          fromInput.value = v.from || '';
          toInput.value = v.to || '';
          altInput.value = v.alt || '';
          if (fromInput.value && toInput.value) go.click();
        },
      };
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    control.addTo(map);
  }

  window.RouteGeometry = { samplePath, inFeature, inRing, distanceKm };
  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
