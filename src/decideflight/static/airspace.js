(function () {
  const STYLES = {
    TMA: { color: '#5aa9ff', fillOpacity: 0.16, weight: 2.5 },
    CTR: { color: '#4ade9a', fillOpacity: 0.26, weight: 2.5 },
    FIR: { color: '#ffffff', fillOpacity: 0, weight: 3, dashArray: '10 8' },
    R: { color: '#ff7a7a', fillOpacity: 0.16, weight: 2 },
    D: { color: '#ff9f6b', fillOpacity: 0.12, weight: 2, dashArray: '4 4' },
    P: { color: '#ff4d4d', fillOpacity: 0.3, weight: 3 },
  };
  const OPTIONS = [
    ['TMA', 'TMA', true],
    ['CTR', 'CTR', true],
    ['FIR', 'FIR', true],
    ['R', 'Kısıtlı (R)', false],
    ['D', 'Tehlikeli (D)', false],
    ['P', 'Yasak (P)', false],
  ];

  function esc(value) {
    const el = document.createElement('div');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
  }

  function popupHtml(p) {
    return '<div class="map-report-popup airspace-popup">' +
      '<p style="font-weight:bold;margin-bottom:5px">' + esc(p.name) + '</p>' +
      '<p>' + esc(p.category) + ' · ' + esc(p.category_label) + '</p>' +
      '<p>Üst limit: ' + esc(p.upper) + '</p>' +
      '<p>Alt limit: ' + esc(p.lower) + '</p>' +
      '</div>';
  }

  const PREF_KEY = 'decideflight.mapLayers';

  function readPrefs() {
    try { return JSON.parse(localStorage.getItem(PREF_KEY)) || {}; } catch (err) { return {}; }
  }

  function attach(map) {
    if (!map || map._airspaceAttached) return;
    map._airspaceAttached = true;
    const layers = {};
    const loaded = {};
    const features = {};
    let level = null;
    let statusEl = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function inRing(pt, ring) {
      let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const a = ring[i], b = ring[j];
        if ((a.lat > pt.lat) !== (b.lat > pt.lat) &&
            pt.lng < ((b.lng - a.lng) * (pt.lat - a.lat)) / (b.lat - a.lat) + a.lng) inside = !inside;
      }
      return inside;
    }

    function contains(layer, pt) {
      const polys = layer.getLatLngs();
      const list = Array.isArray(polys[0]) && Array.isArray(polys[0][0]) && polys[0][0].lat === undefined ? polys : [polys];
      return list.some((rings) => inRing(pt, rings[0]) && !rings.slice(1).some((h) => inRing(pt, h)));
    }

    let searchIndex = null;

    function fold(text) {
      return String(text || '').toLocaleLowerCase('tr').replace(/ı/g, 'i').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    }

    async function findPlace(query) {
      const q = fold(query).trim();
      if (!q) return false;
      if (!searchIndex) {
        try {
          const [asp, apt] = await Promise.all([
            fetch('/api/v1/airspace?categories=TMA,CTR,FIR').then((r) => r.json()),
            fetch('/api/v1/map-layers/airports').then((r) => r.json()),
          ]);
          searchIndex = { airspaces: asp.features || [], airports: apt.airports || [] };
        } catch (err) {
          return false;
        }
      }
      const airport = searchIndex.airports.find((a) => fold(a.icao) === q) ||
        searchIndex.airports.find((a) => fold(a.name).includes(q) || fold(a.icao).includes(q));
      if (airport) {
        map.setView([airport.lat, airport.lon], 10);
        return true;
      }
      const hit = searchIndex.airspaces.find((f) => fold(f.properties.name).includes(q));
      if (!hit) return false;
      const bounds = L.geoJSON(hit).getBounds();
      map.fitBounds(bounds, { maxZoom: 10 });
      L.popup({ maxHeight: 260 }).setLatLng(bounds.getCenter()).setContent(popupHtml(hit.properties)).openOn(map);
      return true;
    }

    function visible(list) {
      if (level == null) return list;
      const ft = level * 100;
      return list.filter((f) => {
        const lo = f.properties.lower_ft;
        const up = f.properties.upper_ft;
        return (lo == null || lo <= ft) && (up == null || up >= ft);
      });
    }

    function refilter() {
      Object.keys(layers).forEach((code) => {
        if (!loaded[code]) return;
        layers[code].clearLayers();
        layers[code].addData(visible(features[code]));
      });
    }

    function openStack(latlng) {
      const order = ['CTR', 'TMA', 'FIR'];
      const hits = [];
      Object.keys(layers).sort((a, b) => order.indexOf(a) - order.indexOf(b)).forEach((code) => {
        if (!map.hasLayer(layers[code])) return;
        layers[code].eachLayer((l) => { if (contains(l, latlng)) hits.push(l.feature.properties); });
      });
      if (!hits.length) return;
      L.popup({ maxHeight: 260 }).setLatLng(latlng).setContent(hits.map(popupHtml).join('<hr>')).openOn(map);
    }

    async function show(code) {
      if (!layers[code]) {
        layers[code] = L.geoJSON(null, {
          style: () => STYLES[code],
          bubblingMouseEvents: false,
          onEachFeature: (feature, layer) => {
            layer.on('click', (e) => openStack(e.latlng));
            layer.bindTooltip(feature.properties.name, { sticky: true });
            layer.on('mouseover', () => layer.setStyle({ weight: STYLES[code].weight + 2, fillOpacity: STYLES[code].fillOpacity + 0.15 }));
            layer.on('mouseout', () => layer.setStyle(STYLES[code]));
          },
        });
      }
      if (!loaded[code]) {
        setStatus('Yükleniyor…');
        try {
          const res = await fetch('/api/v1/airspace?categories=' + encodeURIComponent(code));
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || 'Hava sahası alınamadı');
          features[code] = data.features;
          layers[code].addData(visible(data.features));
          loaded[code] = true;
          setStatus('openAIP · ' + String(data.fetched_at).slice(0, 10) + ' · doğrulama gerekir');
        } catch (err) {
          setStatus(err.message);
          return false;
        }
      }
      layers[code].addTo(map);
      if (code === 'FIR') layers[code].bringToBack();
      return true;
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar');
      box.innerHTML = '<strong>Hava sahası</strong>';
      OPTIONS.forEach(([code, label, on]) => {
        const row = L.DomUtil.create('label', 'airspace-row', box);
        const input = L.DomUtil.create('input', '', row);
        input.type = 'checkbox';
        input.checked = readPrefs()['Hava sahası|' + label] ?? on;
        const swatch = L.DomUtil.create('span', 'airspace-swatch', row);
        swatch.style.borderColor = STYLES[code].color;
        swatch.style.background = STYLES[code].fillOpacity ? STYLES[code].color + '55' : 'transparent';
        if (STYLES[code].dashArray) swatch.style.borderStyle = 'dashed';
        row.appendChild(document.createTextNode(label));
        input.addEventListener('change', async () => {
          if (input.checked) {
            if (!(await show(code))) input.checked = false;
          } else if (layers[code]) {
            map.removeLayer(layers[code]);
          }
        });
        if (input.checked) show(code);
      });
      const filterRow = L.DomUtil.create('label', 'airspace-row', box);
      const filterBox = L.DomUtil.create('input', '', filterRow);
      filterBox.type = 'checkbox';
      const filterText = document.createTextNode(' İrtifa süz: FL100');
      filterRow.appendChild(filterText);
      const slider = L.DomUtil.create('input', 'airspace-slider', box);
      slider.type = 'range';
      slider.min = 0;
      slider.max = 460;
      slider.step = 10;
      slider.value = 100;
      slider.disabled = true;
      const apply = () => {
        level = filterBox.checked ? Number(slider.value) : null;
        slider.disabled = !filterBox.checked;
        filterText.textContent = ' İrtifa süz: FL' + String(slider.value).padStart(3, '0');
        refilter();
      };
      filterBox.addEventListener('change', apply);
      slider.addEventListener('input', apply);
      const search = L.DomUtil.create('input', 'airspace-select', box);
      search.type = 'search';
      search.placeholder = 'Ara: LTAC, İzmir TMA';
      search.addEventListener('keydown', async (e) => {
        if (e.key !== 'Enter') return;
        const found = await findPlace(search.value);
        setStatus(found ? '' : 'Sonuç bulunamadı');
      });
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    control.addTo(map);
    (window.MapLayerPlugins || []).forEach((plugin) => plugin(map));

    const panels = map.getContainer().querySelectorAll('.airspace-control');
    panels.forEach((panel, index) => {
      const title = panel.querySelector('strong');
      if (!title) return;
      title.classList.add('airspace-title');
      title.setAttribute('role', 'button');
      title.setAttribute('tabindex', '0');
      const toggle = () => {
        const open = panel.classList.contains('collapsed');
        panels.forEach((other) => other.classList.add('collapsed'));
        if (open) {
          panel.classList.remove('collapsed');
          panel.scrollIntoView({ block: 'nearest' });
        }
      };
      title.addEventListener('click', toggle);
      title.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
      if (index > 0) panel.classList.add('collapsed');
    });

    const saved = readPrefs();
    const checkboxes = () => [...map.getContainer().querySelectorAll('.airspace-control')].flatMap((panel) => {
      const title = panel.querySelector('strong');
      return [...panel.querySelectorAll('label.airspace-row')].map((row) => ({
        key: (title ? title.textContent : '') + '|' + row.textContent.trim(),
        input: row.querySelector('input[type=checkbox]'),
      })).filter((entry) => entry.input);
    });
    checkboxes().forEach(({ key, input }) => {
      if (!key.startsWith('Hava sahası|') && saved[key] && !input.checked) {
        input.checked = true;
        input.dispatchEvent(new Event('change'));
      }
    });
    map.getContainer().addEventListener('change', () => {
      const state = {};
      checkboxes().forEach(({ key, input }) => { state[key] = input.checked; });
      try { localStorage.setItem(PREF_KEY, JSON.stringify(state)); } catch (err) { /* storage unavailable */ }
    });
  }

  window.AirspaceLayer = { attach };
})();
