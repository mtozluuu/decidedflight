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

  function attach(map) {
    if (!map || map._airspaceAttached) return;
    map._airspaceAttached = true;
    const layers = {};
    const loaded = {};
    let statusEl = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    async function show(code) {
      if (!layers[code]) {
        layers[code] = L.geoJSON(null, {
          style: () => STYLES[code],
          bubblingMouseEvents: false,
          onEachFeature: (feature, layer) => {
            layer.bindPopup(popupHtml(feature.properties));
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
          layers[code].addData(data);
          loaded[code] = true;
          setStatus('openAIP · ' + String(data.fetched_at).slice(0, 10) + ' · doğrulama gerekir');
        } catch (err) {
          setStatus(err.message);
          return false;
        }
      }
      layers[code].addTo(map);
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
        input.checked = on;
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
        if (on) show(code);
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
      const toggle = () => panel.classList.toggle('collapsed');
      title.addEventListener('click', toggle);
      title.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
      if (index > 0) panel.classList.add('collapsed');
    });
  }

  window.AirspaceLayer = { attach };
})();
