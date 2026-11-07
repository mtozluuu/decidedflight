(function () {
  const STYLES = {
    TMA: { color: '#93a9c1', fillOpacity: 0.08, weight: 1.5 },
    CTR: { color: '#a6d7bd', fillOpacity: 0.12, weight: 1.5 },
    FIR: { color: '#a0aab6', fillOpacity: 0, weight: 1.5, dashArray: '6 6' },
    R: { color: '#e5a5a5', fillOpacity: 0.1, weight: 1.5 },
    D: { color: '#e5a5a5', fillOpacity: 0.06, weight: 1, dashArray: '3 4' },
    P: { color: '#e5a5a5', fillOpacity: 0.2, weight: 2 },
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
          onEachFeature: (feature, layer) => layer.bindPopup(popupHtml(feature.properties)),
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
        row.appendChild(document.createTextNode(' ' + label));
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
  }

  window.AirspaceLayer = { attach };
})();
