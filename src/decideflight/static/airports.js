(function () {
  function esc(value) {
    const el = document.createElement('div');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
  }

  function observationHtml(airport, data) {
    const part = (title, text, timing) =>
      '<p style="font-weight:bold;margin:6px 0 2px">' + title + '</p>' +
      '<p style="font-family:monospace;word-break:break-word">' + esc(text || 'Bulunamadı') + '</p>' +
      (timing && timing.message ? '<p style="opacity:.75">' + esc(timing.message) + '</p>' : '');
    return '<div class="map-report-popup airspace-popup" style="max-width:260px">' +
      '<p style="font-weight:bold">' + esc(airport.icao) + ' · ' + esc(airport.name) + '</p>' +
      part('METAR', data.metar, data.metar_timing) +
      part('TAF', data.taf, data.taf_timing) +
      '<p style="opacity:.75;margin-top:6px">Kaynak: MGM. Bilgilendirme amaçlıdır.</p></div>';
  }

  function attach(map) {
    const group = L.layerGroup();
    let loaded = false;
    let statusEl = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    async function load() {
      setStatus('Yükleniyor…');
      try {
        const res = await fetch('/api/v1/map-layers/airports');
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Havalimanları alınamadı');
        data.airports.forEach((airport) => {
          const marker = L.circleMarker([airport.lat, airport.lon], {
            radius: 5, color: '#14181d', weight: 1, fillColor: '#edf0f3', fillOpacity: 0.95,
            bubblingMouseEvents: false,
          });
          marker.bindTooltip(airport.icao);
          marker.on('click', async () => {
            const popup = L.popup({ maxWidth: 280 })
              .setLatLng([airport.lat, airport.lon])
              .setContent('<span class="map-report-popup">Yükleniyor…</span>')
              .openOn(map);
            try {
              const response = await fetch('/api/v1/weather/airport-observations/' + encodeURIComponent(airport.icao));
              const obs = await response.json();
              if (!response.ok) throw new Error(obs.detail || 'METAR/TAF alınamadı');
              popup.setContent(observationHtml(airport, obs));
            } catch (err) {
              popup.setContent('<span class="text-red-300">' + esc(err.message) + '</span>');
            }
          });
          marker.addTo(group);
        });
        loaded = true;
        setStatus(data.airports.length + ' havalimanı');
        return true;
      } catch (err) {
        setStatus(err.message);
        return false;
      }
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar');
      box.innerHTML = '<strong>Havalimanları</strong>';
      const row = L.DomUtil.create('label', 'airspace-row', box);
      const input = L.DomUtil.create('input', '', row);
      input.type = 'checkbox';
      row.appendChild(document.createTextNode(' METAR / TAF'));
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      input.addEventListener('change', async () => {
        if (!input.checked) {
          map.removeLayer(group);
          return;
        }
        if (!loaded && !(await load())) {
          input.checked = false;
          return;
        }
        group.addTo(map);
      });
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    control.addTo(map);
  }

  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
