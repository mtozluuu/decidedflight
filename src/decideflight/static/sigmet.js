(function () {
  const HAZARDS = { TS: 'Gök gürültülü fırtına', TURB: 'Türbülans', ICE: 'Buzlanma', VA: 'Volkanik kül', MTW: 'Dağ dalgası', TC: 'Tropik siklon', DS: 'Toz fırtınası', SS: 'Kum fırtınası', GR: 'Dolu', RDOACT: 'Radyoaktif bulut' };

  function esc(value) {
    const el = document.createElement('div');
    el.textContent = value == null ? '' : String(value);
    return el.innerHTML;
  }

  function utc(sec) {
    return sec ? new Date(sec * 1000).toISOString().slice(5, 16).replace('T', ' ') + 'Z' : '—';
  }

  function popupHtml(s) {
    const name = HAZARDS[s.hazard] || s.hazard || 'SIGMET';
    return '<div class="map-report-popup airspace-popup" style="max-width:260px">' +
      '<p style="font-weight:bold">' + esc(name) + (s.qualifier ? ' · ' + esc(s.qualifier) : '') + '</p>' +
      '<p>' + esc(s.fir) + '</p>' +
      '<p>Taban/tavan: ' + esc(s.base_ft == null ? '—' : s.base_ft) + ' / ' + esc(s.top_ft == null ? '—' : s.top_ft) + ' ft</p>' +
      '<p>Geçerlilik: ' + utc(s.valid_from) + ' – ' + utc(s.valid_to) + '</p>' +
      '<p style="font-family:monospace;font-size:11px;white-space:pre-wrap;margin-top:6px">' + esc(s.raw) + '</p></div>';
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
        const res = await fetch('/api/v1/map-layers/sigmet');
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'SIGMET alınamadı');
        group.clearLayers();
        data.sigmets.forEach((s) => {
          const style = { color: '#ff9f6b', weight: 2.5, fillOpacity: 0.2, bubblingMouseEvents: false };
          const shape = s.coords.length > 2 ? L.polygon(s.coords, style) : L.polyline(s.coords, style);
          shape.bindPopup(popupHtml(s), { maxHeight: 260 });
          shape.bindTooltip((HAZARDS[s.hazard] || s.hazard || 'SIGMET') + ' · ' + (s.fir || ''), { sticky: true });
          shape.addTo(group);
        });
        loaded = true;
        setStatus(data.sigmets.length ? data.sigmets.length + ' aktif SIGMET (NOAA AWC)' : 'Bölgede aktif SIGMET yok');
        return true;
      } catch (err) {
        setStatus(err.message);
        return false;
      }
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar');
      box.innerHTML = '<strong>Uyarılar</strong>';
      const row = L.DomUtil.create('label', 'airspace-row', box);
      const input = L.DomUtil.create('input', '', row);
      input.type = 'checkbox';
      row.appendChild(document.createTextNode(' SIGMET'));
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      input.addEventListener('change', async () => {
        if (!input.checked) {
          map.removeLayer(group);
          return;
        }
        if (!(await load())) {
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
