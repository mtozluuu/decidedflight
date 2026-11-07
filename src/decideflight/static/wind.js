(function () {
  const LEVELS = [
    ['surface', 'Yüzey (10 m)'],
    ['850', '850 hPa (~FL050)'],
    ['700', '700 hPa (~FL100)'],
    ['500', '500 hPa (~FL180)'],
  ];
  const CELL = 0.75;

  function cloudColor(c) {
    const mix = (a, b) => Math.round(a + (b - a) * c).toString(16).padStart(2, '0');
    return '#' + mix(0xc9, 0x1d) + mix(0xd8, 0x3b) + mix(0xe8, 0x6b);
  }

  function lerp(a, b, t) { return a + (b - a) * Math.min(Math.max(t, 0), 1); }
  function windColor(knots) {
    const t = knots / 40;
    const hex = [lerp(0x93, 0xe5, t), lerp(0xa9, 0xa5, t), lerp(0xc1, 0xa5, t)]
      .map((v) => Math.round(v).toString(16).padStart(2, '0'));
    return '#' + hex.join('');
  }

  function arrowIcon(point) {
    const toward = (point.wind_direction_deg + 180) % 360;
    const color = windColor(point.wind_speed_knots);
    const size = 18 + Math.min(point.wind_speed_knots, 40) / 2;
    const html = '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" ' +
      'style="transform:rotate(' + toward + 'deg)"><path d="M12 2 L18 20 L12 16 L6 20 Z" ' +
      'fill="' + color + '" stroke="#14181d" stroke-width="1"/></svg>';
    return L.divIcon({ html, className: 'wind-arrow-icon', iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
  }

  function attach(map) {
    const group = L.layerGroup();
    let showWind = false;
    let showCloud = false;
    let level = 'surface';
    let statusEl = null;
    let requestId = 0;

    let hour = 0;
    let levelSelect = null;

    function setStatus(text, retry) {
      MapUtil.setStatus(statusEl, text, retry);
    }

    async function redraw() {
      group.clearLayers();
      if (!showWind && !showCloud) {
        map.removeLayer(group);
        setStatus('');
        return;
      }
      const current = ++requestId;
      setStatus('Yükleniyor…');
      try {
        const res = await fetch('/api/v1/map-layers/wind-cloud?level=' + encodeURIComponent(level) + (hour ? '&hour=' + hour : ''));
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Veri alınamadı');
        if (current !== requestId) return;
        data.points.forEach((p) => {
          if (showCloud && p.cloud_cover_pct != null) {
            const c = p.cloud_cover_pct / 100;
            L.rectangle([[p.lat - CELL, p.lon - CELL], [p.lat + CELL, p.lon + CELL]], {
              stroke: false,
              interactive: false,
              fillColor: cloudColor(c),
              fillOpacity: 0.12 + 0.6 * c,
            }).addTo(group);
            L.marker([p.lat, p.lon], {
              interactive: false,
              keyboard: false,
              icon: L.divIcon({
                className: 'cloud-label',
                html: Math.round(p.cloud_cover_pct) + '%',
                iconSize: [36, 14],
                iconAnchor: [18, 7],
              }),
            }).addTo(group);
          }
          if (showWind && p.wind_speed_knots != null && p.wind_direction_deg != null) {
            const marker = L.marker([p.lat, p.lon], { icon: arrowIcon(p), keyboard: false });
            marker.bindTooltip(
              Math.round(p.wind_speed_knots) + ' kt · ' + Math.round(p.wind_direction_deg) + '°' +
              (p.cloud_cover_pct != null ? ' · bulut %' + Math.round(p.cloud_cover_pct) : '')
            );
            marker.addTo(group);
          }
        });
        group.addTo(map);
        setStatus(data.label + (hour ? ' · +' + hour + ' sa' : '') + ' · ' + String(data.valid_time_utc).replace('T', ' ') + ' UTC · model tahmini');
      } catch (err) {
        setStatus(err.message, redraw);
      }
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar');
      box.innerHTML = '<strong>Rüzgâr ve bulut</strong>';
      [['Rüzgâr okları', (v) => { showWind = v; }], ['Bulut örtüsü', (v) => { showCloud = v; }]]
        .forEach(([label, setter]) => {
          const row = L.DomUtil.create('label', 'airspace-row', box);
          const input = L.DomUtil.create('input', '', row);
          input.type = 'checkbox';
          row.appendChild(document.createTextNode(' ' + label));
          input.addEventListener('change', () => { setter(input.checked); redraw(); });
        });
      const select = L.DomUtil.create('select', 'airspace-select', box);
      select.setAttribute('aria-label', 'Basınç seviyesi');
      LEVELS.forEach(([value, text]) => {
        const option = L.DomUtil.create('option', '', select);
        option.value = value;
        option.textContent = text;
      });
      select.addEventListener('change', () => { level = select.value; redraw(); });
      levelSelect = select;
      map._windApi = {
        setLevel(value) {
          if (value === level) return;
          level = value;
          select.value = value;
          redraw();
        },
        setHour(value) {
          if (value === hour) return;
          hour = value;
          redraw();
        },
      };
      const legend = L.DomUtil.create('div', 'cloud-legend', box);
      legend.innerHTML = '<span>Bulut %0</span><i></i><span>%100</span>';
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    control.addTo(map);
  }

  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
