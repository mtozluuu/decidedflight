(function () {
  const API_URL = 'https://api.rainviewer.com/public/weather-maps.json';
  const REFRESH_MS = 5 * 60 * 1000;

  function attach(map) {
    let layer = null;
    let timer = null;
    let statusEl = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    async function refresh() {
      try {
        const res = await fetch(API_URL);
        if (!res.ok) throw new Error('Radar verisi alınamadı');
        const data = await res.json();
        const frames = (data.radar && data.radar.past) || [];
        const latest = frames[frames.length - 1];
        if (!latest) throw new Error('Radar karesi yok');
        const next = L.tileLayer(data.host + latest.path + '/256/{z}/{x}/{y}/2/1_1.png', {
          opacity: 0.6,
          maxNativeZoom: 7,
          maxZoom: 18,
          attribution: 'Radar &copy; <a href="https://www.rainviewer.com">RainViewer</a>',
        });
        next.addTo(map);
        if (layer) map.removeLayer(layer);
        layer = next;
        const time = new Date(latest.time * 1000).toISOString().slice(11, 16);
        setStatus('Son kare ' + time + ' UTC');
      } catch (err) {
        setStatus(err.message);
      }
    }

    function stop() {
      clearInterval(timer);
      timer = null;
      if (layer) map.removeLayer(layer);
      layer = null;
      setStatus('');
    }

    const control = L.control({ position: 'topright' });
    control.onAdd = function () {
      const box = L.DomUtil.create('div', 'airspace-control leaflet-bar');
      box.innerHTML = '<strong>Meteoroloji</strong>';
      const row = L.DomUtil.create('label', 'airspace-row', box);
      const input = L.DomUtil.create('input', '', row);
      input.type = 'checkbox';
      row.appendChild(document.createTextNode(' Yağış radarı'));
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      input.addEventListener('change', () => {
        if (input.checked) {
          refresh();
          timer = setInterval(refresh, REFRESH_MS);
        } else {
          stop();
        }
      });
      L.DomEvent.disableClickPropagation(box);
      return box;
    };
    control.addTo(map);
  }

  (window.MapLayerPlugins = window.MapLayerPlugins || []).push(attach);
})();
