(function () {
  const API_URL = 'https://api.rainviewer.com/public/weather-maps.json';
  const REFRESH_MS = 5 * 60 * 1000;

  const FRAME_MS = 700;

  function attach(map) {
    let layers = [];
    let frames = [];
    let host = '';
    let index = 0;
    let playing = false;
    let refreshTimer = null;
    let animTimer = null;
    let statusEl = null;
    let slider = null;
    let playBtn = null;

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    }

    function clearLayers() {
      layers.forEach((l) => l && map.removeLayer(l));
      layers = [];
    }

    function frameLayer(i) {
      if (!layers[i]) {
        layers[i] = L.tileLayer(host + frames[i].path + '/256/{z}/{x}/{y}/2/1_1.png', {
          opacity: 0,
          maxNativeZoom: 7,
          maxZoom: 18,
          attribution: 'Radar &copy; <a href="https://www.rainviewer.com">RainViewer</a>',
        }).addTo(map);
      }
      return layers[i];
    }

    function showFrame(i) {
      index = i;
      layers.forEach((l, n) => l && l.setOpacity(n === i ? 0.6 : 0));
      frameLayer(i).setOpacity(0.6);
      slider.value = i;
      const f = frames[i];
      const time = new Date(f.time * 1000).toISOString().slice(11, 16);
      setStatus(time + ' UTC · ' + (f.forecast ? 'tahmin' : 'gözlem'));
    }

    function stopAnimation() {
      playing = false;
      clearInterval(animTimer);
      animTimer = null;
      if (playBtn) playBtn.textContent = '▶';
    }

    function startAnimation() {
      playing = true;
      playBtn.textContent = '⏸';
      animTimer = setInterval(() => showFrame((index + 1) % frames.length), FRAME_MS);
    }

    async function refresh() {
      try {
        const res = await fetch(API_URL);
        if (!res.ok) throw new Error('Radar verisi alınamadı');
        const data = await res.json();
        const past = (data.radar && data.radar.past) || [];
        const nowcast = ((data.radar && data.radar.nowcast) || []).map((f) => ({ ...f, forecast: true }));
        if (!past.length) throw new Error('Radar karesi yok');
        const wasLast = !frames.length || index >= past.length - 1;
        clearLayers();
        host = data.host;
        frames = past.concat(nowcast);
        slider.max = frames.length - 1;
        showFrame(wasLast ? past.length - 1 : Math.min(index, frames.length - 1));
      } catch (err) {
        setStatus(err.message);
      }
    }

    function stop() {
      stopAnimation();
      clearInterval(refreshTimer);
      refreshTimer = null;
      clearLayers();
      frames = [];
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
      const player = L.DomUtil.create('div', 'airspace-row', box);
      playBtn = L.DomUtil.create('button', 'airspace-play', player);
      playBtn.type = 'button';
      playBtn.textContent = '▶';
      slider = L.DomUtil.create('input', 'airspace-slider', player);
      slider.type = 'range';
      slider.min = 0;
      slider.max = 0;
      slider.value = 0;
      playBtn.addEventListener('click', () => {
        if (!frames.length) return;
        if (playing) stopAnimation(); else startAnimation();
      });
      slider.addEventListener('input', () => {
        stopAnimation();
        showFrame(Number(slider.value));
      });
      statusEl = L.DomUtil.create('div', 'airspace-status', box);
      input.addEventListener('change', () => {
        if (input.checked) {
          refresh();
          refreshTimer = setInterval(refresh, REFRESH_MS);
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
