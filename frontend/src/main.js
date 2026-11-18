/**
 * PathSnap Frontend Application Logic
 * Supports cross-machine distributed deployment (Mac Backend + Windows Frontend).
 */

// Initialize or restore Backend API URL
const defaultUrl = import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
let apiBaseUrl = localStorage.getItem('pathsnap_api_url') || defaultUrl;

// DOM Elements
const backendUrlInput = document.getElementById('backendUrlInput');
const reconnectBtn = document.getElementById('reconnectBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const pingBadge = document.getElementById('pingBadge');

const tripSelect = document.getElementById('tripSelect');
const metaPartition = document.getElementById('metaPartition');
const metaPoints = document.getElementById('metaPoints');
const metaGT = document.getElementById('metaGT');

const modelSelect = document.getElementById('modelSelect');
const calibrationCheck = document.getElementById('calibrationCheck');
const paramBeta = document.getElementById('paramBeta');
const paramUnreachable = document.getElementById('paramUnreachable');
const paramTheta0 = document.getElementById('paramTheta0');
const paramEmissionWeight = document.getElementById('paramEmissionWeight');
const runMatchBtn = document.getElementById('runMatchBtn');

const kpiHmmAcc = document.getElementById('kpiHmmAcc');
const kpiAccDelta = document.getElementById('kpiAccDelta');
const kpiContinuity = document.getElementById('kpiContinuity');
const kpiClfAcc = document.getElementById('kpiClfAcc');
const kpiLatency = document.getElementById('kpiLatency');
const kpiResets = document.getElementById('kpiResets');

const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMsg = document.getElementById('loadingMsg');

const viewFoliumBtn = document.getElementById('viewFoliumBtn');
const viewBenchmarkBtn = document.getElementById('viewBenchmarkBtn');
const benchmarkModal = document.getElementById('benchmarkModal');
const closeBenchmarkBtn = document.getElementById('closeBenchmarkBtn');
const benchmarkTableBody = document.getElementById('benchmarkTableBody');

const inspectorPanel = document.getElementById('inspectorPanel');
const toggleInspectorBtn = document.getElementById('toggleInspectorBtn');
const obsTableBody = document.getElementById('obsTableBody');
const inspectorCount = document.getElementById('inspectorCount');

// Leaflet Map State
let map = null;
let layers = {
  gps: null,
  predictedRoute: null,
  referenceRoute: null,
};

let currentTripData = null;
let tripsCatalog = [];

// Set initial backend input
backendUrlInput.value = apiBaseUrl;

// 1. Initialize Map
function initMap() {
  map = L.map('map', {
    center: [30.2741, 120.1551],
    zoom: 13,
    zoomControl: true,
  });

  // Base tile layer: CartoDB Dark Matter / Positron
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  layers.gps = L.layerGroup().addTo(map);
  layers.predictedRoute = L.layerGroup().addTo(map);
  layers.referenceRoute = L.layerGroup().addTo(map);
}

// 2. Health & Connectivity Monitoring
async function checkBackendHealth() {
  const t0 = performance.now();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const res = await fetch(`${apiBaseUrl}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);

    const rtt = Math.round(performance.now() - t0);

    if (res.ok) {
      const data = await res.json();
      statusDot.className = 'status-indicator online';
      const host = data.platform?.hostname || 'Backend';
      const ip = data.network?.lan_ip || '';
      statusText.textContent = `Online: ${host} (${ip})`;
      pingBadge.textContent = `${rtt} ms`;
      return true;
    } else {
      throw new Error(`Status ${res.status}`);
    }
  } catch (err) {
    statusDot.className = 'status-indicator offline';
    statusText.textContent = 'Backend Offline';
    pingBadge.textContent = '-- ms';
    return false;
  }
}

// 3. Load Trips Catalog
async function loadTrips() {
  try {
    const res = await fetch(`${apiBaseUrl}/api/trips`);
    if (!res.ok) throw new Error('Failed to fetch trips');
    const data = await res.json();
    tripsCatalog = data.trips;

    tripSelect.innerHTML = '';

    const groupTest = document.createElement('optgroup');
    groupTest.label = 'Student Test Partition (10 Trajectories)';

    const groupTrain = document.createElement('optgroup');
    groupTrain.label = 'Train Partition (80 Trajectories)';

    tripsCatalog.forEach((t) => {
      const opt = document.createElement('option');
      opt.value = t.trip_id;
      opt.dataset.partition = t.partition;
      opt.textContent = `Trip ${t.trip_id} (${t.num_observations} pts${t.has_ground_truth ? ', with GT' : ''})`;

      if (t.partition === 'student_test') {
        groupTest.appendChild(opt);
      } else {
        groupTrain.appendChild(opt);
      }
    });

    tripSelect.appendChild(groupTest);
    tripSelect.appendChild(groupTrain);

    if (tripsCatalog.length > 0) {
      // Default to student test trip 00000000 or first available
      tripSelect.value = tripsCatalog[0].trip_id;
      onTripSelected();
    }
  } catch (err) {
    console.error('Error loading trips catalog:', err);
  }
}

// 4. Handle Trip Selection
async function onTripSelected() {
  const tid = tripSelect.value;
  if (!tid) return;

  const tripMeta = tripsCatalog.find((t) => t.trip_id === tid);
  const partition = tripMeta ? tripMeta.partition : 'student_test';

  metaPartition.textContent = `Partition: ${partition}`;
  metaPoints.textContent = `Observations: ${tripMeta ? tripMeta.num_observations : '--'}`;
  metaGT.textContent = `Ground Truth: ${tripMeta && tripMeta.has_ground_truth ? 'Available' : 'Unlabelled'}`;

  // Reset KPI cards
  kpiHmmAcc.textContent = '--%';
  kpiAccDelta.textContent = 'vs standalone classifier';
  kpiContinuity.textContent = '--%';
  kpiClfAcc.textContent = '--%';
  kpiLatency.textContent = '-- ms';
  kpiResets.textContent = '0 resets';

  // Clear map layers
  layers.gps.clearLayers();
  layers.predictedRoute.clearLayers();
  layers.referenceRoute.clearLayers();

  // Clear table
  obsTableBody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">Click "Run HMM Map Matching" to decode trip ${tid}.</td></tr>`;
  inspectorCount.textContent = '0';

  // Fetch track points
  showLoading(`Loading GPS track for trip ${tid}...`);
  try {
    const res = await fetch(`${apiBaseUrl}/api/trips/${tid}?partition=${partition}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    currentTripData = await res.json();

    // Plot raw GPS trajectory
    const latlngs = [];
    currentTripData.track.forEach((pt) => {
      const ll = [pt.latitude, pt.longitude];
      latlngs.push(ll);

      const marker = L.circleMarker(ll, {
        radius: 4,
        fillColor: '#38bdf8',
        color: '#0284c7',
        weight: 1,
        opacity: 0.9,
        fillOpacity: 0.7,
      });

      marker.bindPopup(`
        <strong>Obs #${pt.observation_id}</strong><br>
        Lat: ${pt.latitude.toFixed(6)}<br>
        Lon: ${pt.longitude.toFixed(6)}<br>
        Time: ${pt.timestamp}s
      `);

      layers.gps.addLayer(marker);
    });

    // Connecting dashed line
    if (latlngs.length > 1) {
      const poly = L.polyline(latlngs, {
        color: '#38bdf8',
        weight: 1.5,
        dashArray: '4, 4',
        opacity: 0.5,
      });
      layers.gps.addLayer(poly);
    }

    // Zoom map to trajectory bounds
    if (currentTripData.bounds) {
      const b = currentTripData.bounds;
      map.fitBounds([
        [b.min_lat, b.min_lon],
        [b.max_lat, b.max_lon],
      ], { padding: [40, 40] });
    }
  } catch (err) {
    alert(`Failed to load trip: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// 5. Run HMM Map Matching
async function runHMMMapMatching() {
  const tid = tripSelect.value;
  if (!tid) return;

  const tripMeta = tripsCatalog.find((t) => t.trip_id === tid);
  const partition = tripMeta ? tripMeta.partition : 'student_test';

  const payload = {
    trip_id: tid,
    partition: partition,
    model_name: modelSelect.value,
    use_calibration: calibrationCheck.checked,
    hmm_params: {
      beta: parseFloat(paramBeta.value) || 20.0,
      unreachable_penalty: parseFloat(paramUnreachable.value) || 5.0,
      theta_0: parseFloat(paramTheta0.value) || 55.0,
      emission_weight: parseFloat(paramEmissionWeight.value) || 1.5,
    },
  };

  showLoading(`Running HMM Viterbi decoding for trip ${tid}...`);

  try {
    const res = await fetch(`${apiBaseUrl}/api/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const matchData = await res.json();

    // 1. Update KPI Cards
    if (matchData.metrics.observation_top1_accuracy !== null) {
      const hmmAcc = (matchData.metrics.observation_top1_accuracy * 100).toFixed(1);
      kpiHmmAcc.textContent = `${hmmAcc}%`;

      if (matchData.metrics.classifier_top1_accuracy !== null) {
        const clfAcc = (matchData.metrics.classifier_top1_accuracy * 100).toFixed(1);
        kpiClfAcc.textContent = `${clfAcc}%`;

        const delta = ((matchData.metrics.observation_top1_accuracy - matchData.metrics.classifier_top1_accuracy) * 100).toFixed(1);
        kpiAccDelta.textContent = `+${delta}% improvement`;
      }
    } else {
      kpiHmmAcc.textContent = 'Unlabelled';
      kpiAccDelta.textContent = 'Student-Test Blind Partition';
      kpiClfAcc.textContent = 'N/A';
    }

    const cont = (matchData.metrics.route_continuity * 100).toFixed(1);
    kpiContinuity.textContent = `${cont}%`;
    kpiLatency.textContent = `${matchData.execution_time_ms} ms`;
    kpiResets.textContent = `${matchData.metrics.num_resets} resets`;

    // 2. Render GeoJSON Layers on Map
    layers.predictedRoute.clearLayers();
    layers.referenceRoute.clearLayers();

    if (matchData.geojson && matchData.geojson.features) {
      matchData.geojson.features.forEach((feat) => {
        if (feat.geometry.type === 'LineString') {
          // Leaflet expects [lat, lon], GeoJSON is [lon, lat]
          const coords = feat.geometry.coordinates.map((c) => [c[1], c[0]]);
          const isPred = feat.properties.layer === 'predicted_route';

          const poly = L.polyline(coords, {
            color: feat.properties.color || (isPred ? '#9400D3' : '#22c55e'),
            weight: isPred ? 4.5 : 2.5,
            opacity: isPred ? 0.9 : 0.75,
          });

          poly.bindTooltip(`${isPred ? 'Predicted HMM Arc' : 'Ground Truth Arc'} #${feat.properties.arc_id}`);

          if (isPred) {
            layers.predictedRoute.addLayer(poly);
          } else {
            layers.referenceRoute.addLayer(poly);
          }
        }
      });
    }

    // 3. Populate Observation Sequence Table
    inspectorCount.textContent = matchData.predictions.length;
    obsTableBody.innerHTML = '';

    matchData.predictions.forEach((p) => {
      const tr = document.createElement('tr');
      const isCorrect = p.is_correct;
      const statusHtml = isCorrect === null
        ? '<span class="status-tag">Decoded</span>'
        : isCorrect
        ? '<span class="status-tag status-correct">✓ Match</span>'
        : '<span class="status-tag status-wrong">✗ Mismatch</span>';

      tr.innerHTML = `
        <td>${p.observation_id}</td>
        <td>${p.raw_lat.toFixed(5)}, ${p.raw_lon.toFixed(5)}</td>
        <td><strong>${p.predicted_arc_id}</strong></td>
        <td>${p.classifier_arc_id}</td>
        <td>${p.true_arc_id !== null ? p.true_arc_id : '--'}</td>
        <td>${statusHtml}</td>
      `;
      obsTableBody.appendChild(tr);
    });

  } catch (err) {
    alert(`Map matching error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

// 6. Section 5A Superiority Benchmark Modal
async function openBenchmarkModal() {
  try {
    const res = await fetch(`${apiBaseUrl}/api/metrics/summary`);
    if (!res.ok) throw new Error('Failed to fetch benchmark summary');
    const data = await res.json();

    benchmarkTableBody.innerHTML = '';
    data.models_comparison.forEach((m) => {
      const tr = document.createElement('tr');
      const isRefinedHMM = m.method.includes('Refined');
      if (isRefinedHMM) tr.className = 'highlight-row';

      tr.innerHTML = `
        <td>${m.method}</td>
        <td>${(m.top1_accuracy * 100).toFixed(2)}%</td>
        <td>${(m.route_continuity * 100).toFixed(2)}%</td>
        <td>${(m.macro_f1 * 100).toFixed(2)}%</td>
        <td>${(m.per_trip_mean_acc * 100).toFixed(2)}%</td>
      `;
      benchmarkTableBody.appendChild(tr);
    });

    benchmarkModal.classList.remove('hidden');
  } catch (err) {
    alert(`Could not load benchmark summary: ${err.message}`);
  }
}

// Helpers
function showLoading(msg) {
  loadingMsg.textContent = msg;
  loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  loadingOverlay.classList.add('hidden');
}

// Event Listeners
reconnectBtn.addEventListener('click', async () => {
  const newUrl = backendUrlInput.value.trim().replace(/\/$/, '');
  if (!newUrl) return;
  apiBaseUrl = newUrl;
  localStorage.setItem('pathsnap_api_url', apiBaseUrl);
  const ok = await checkBackendHealth();
  if (ok) {
    await loadTrips();
  } else {
    alert(`Unable to connect to backend at ${apiBaseUrl}. Ensure the Mac backend server is running and port 8000 is reachable.`);
  }
});

tripSelect.addEventListener('change', onTripSelected);
runMatchBtn.addEventListener('click', runHMMMapMatching);

viewFoliumBtn.addEventListener('click', () => {
  const tid = tripSelect.value;
  if (!tid) return;
  const tripMeta = tripsCatalog.find((t) => t.trip_id === tid);
  const partition = tripMeta ? tripMeta.partition : 'student_test';
  window.open(`${apiBaseUrl}/api/trips/${tid}/map?partition=${partition}`, '_blank');
});

viewBenchmarkBtn.addEventListener('click', openBenchmarkModal);
closeBenchmarkBtn.addEventListener('click', () => benchmarkModal.classList.add('hidden'));

toggleInspectorBtn.addEventListener('click', () => {
  inspectorPanel.classList.toggle('collapsed');
  toggleInspectorBtn.textContent = inspectorPanel.classList.contains('collapsed') ? 'Expand ▼' : 'Collapse ▲';
});

// App Initialization
initMap();
checkBackendHealth().then((ok) => {
  if (ok) loadTrips();
});

// Periodic ping every 10 seconds
setInterval(checkBackendHealth, 10000);

