/*
  AI Disinfo Shield -- Frontend JavaScript Application Logic
  Handles Fake Tweet Verification, Self-Updating Engine Polling, Preset Chips, & Benchmark Graphs
*/

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const verifyForm = document.getElementById('verify-form');
    const btnSubmit = document.getElementById('btn-submit');
    const claimInput = document.getElementById('claim_text');
    const domainInput = document.getElementById('source_domain');
    const imageInput = document.getElementById('image_url');

    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    
    // Results Elements
    const verdictBadge = document.getElementById('verdict-badge');
    const verdictConfidence = document.getElementById('verdict-confidence');
    const verdictTimestamp = document.getElementById('verdict-timestamp');

    const probRealVal = document.getElementById('prob-real-val');
    const probFakeVal = document.getElementById('prob-fake-val');
    const probSuspiciousVal = document.getElementById('prob-suspicious-val');

    const probRealBar = document.getElementById('prob-real-bar');
    const probFakeBar = document.getElementById('prob-fake-bar');
    const probSuspiciousBar = document.getElementById('prob-suspicious-bar');

    const p1Metric = document.getElementById('p1-metric');
    const p2Metric = document.getElementById('p2-metric');
    const p3aMetric = document.getElementById('p3a-metric');
    const p3bMetric = document.getElementById('p3b-metric');
    const p4Metric = document.getElementById('p4-metric');
    const p5Metric = document.getElementById('p5-metric');
    const explanationText = document.getElementById('explanation-text');

    // Self-Updating Engine Elements
    const countdownBadge = document.getElementById('countdown-badge');
    const lastUpdateTime = document.getElementById('last-update-time');
    const btnTriggerUpdate = document.getElementById('btn-trigger-update');
    const intervalSelect = document.getElementById('update-interval-select');
    const logStreamContainer = document.getElementById('log-stream-container');

    // Graph Gallery Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const activeGraphImg = document.getElementById('active-graph-img');

    // Preset Tweets Database
    const PRESETS = {
        fake1: {
            text: "BREAKING: Leaked military documents PROVE secret underground base discovered under Kyiv city center! SHOCKING video deleted by Twitter! #BreakingNews @conspiracy_daily",
            domain: "twitter.com"
        },
        fake2: {
            text: "ALERT: Top doctor whistleblower exposes government secretly poisoning city water supply before being silenced! MUST WATCH before deleted! #Whistleblower",
            domain: "yournewswire.com"
        },
        real1: {
            text: "Reuters: U.S. Department of Defense announces new $400M security assistance package for Ukraine including air defense munitions and armored vehicles. @reuters",
            domain: "reuters.com"
        },
        real2: {
            text: "BBC News: World Health Organization publishes updated global health surveillance guidelines following expert panel review in Geneva. @bbcnews",
            domain: "bbc.com"
        }
    };

    // Preset Chip Click Handlers
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const key = chip.getAttribute('data-type');
            if (PRESETS[key]) {
                claimInput.value = PRESETS[key].text;
                domainInput.value = PRESETS[key].domain;
                verifyForm.dispatchEvent(new Event('submit'));
            }
        });
    });

    // Handle Tab Graph Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const graphKey = btn.getAttribute('data-graph');
            activeGraphImg.src = `/api/graphs/${graphKey}?t=${Date.now()}`;
        });
    });

    // Handle Verification Form Submission
    verifyForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const claimText = claimInput.value.trim();
        const sourceDomain = domainInput.value.trim();
        const imageUrl = imageInput.value.trim();

        if (!claimText) return;

        // Set Loading State
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = '<span>Processing 5-Phase Pipeline...</span>';

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    claim_text: claimText,
                    source_domain: sourceDomain || 'twitter.com',
                    image_url: imageUrl || null
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Verification request failed.');
            }

            const data = await response.json();
            renderResults(data);

        } catch (err) {
            alert(`Verification Error: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = '<span>Execute Verification Pipeline</span>';
        }
    });

    function renderResults(data) {
        resultsPlaceholder.classList.add('hidden');
        resultsContent.classList.remove('hidden');

        const verdict = data.verdict;
        const confidence = data.confidence_percentage;
        const probs = data.class_probabilities;
        const pb = data.phase_breakdown;

        // Update Verdict Badge & Colors
        verdictBadge.innerText = verdict;
        verdictBadge.className = 'verdict-badge-large';
        if (verdict === 'REAL') verdictBadge.classList.add('verdict-real');
        else if (verdict === 'FAKE') verdictBadge.classList.add('verdict-fake');
        else verdictBadge.classList.add('verdict-suspicious');

        verdictConfidence.innerText = `${confidence}%`;
        verdictTimestamp.innerText = `Evaluated at ${new Date().toLocaleTimeString()}`;

        // Update Class Probabilities
        probRealVal.innerText = (probs.REAL * 100).toFixed(1) + '%';
        probFakeVal.innerText = (probs.FAKE * 100).toFixed(1) + '%';
        probSuspiciousVal.innerText = (probs.SUSPICIOUS * 100).toFixed(1) + '%';

        probRealBar.style.width = (probs.REAL * 100) + '%';
        probFakeBar.style.width = (probs.FAKE * 100) + '%';
        probSuspiciousBar.style.width = (probs.SUSPICIOUS * 100) + '%';

        // Update Phase Grid Metrics
        p1Metric.innerText = `Tokens: ${pb.phase1_text_stats.tokens}`;
        p2Metric.innerText = `FAISS: ${pb.phase2_kg_retrieval.faiss_top_score.toFixed(2)}`;
        p3aMetric.innerText = pb.phase3a_sourcerank.rating.split('/')[0].trim();
        p3bMetric.innerText = pb.phase3b_deberta.is_factual_claim ? 'Factual Claim' : 'Opinion';
        p4Metric.innerText = `Sim: ${pb.phase4_multimodal.image_text_alignment.toFixed(2)}`;
        p5Metric.innerText = `${verdict} (${confidence}%)`;

        // Update Explanation
        explanationText.innerText = data.explanation;
    }

    // =========================================================================
    // SELF-UPDATING ENGINE POLLING & CONTROLS
    // =========================================================================
    async function fetchSelfUpdateStatus() {
        try {
            const resp = await fetch('/api/self-update/status');
            if (!resp.ok) return;
            const data = await resp.json();

            // Format Countdown Timer (MM:SS)
            const secsLeft = data.next_update_in_seconds;
            const mins = Math.floor(secsLeft / 60);
            const secs = secsLeft % 60;
            const formattedTime = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            countdownBadge.innerText = `Next Update in ${formattedTime}`;

            lastUpdateTime.innerText = `Last Updated: ${data.last_update} (Cycle #${data.total_updates_performed})`;

            // Render Log Stream
            if (data.history_logs && data.history_logs.length > 0) {
                logStreamContainer.innerHTML = '';
                data.history_logs.forEach(log => {
                    const div = document.createElement('div');
                    div.className = 'log-entry';
                    div.innerHTML = `<span class="log-time">[${log.timestamp}]</span><span class="log-msg">${log.event} | Triples: ${log.triples_count} | Status: ${log.status}</span>`;
                    logStreamContainer.appendChild(div);
                });
            }
        } catch (err) {
            console.warn('Failed to fetch self-update status:', err);
        }
    }

    // Poll self-updater status every 2 seconds
    setInterval(fetchSelfUpdateStatus, 2000);
    fetchSelfUpdateStatus();

    // Trigger Manual Self-Update Button
    btnTriggerUpdate.addEventListener('click', async () => {
        btnTriggerUpdate.disabled = true;
        btnTriggerUpdate.innerHTML = '<span>Executing Auto-Update...</span>';
        try {
            const resp = await fetch('/api/self-update/trigger', { method: 'POST' });
            if (resp.ok) {
                await fetchSelfUpdateStatus();
            }
        } catch (err) {
            alert('Failed to trigger self-update.');
        } finally {
            btnTriggerUpdate.disabled = false;
            btnTriggerUpdate.innerHTML = '<span>Trigger Self-Update Now</span>';
        }
    });

    // Update Interval Selector Change Handler
    intervalSelect.addEventListener('change', async () => {
        const secs = parseInt(intervalSelect.value);
        try {
            await fetch('/api/self-update/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ interval_seconds: secs, enabled: true })
            });
            fetchSelfUpdateStatus();
        } catch (err) {
            console.error('Failed to set interval', err);
        }
    });
});
