document.addEventListener('DOMContentLoaded', () => {
    const listEl = document.getElementById('analyses-list');
    const resultCountEl = document.getElementById('result-count');
    const statTotalEl = document.getElementById('stat-total');
    const statHighEl = document.getElementById('stat-high');
    const statAvgEl = document.getElementById('stat-avg');
    const filterSearch = document.getElementById('filter-search');
    const filterSort = document.getElementById('filter-sort');
    const riskChips = document.querySelectorAll('#filter-risk .chip');
    const btnReset = document.getElementById('btn-reset-filters');

    let allAnalyses = [];
    let activeRisk = 'ALL';

    async function loadAnalyses() {
        listEl.innerHTML = `
            <div class="loading-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin"></i>
                <p>Loading analysis history...</p>
            </div>
        `;
        try {
            const res = await fetch('/api/analyses/');
            if (!res.ok) throw new Error('Failed to fetch analyses');
            allAnalyses = await res.json();
            renderStats(allAnalyses);
            applyFilters();
        } catch (err) {
            console.error(err);
            listEl.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-triangle-exclamation" style="color: var(--risk-high);"></i>
                    <h3>Failed to load history</h3>
                    <p style="font-size: 13px;">Could not connect to the API. Please try again.</p>
                </div>
            `;
        }
    }

    function renderStats(data) {
        const total = data.length;
        const highCount = data.filter(a => a.risk_label === 'HIGH').length;
        const avg = total > 0
            ? Math.round(data.reduce((s, a) => s + a.overall_score, 0) / total)
            : 0;
        statTotalEl.textContent = total;
        statHighEl.textContent = highCount;
        statAvgEl.textContent = avg + '%';
    }

    function applyFilters() {
        const search = filterSearch.value.trim().toLowerCase();
        const sort = filterSort.value;

        let filtered = allAnalyses.filter(a => {
            const matchRisk = activeRisk === 'ALL' || a.risk_label === activeRisk;
            const matchSearch = !search ||
                a.contract_code.toLowerCase().includes(search) ||
                a.contract_title.toLowerCase().includes(search);
            return matchRisk && matchSearch;
        });

        filtered.sort((a, b) => {
            if (sort === 'oldest') return new Date(a.created_at) - new Date(b.created_at);
            if (sort === 'score-desc') return b.overall_score - a.overall_score;
            if (sort === 'score-asc') return a.overall_score - b.overall_score;
            return new Date(b.created_at) - new Date(a.created_at); // newest default
        });

        renderList(filtered);
        resultCountEl.textContent = `${filtered.length} result${filtered.length !== 1 ? 's' : ''}`;
    }

    function renderList(data) {
        if (data.length === 0) {
            listEl.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <h3>No analyses found</h3>
                    <p style="font-size: 13px; max-width: 280px;">Try adjusting your search or filter criteria.</p>
                </div>
            `;
            return;
        }

        listEl.innerHTML = data.map(a => {
            const scoreClass = a.risk_label === 'HIGH' ? 'score-high'
                : a.risk_label === 'MEDIUM' ? 'score-medium' : 'score-low';

            const statusClass = {
                ANALYZED: 'status-analyzed',
                APPROVED: 'status-approved',
                DRAFT: 'status-draft',
                ANALYZING: 'status-draft'
            }[a.contract_status] || 'status-draft';

            let findingsHTML = '';
            if (a.findings_preview && a.findings_preview.length > 0) {
                findingsHTML = a.findings_preview.map(f => {
                    const cls = (f.risk_level === 'HIGH' || f.risk_level === 'CRITICAL') ? 'finding-high'
                        : f.risk_level === 'MEDIUM' ? 'finding-medium' : 'finding-low';
                    return `<span class="finding-tag ${cls}">${f.risk_name}</span>`;
                }).join('');

                const extra = a.findings_count - a.findings_preview.length;
                if (extra > 0) {
                    findingsHTML += `<span class="finding-tag finding-more">+${extra} more</span>`;
                }
            } else {
                findingsHTML = `<span class="finding-tag finding-low" style="font-style: italic;">No findings detected</span>`;
            }

            return `
                <div class="analysis-card">
                    <div class="score-circle ${scoreClass}">${a.overall_score}%</div>
                    <div class="card-body">
                        <div class="card-title">${a.contract_title}</div>
                        <div class="card-meta">
                            <span class="meta-pill"><i class="fa-solid fa-barcode"></i>${a.contract_code}</span>
                            <span class="meta-pill"><i class="fa-solid fa-microchip"></i>${a.model_name}</span>
                            <span class="meta-pill"><i class="fa-solid fa-flag"></i>${a.findings_count} finding${a.findings_count !== 1 ? 's' : ''}</span>
                        </div>
                        <div class="findings-row">${findingsHTML}</div>
                    </div>
                    <div class="card-actions">
                        <span class="card-timestamp"><i class="fa-regular fa-clock"></i> ${a.created_at}</span>
                        <span class="status-badge ${statusClass}">${a.contract_status}</span>
                        <a href="/contracts/${a.contract_id}/" class="btn-view">
                            <i class="fa-solid fa-arrow-up-right-from-square"></i> View Contract
                        </a>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Filter event bindings
    filterSearch.addEventListener('input', applyFilters);
    filterSort.addEventListener('change', applyFilters);

    riskChips.forEach(chip => {
        chip.addEventListener('click', () => {
            riskChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            activeRisk = chip.dataset.val;
            applyFilters();
        });
    });

    btnReset.addEventListener('click', () => {
        filterSearch.value = '';
        filterSort.value = 'newest';
        activeRisk = 'ALL';
        riskChips.forEach(c => c.classList.remove('active'));
        riskChips[0].classList.add('active');
        applyFilters();
    });

    loadAnalyses();
});
