document.addEventListener('DOMContentLoaded', () => {
    const listContainer = document.getElementById('contracts-list-container');
    const detailPanel = document.getElementById('analysis-detail-panel');
    
    // Modal Elements
    const uploadModal = document.getElementById('upload-modal');
    const openUploadBtn = document.getElementById('btn-open-upload');
    const closeUploadBtn = document.getElementById('btn-close-upload');
    const cancelUploadBtn = document.getElementById('btn-cancel-upload');
    const uploadForm = document.getElementById('upload-form');
    const scannerLoader = document.getElementById('scanner-loader');
    const fileInput = document.getElementById('file-input');
    const fileDropZone = document.getElementById('file-drop-zone');
    const fileDropText = document.getElementById('file-drop-text');

    let currentContractId = null;
    let allContracts = [];

    // Load Contracts list
    async function loadContracts(selectedId = null) {
        try {
            const res = await fetch('/api/contracts/');
            if (!res.ok) throw new Error("Failed to fetch contracts");
            const data = await res.json();
            
            allContracts = data;
            renderStats(data);
            
            const searchInput = document.getElementById('contract-search-input');
            const query = searchInput ? searchInput.value.toLowerCase().trim() : '';
            const filtered = query 
                ? allContracts.filter(c => {
                    const code = (c.contract_code || '').toLowerCase();
                    const title = (c.title || '').toLowerCase();
                    const status = (c.status || '').toLowerCase();
                    return code.includes(query) || title.includes(query) || status.includes(query);
                  })
                : allContracts;
                
            renderContractsList(filtered, selectedId || currentContractId);
        } catch (err) {
            console.error(err);
            listContainer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--risk-high)">
                <i class="fa-solid fa-triangle-exclamation" style="font-size: 32px; margin-bottom: 12px;"></i>
                <p>Error loading registry database.</p>
            </div>`;
        }
    }

    function renderContractsList(contracts, selectedId = null) {
        if (contracts.length === 0) {
            listContainer.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-muted)">
                <i class="fa-regular fa-folder-open" style="font-size: 32px; margin-bottom: 12px;"></i>
                <p>No matching contracts found.</p>
            </div>`;
            return;
        }

        listContainer.innerHTML = '';
        contracts.forEach(c => {
            const card = document.createElement('div');
            card.className = `contract-card ${selectedId == c.id ? 'active' : ''}`;
            card.dataset.id = c.id;

            // Risk display
            let riskClass = 'score-low';
            let indicatorClass = 'bg-low';
            let riskLabel = 'Low Risk';
            
            if (c.risk_score !== null) {
                if (c.risk_score >= 80) {
                    riskClass = 'score-high';
                    indicatorClass = 'bg-high';
                    riskLabel = `Score: ${c.risk_score}% (High)`;
                } else if (c.risk_score >= 50) {
                    riskClass = 'score-medium';
                    indicatorClass = 'bg-medium';
                    riskLabel = `Score: ${c.risk_score}% (Medium)`;
                } else {
                    riskClass = 'score-low';
                    indicatorClass = 'bg-low';
                    riskLabel = `Score: ${c.risk_score}% (Low)`;
                }
            } else {
                riskLabel = 'Pending AI Run';
                riskClass = 'score-low';
                indicatorClass = 'bg-low';
                if (c.status === 'ANALYZING') {
                    riskLabel = 'Analyzing...';
                }
            }

            card.innerHTML = `
                <div class="card-top">
                    <span class="card-code">${c.contract_code}</span>
                    <span class="status-badge status-${c.status.toLowerCase()}">${c.status}</span>
                </div>
                <div class="card-title">${c.title}</div>
                <div class="card-footer">
                    <span>$${c.contract_value !== null && c.contract_value !== undefined ? c.contract_value.toLocaleString() : '0'}</span>
                    <div class="card-score ${riskClass}">
                        <div class="score-indicator ${indicatorClass}"></div>
                        <span>${riskLabel}</span>
                    </div>
                </div>
            `;
            
            card.addEventListener('click', () => {
                document.querySelectorAll('.contract-card').forEach(cc => cc.classList.remove('active'));
                card.classList.add('active');
                showContractDetail(c.id);
            });

            listContainer.appendChild(card);
        });

        // Auto-select active if set
        if (selectedId) {
            const activeCard = listContainer.querySelector(`[data-id="${selectedId}"]`);
            if (activeCard) activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // Search events Setup
    const searchInput = document.getElementById('contract-search-input');
    const clearSearchBtn = document.getElementById('btn-clear-search');
    
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.toLowerCase().trim();
            if (query) {
                if (clearSearchBtn) clearSearchBtn.style.display = 'block';
            } else {
                if (clearSearchBtn) clearSearchBtn.style.display = 'none';
            }
            
            const filtered = allContracts.filter(c => {
                const code = (c.contract_code || '').toLowerCase();
                const title = (c.title || '').toLowerCase();
                const status = (c.status || '').toLowerCase();
                return code.includes(query) || title.includes(query) || status.includes(query);
            });
            
            renderContractsList(filtered, currentContractId);
        });
    }
    
    if (clearSearchBtn) {
        clearSearchBtn.addEventListener('click', () => {
            searchInput.value = '';
            clearSearchBtn.style.display = 'none';
            renderContractsList(allContracts, currentContractId);
            searchInput.focus();
        });
    }

    // Stats calculator
    function renderStats(contracts) {
        const total = contracts.length;
        let highRiskCount = 0;
        let pendingCount = 0;

        contracts.forEach(c => {
            if (c.risk_score && c.risk_score >= 80) highRiskCount++;
            if (c.status === 'ANALYZED') pendingCount++;
        });

        document.getElementById('stat-total').innerText = total;
        document.getElementById('stat-high-risk').innerText = highRiskCount;
        document.getElementById('stat-pending').innerText = pendingCount;
    }

    // Load single contract details
    async function showContractDetail(id, versionId = null) {
        currentContractId = id;
        detailPanel.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted);">
                <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 32px; margin-bottom: 12px;"></i>
            </div>
        `;

        try {
            const apiUrl = versionId ? `/api/contracts/${id}/?version_id=${versionId}` : `/api/contracts/${id}/`;
            const res = await fetch(apiUrl);
            if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                throw new Error(errJson.error || "Failed to fetch contract details");
            }
            const data = await res.json();

            // Render risk assessment tab content
            let aiScanHTML = '';
            
            // 1. AI Analysis Status (Gauge & Findings or Pending message)
            if (data.status === 'ANALYZING') {
                aiScanHTML += `
                    <div style="text-align: center; color: var(--text-muted); padding: 32px 24px; display: flex; flex-direction: column; align-items: center; gap: 12px; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.08); border-radius: 12px; margin-bottom: 24px;">
                        <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 36px; color: var(--accent-primary);"></i>
                        <h3 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 16px; margin: 0;">AI Scanning in Progress</h3>
                        <p style="max-width: 320px; font-size: 13px; color: var(--text-secondary); margin: 0;">Extracting agreement clauses and analyzing legal language...</p>
                    </div>
                `;
            } else if (data.analysis && data.analysis.model_name !== 'N/A') {
                const score = data.analysis.overall_score;
                let strokeColor = '#10b981';
                let scoreStatus = 'Low Risk';
                let scoreStatusClass = 'score-low';
                
                if (score >= 80) {
                    strokeColor = '#ef4444';
                    scoreStatus = 'High Risk';
                    scoreStatusClass = 'score-high';
                } else if (score >= 50) {
                    strokeColor = '#f97316';
                    scoreStatus = 'Medium Risk';
                    scoreStatusClass = 'score-medium';
                }

                const r = 36;
                const circumference = 2 * Math.PI * r;
                const dashoffset = circumference - (score / 100) * circumference;

                let findingsHTML = '';
                if (data.findings && data.findings.length > 0) {
                    data.findings.forEach(f => {
                        findingsHTML += `
                            <div class="finding-item">
                                <div class="finding-header" onclick="toggleFinding(this)">
                                    <div class="finding-left">
                                        <span class="risk-badge badge-${f.risk_level.toLowerCase()}">${f.risk_level}</span>
                                        <span>${f.risk_name}</span>
                                    </div>
                                    <i class="fa-solid fa-chevron-down" style="color: var(--text-muted); font-size: 12px; transition: transform 0.2s;"></i>
                                </div>
                                <div class="finding-body" style="display: none;">
                                    <div class="finding-block">
                                        <div class="block-label">Violated Clause</div>
                                        <div class="block-text" style="font-weight: 500; color: #ffffff;">${f.clause_title}</div>
                                    </div>
                                    <div class="finding-block">
                                        <div class="block-label">Risk Explanation</div>
                                        <div class="block-text">${f.explanation}</div>
                                    </div>
                                    ${f.disadvantaged_party ? `
                                    <div class="finding-block" style="margin-top: 8px;">
                                        <div class="block-label" style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Bên gặp bất lợi / Disadvantaged Party</div>
                                        <div class="block-text" style="font-weight: 600; color: #f87171;">${f.disadvantaged_party}</div>
                                    </div>` : ''}
                                    ${f.recommendation ? `
                                    <div class="rec-box">
                                        <div class="block-label" style="color: var(--risk-low);"><i class="fa-solid fa-lightbulb"></i> AI Recommendation</div>
                                        <div class="block-text">${f.recommendation}</div>
                                    </div>` : ''}
                                </div>
                            </div>
                        `;
                    });
                } else {
                    findingsHTML = `<p style="color: var(--text-muted); font-style: italic;">No specific risks identified.</p>`;
                }

                aiScanHTML += `
                    <h4 class="section-title"><i class="fa-solid fa-chart-line"></i> Risk Assessment Summary</h4>
                    <div class="gauge-area" style="margin-bottom: 20px;">
                        <div class="gauge-circle">
                            <svg>
                                <circle class="bg-circle" cx="40" cy="40" r="${r}"></circle>
                                <circle class="val-circle" cx="40" cy="40" r="${r}" style="stroke-dashoffset: ${dashoffset}; stroke: ${strokeColor};"></circle>
                            </svg>
                            <div class="gauge-text">${score}%</div>
                        </div>
                        <div class="gauge-desc">
                            <div class="gauge-label">Overall AI Threat Level</div>
                            <div class="gauge-status ${scoreStatusClass}">${scoreStatus}</div>
                        </div>
                    </div>

                    <div class="summary-box">
                        ${data.analysis.summary}
                    </div>

                    <h4 class="section-title"><i class="fa-solid fa-magnifying-glass-warning"></i> Detected Clause Violations</h4>
                    <div class="findings-list" style="margin-bottom: 24px;">
                        ${findingsHTML}
                    </div>
                `;
            } else {
                aiScanHTML += `
                    <div style="text-align: center; color: var(--text-muted); padding: 32px 24px; display: flex; flex-direction: column; align-items: center; gap: 12px; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.08); border-radius: 12px; margin-bottom: 24px;">
                        <i class="fa-solid fa-wand-magic-sparkles" style="font-size: 36px; color: var(--accent-primary);"></i>
                        <h3 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 16px; margin: 0;">Run AI Risk Scanner</h3>
                        <p style="max-width: 320px; font-size: 13px; line-height: 1.4; color: var(--text-secondary); margin: 0;">Click below to run the AI risk scanner to extract clauses and flag violations.</p>
                        <button class="btn btn-trigger-analysis" style="margin-top: 4px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none;">
                            <i class="fa-solid fa-play"></i> Run AI Analysis
                        </button>
                    </div>
                `;
            }

            let analysisHTML = `<div id="ai-scanner-results-container">${aiScanHTML}</div>`;

            // 2. Legal Expert Review Section (Always shown at the bottom)
            let reviewsHTML = '';
            if (data.reviews && data.reviews.length > 0) {
                data.reviews.forEach(r => {
                    reviewsHTML += `
                        <div class="review-card">
                            <div class="review-header">
                                <div class="reviewer-info">
                                    <i class="fa-solid fa-user-tie" style="color: var(--accent-primary);"></i>
                                    <span>${r.reviewer}</span>
                                </div>
                                <span class="risk-badge badge-${r.final_risk_level.toLowerCase()}">${r.final_risk_level} Decision</span>
                            </div>
                            <div class="review-comment">"${r.comment}"</div>
                        </div>
                    `;
                });
            }

            const showReviewForm = (!data.reviews || data.reviews.length === 0) && data.status !== 'APPROVED';

            analysisHTML += `
                <div class="reviews-section">
                    <h4 class="section-title"><i class="fa-solid fa-user-shield"></i> Legal Expert Review</h4>
                    ${reviewsHTML}
                    
                    ${showReviewForm ? `
                    <form class="review-form" id="submit-review-form">
                        <div class="form-group">
                            <label class="form-label">Review Verdict *</label>
                            <select class="form-control" name="final_risk_level" required>
                                <option value="LOW">LOW RISK - Approved</option>
                                <option value="MEDIUM" selected>MEDIUM RISK - Mitigation Required</option>
                                <option value="HIGH">HIGH RISK - Rejection/Renegotiate</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Expert Commentary *</label>
                            <textarea class="form-control" name="comment" placeholder="Provide legal reasoning and next action steps..." required></textarea>
                        </div>
                        <button type="submit" class="btn" style="align-self: flex-end; margin-top: 8px;">
                            <i class="fa-solid fa-signature"></i> Sign & Submit Review
                        </button>
                    </form>
                    ` : `
                    <div style="display: flex; gap: 8px; align-items: center; color: var(--risk-low); font-size: 13.5px; background: rgba(16,185,129,0.05); padding: 12px; border-radius: 8px; border: 1px dashed rgba(16,185,129,0.2);">
                        <i class="fa-solid fa-circle-check"></i>
                        <span>Expert review has been successfully submitted and the contract status has been marked as <strong>Approved</strong>.</span>
                    </div>
                    `}

                    <!-- Workflow Approval Block at the very bottom -->
                    <div class="workflow-section" style="margin-top: 24px; padding-top: 20px; border-top: 1px dashed rgba(255,255,255,0.1);">
                        <h4 class="section-title"><i class="fa-solid fa-diagram-project"></i> Workflow Approval System</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); padding: 16px 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); gap: 16px; flex-wrap: wrap;">
                            <div>
                                <h5 style="margin: 0 0 4px 0; color: #ffffff; font-size: 14px; font-weight: 600;">Khởi tạo quy trình phê duyệt (Workflow)</h5>
                                <p style="margin: 0; color: var(--text-secondary); font-size: 12.5px; line-height: 1.4;">Đẩy hợp đồng và đánh giá rủi ro sang dịch vụ Workflow Service để khởi tạo luồng duyệt.</p>
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                ${data.has_workflow && data.workflow_id ? `
                                <a href="/board/${data.workflow_id}/" target="_blank" style="color: #a5b4fc; text-decoration: none; font-size: 12.5px; font-weight: 600; background: rgba(99,102,241,0.12); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(99,102,241,0.3); display: inline-flex; align-items: center; gap: 6px;" title="Mở trang chi tiết quy trình phê duyệt">
                                    <i class="fa-solid fa-route"></i> Xem chi tiết Workflow <i class="fa-solid fa-up-right-from-square" style="font-size: 11px;"></i>
                                </a>` : ''}
                                <button class="btn btn-push-workflow" data-contract-id="${data.id}" style="background: linear-gradient(135deg, #059669, #10b981); color: #ffffff; border: none; font-weight: 600; padding: 10px 22px; font-size: 13.5px; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 14px rgba(16,185,129,0.3); cursor: pointer; white-space: nowrap; border-radius: 8px;">
                                    ${data.has_workflow ? '<i class="fa-solid fa-arrows-rotate"></i> Re-push to Workflow' : '<i class="fa-solid fa-paper-plane"></i> Push to Workflow'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Render original clauses list HTML
            let clausesHTML = `
                <div style="display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;">
                    <button class="btn btn-manual-extract-dash" data-contract-id="${data.id}" data-version-id="${data.active_version_id || ''}" style="flex: 1; min-width: 180px; display: flex; align-items: center; justify-content: center; gap: 8px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.12); color: #ffffff; font-weight: 500; padding: 10px 16px; border-radius: 8px; cursor: pointer; transition: all 0.2s;">
                        <i class="fa-solid fa-gears" style="color: #a5b4fc;"></i> Trích xuất Thủ công
                    </button>
                    <button class="btn btn-ai-extract-dash" data-contract-id="${data.id}" data-version-id="${data.active_version_id || ''}" style="flex: 1; min-width: 180px; display: flex; align-items: center; justify-content: center; gap: 8px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; color: #ffffff; font-weight: 600; padding: 10px 16px; border-radius: 8px; cursor: pointer; box-shadow: 0 4px 14px rgba(99,102,241,0.3);">
                        <i class="fa-solid fa-wand-magic-sparkles"></i> Trích xuất AI
                    </button>
                </div>
            `;

            if (data.clauses && data.clauses.length > 0) {
                data.clauses.forEach(cl => {
                    clausesHTML += `
                        <div class="clause-item">
                            <div class="clause-title-text">
                                <i class="fa-solid fa-paragraph" style="color: var(--accent-primary);"></i>
                                <span>${cl.title}</span>
                            </div>
                            <div class="clause-body-text">${cl.content.replace(/\n/g, '<br>')}</div>
                        </div>
                    `;
                });
            } else {
                clausesHTML += `
                    <div style="text-align: center; padding: 40px 24px; background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.08); border-radius: 12px; display: flex; flex-direction: column; align-items: center; gap: 10px;">
                        <i class="fa-solid fa-file-circle-exclamation" style="font-size: 36px; color: var(--text-muted);"></i>
                        <h4 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 15px; margin: 0;">Chưa tìm thấy điều khoản (No Clauses Extracted)</h4>
                        <p style="max-width: 440px; font-size: 13px; color: var(--text-secondary); margin: 0; line-height: 1.5;">Vui lòng bấm <strong>Trích xuất Thủ công</strong> hoặc <strong>Trích xuất AI</strong> ở trên để bóc tách thông tin điều khoản cho hợp đồng này.</p>
                    </div>
                `;
            }

            // Render the full detail panel structure
            detailPanel.innerHTML = `
                <div class="analysis-header">
                    <div class="analysis-title-area">
                        <h2 class="analysis-title">${data.title}</h2>
                        <div class="analysis-meta">
                            <div class="meta-tag"><i class="fa-solid fa-barcode"></i> ${data.contract_code}</div>
                            <div class="meta-tag"><i class="fa-solid fa-tag"></i> ${data.contract_type}</div>
                            <div class="meta-tag"><i class="fa-solid fa-dollar-sign"></i> ${data.contract_value !== null && data.contract_value !== undefined ? data.contract_value.toLocaleString() : '0'}</div>
                            ${data.start_date ? `<div class="meta-tag"><i class="fa-solid fa-calendar-days"></i> ${data.start_date} to ${data.end_date}</div>` : ''}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-trigger-analysis" data-contract-id="${data.id}" style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #ffffff; border: none; font-weight: 600; display: flex; align-items: center; gap: 6px; box-shadow: 0 4px 12px rgba(99,102,241,0.3); cursor: pointer;">
                            <i class="fa-solid fa-wand-magic-sparkles"></i> Run AI Analysis
                        </button>
                        <a href="/contracts/${data.id}/" class="btn" style="background: var(--accent-primary); color: #ffffff; text-decoration: none;">
                            <i class="fa-solid fa-expand"></i> Full Reader Page
                        </a>
                        ${data.file_path ? `
                        <a href="${data.file_path}" target="_blank" download class="btn" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: var(--text-main); box-shadow: none; text-decoration: none;">
                            <i class="fa-solid fa-file-pdf"></i> Download PDF
                        </a>` : ''}
                    </div>
                </div>

                <div class="tabs-header">
                    <button class="tab-btn active" id="btn-tab-analysis"><i class="fa-solid fa-chart-line"></i> Risk Assessment</button>
                    <button class="tab-btn" id="btn-tab-clauses"><i class="fa-solid fa-file-invoice"></i> Original Clauses (${data.clauses ? data.clauses.length : 0})</button>
                    <button class="tab-btn" id="btn-tab-fulltext"><i class="fa-solid fa-file-lines"></i> Full Document</button>
                </div>

                <div id="tab-analysis" class="tab-content active">
                    ${analysisHTML}
                </div>

                <div id="tab-clauses" class="tab-content">
                    <div class="clauses-list">
                        ${clausesHTML}
                    </div>
                </div>

                <div id="tab-fulltext" class="tab-content">
                    <pre class="doc-preview-container">${data.raw_content || 'No content available.'}</pre>
                </div>
            `;

            // Bind Tab Navigation Events
            const tabAnalysisBtn = document.getElementById('btn-tab-analysis');
            const tabClausesBtn = document.getElementById('btn-tab-clauses');
            const tabFulltextBtn = document.getElementById('btn-tab-fulltext');
            
            const tabAnalysisContent = document.getElementById('tab-analysis');
            const tabClausesContent = document.getElementById('tab-clauses');
            const tabFulltextContent = document.getElementById('tab-fulltext');

            function deactivateAllTabs() {
                tabAnalysisBtn.classList.remove('active');
                tabClausesBtn.classList.remove('active');
                tabFulltextBtn.classList.remove('active');
                
                tabAnalysisContent.classList.remove('active');
                tabClausesContent.classList.remove('active');
                tabFulltextContent.classList.remove('active');
            }

            tabAnalysisBtn.addEventListener('click', () => {
                deactivateAllTabs();
                tabAnalysisBtn.classList.add('active');
                tabAnalysisContent.classList.add('active');
            });

            tabClausesBtn.addEventListener('click', () => {
                deactivateAllTabs();
                tabClausesBtn.classList.add('active');
                tabClausesContent.classList.add('active');
            });

            tabFulltextBtn.addEventListener('click', () => {
                deactivateAllTabs();
                tabFulltextBtn.classList.add('active');
                tabFulltextContent.classList.add('active');
            });


            // Bind Run AI Analysis Event if present
            const triggerAnalysisBtns = document.querySelectorAll('.btn-trigger-analysis, #btn-trigger-analysis');
            triggerAnalysisBtns.forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    const targetContractId = btn.dataset.contractId || data.id;

                    // 1. Ensure Risk Assessment tab is active
                    const tabAnalysisBtn = document.getElementById('btn-tab-analysis');
                    const tabAnalysisContent = document.getElementById('tab-analysis');
                    if (tabAnalysisBtn && tabAnalysisContent) {
                        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        tabAnalysisBtn.classList.add('active');
                        tabAnalysisContent.classList.add('active');
                    }
                    
                    // 2. Disable trigger buttons to prevent duplicate clicks
                    triggerAnalysisBtns.forEach(b => {
                        b.disabled = true;
                        b.style.opacity = '0.7';
                        b.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...`;
                    });

                    // 3. Render inline loading indicator strictly inside AI scanner section (keeping Legal Expert Review intact)
                    const scannerContainer = document.getElementById('ai-scanner-results-container') || tabAnalysisContent;
                    if (scannerContainer) {
                        scannerContainer.innerHTML = `
                            <div class="risk-loading-card" style="text-align: center; color: var(--text-muted); padding: 50px 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; background: rgba(99, 102, 241, 0.04); border: 1px dashed rgba(99, 102, 241, 0.3); border-radius: 16px; margin: 12px 0;">
                                <div style="position: relative; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center;">
                                    <i class="fa-solid fa-wand-magic-sparkles" style="font-size: 28px; color: #a5b4fc;"></i>
                                    <div style="position: absolute; top:0; left:0; width:64px; height:64px; border:3px solid rgba(99,102,241,0.15); border-top-color:#6366f1; border-radius:50%; animation:spin 1s linear infinite;"></div>
                                </div>
                                <div>
                                    <h3 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 17px; margin: 0 0 6px 0; font-weight: 700;">AI Engine is Scanning Contract Risks...</h3>
                                    <p style="max-width: 400px; font-size: 13px; color: var(--text-secondary); margin: 0; line-height: 1.5;">Sending contract text to Kaggle fine-tuned Qwen2.5-3B model. Extracting clauses and calculating risk scores...</p>
                                </div>
                            </div>
                        `;
                    }
                    
                    try {
                        const runRes = await fetch(`/api/contracts/${targetContractId}/analyze/`, {
                            method: 'POST',
                            headers: {
                                'X-CSRFToken': getCookie('csrftoken')
                            }
                        });
                        const runData = await runRes.json();
                        
                        if (runData.success) {
                            // Reload details panel and contract list immediately
                            await showContractDetail(targetContractId);
                            await loadContracts(targetContractId);
                            showToast("Phân tích AI hoàn tất! Form ký duyệt hợp đồng đã sẵn sàng bên dưới.", "success");
                        } else {
                            const errorMsg = runData.error || "Unknown error";
                            const isModelError = errorMsg.includes("503") || errorMsg.includes("communication failed") || errorMsg.includes("not loaded");
                            if (isModelError) {
                                showToast("Hệ thống chưa kết nối được với mô hình AI. Vui lòng tải mô hình hoặc kích hoạt GPU để tiến hành phân tích hợp đồng.", "warning");
                            } else {
                                showToast(errorMsg, "error");
                            }
                            await showContractDetail(targetContractId);
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("Lỗi kết nối: Không thể gửi yêu cầu phân tích tới hệ thống.", "error");
                        await showContractDetail(targetContractId);
                    }
                });
            });

            // Bind Push to Workflow Event
            detailPanel.querySelectorAll('.btn-push-workflow').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const targetContractId = btn.getAttribute('data-contract-id') || currentContractId;
                    btn.disabled = true;
                    btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Pushing...';

                    try {
                        const wfRes = await fetch(`/api/contracts/${targetContractId}/workflow/`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({ version_id: data.active_version_id || null })
                        });
                        const wfData = await wfRes.json();
                        
                        if (wfData.success) {
                            showToast("Đã đẩy hợp đồng lên quy trình phê duyệt Workflow thành công!", "success");
                            await showContractDetail(targetContractId);
                            await loadContracts(targetContractId);
                        } else {
                            showToast("Lỗi đẩy Workflow: " + (wfData.error || "Không thể đẩy workflow"), "error");
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("Không thể kết nối với dịch vụ Workflow.", "error");
                    } finally {
                        btn.disabled = false;
                        btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Push to Workflow';
                    }
                });
            });

            // Bind Manual Extract in Dashboard
            detailPanel.querySelectorAll('.btn-manual-extract-dash').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const contractId = btn.getAttribute('data-contract-id');
                    const vId = btn.getAttribute('data-version-id') || null;
                    const scannerLoader = document.getElementById('scanner-loader');
                    const origHTML = btn.innerHTML;
                    
                    btn.disabled = true;
                    btn.style.opacity = '0.75';
                    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang trích xuất thủ công...';
                    
                    if (scannerLoader) {
                        scannerLoader.classList.add('active');
                        const scannerText = scannerLoader.querySelector('.scanner-text');
                        if (scannerText) scannerText.innerHTML = "RUNNING RULE-BASED MANUAL EXTRACTION...";
                    }
                    
                    try {
                        const res = await fetch(`/api/contracts/${contractId}/manual-extract/`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({ version_id: vId })
                        });
                        const resData = await res.json();
                        
                        if (resData.success) {
                            showToast("Trích xuất thủ công hoàn tất!", "success");
                            await showContractDetail(contractId, vId);
                            await loadContracts(contractId);
                        } else {
                            showToast(resData.error || "Lỗi khi trích xuất thủ công.", "error");
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("Lỗi kết nối khi trích xuất thủ công.", "error");
                    } finally {
                        if (scannerLoader) scannerLoader.classList.remove('active');
                        btn.disabled = false;
                        btn.style.opacity = '1';
                        btn.innerHTML = origHTML;
                    }
                });
            });

            // Bind AI Extract in Dashboard
            detailPanel.querySelectorAll('.btn-ai-extract-dash').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const contractId = btn.getAttribute('data-contract-id');
                    const vId = btn.getAttribute('data-version-id') || null;
                    const scannerLoader = document.getElementById('scanner-loader');
                    const origHTML = btn.innerHTML;
                    
                    btn.disabled = true;
                    btn.style.opacity = '0.75';
                    btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles fa-spin"></i> Đang trích xuất AI...';
                    
                    if (scannerLoader) {
                        scannerLoader.classList.add('active');
                        const scannerText = scannerLoader.querySelector('.scanner-text');
                        if (scannerText) scannerText.innerHTML = "AI SCANNING CLAUSES & EXTRACTING ENTITIES...";
                    }
                    
                    try {
                        const res = await fetch(`/api/ai/contracts/${contractId}/extract-entities/`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({ version_id: vId, re_extract: true })
                        });
                        const resData = await res.json();
                        
                        if (resData.version_id || resData.clauses) {
                            showToast("Trích xuất AI hoàn tất!", "success");
                            await showContractDetail(contractId, vId);
                            await loadContracts(contractId);
                        } else {
                            showToast(resData.error || "Lỗi khi trích xuất AI.", "error");
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("Lỗi kết nối khi trích xuất AI.", "error");
                    } finally {
                        if (scannerLoader) scannerLoader.classList.remove('active');
                        btn.disabled = false;
                        btn.style.opacity = '1';
                        btn.innerHTML = origHTML;
                    }
                });
            });

            // Bind Review Form Event if present
            const reviewForm = document.getElementById('submit-review-form');
            if (reviewForm) {
                reviewForm.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const final_risk_level = reviewForm.final_risk_level.value;
                    const comment = reviewForm.comment.value;
                    
                    const analysisId = (data.analysis && data.analysis.id) ? data.analysis.id : null;
                    if (!analysisId) {
                        showToast("Chưa có kết quả phân tích AI để tiến hành ký duyệt.", "warning");
                        return;
                    }
                    
                    try {
                        const rRes = await fetch(`/api/analyses/${analysisId}/review/`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            },
                            body: JSON.stringify({ final_risk_level, comment })
                        });
                        
                        const rData = await rRes.json();
                        if (rData.success) {
                            // Reload details and list to update status
                            showContractDetail(currentContractId);
                            loadContracts(currentContractId);
                        } else {
                            showToast("Lỗi gửi đánh giá: " + (rData.error || "Unknown error"), "error");
                        }
                    } catch (err) {
                        console.error(err);
                        showToast("Không thể gửi yêu cầu đánh giá.", "error");
                    }
                });
            }

        } catch (err) {
            console.error(err);
            detailPanel.innerHTML = `
                <div class="empty-state" style="color: var(--risk-high)">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 64px;"></i>
                    <h2>Failed to Load Details</h2>
                    <p>${err.message || 'An error occurred retrieving database records for this contract ID.'}</p>
                </div>
            `;
        }
    }

    // Modal Interactions
    openUploadBtn.addEventListener('click', () => {
        uploadForm.reset();
        fileDropText.innerHTML = 'Drag and drop file here, or click to choose file';
        uploadModal.classList.add('active');
    });

    function closeModal() {
        uploadModal.classList.remove('active');
    }

    closeUploadBtn.addEventListener('click', closeModal);
    cancelUploadBtn.addEventListener('click', closeModal);

    // Drag and drop events
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileDropText.innerHTML = `<strong>Selected file:</strong> ${fileInput.files[0].name}`;
        }
    });

    fileDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropZone.classList.add('dragover');
    });

    fileDropZone.addEventListener('dragleave', () => {
        fileDropZone.classList.remove('dragover');
    });

    fileDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            fileDropText.innerHTML = `<strong>Selected file:</strong> ${fileInput.files[0].name}`;
        }
    });

    // Form Submit
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const fileSelected = fileInput.files.length > 0;
        const rawContentVal = uploadForm.raw_content.value.trim();
        
        if (!fileSelected && !rawContentVal) {
            alert("Please either upload a contract file or paste the contract text.");
            return;
        }

        const formData = new FormData(uploadForm);
        
        // Show AI Scanner loader
        scannerLoader.classList.add('active');
        
        try {
            const res = await fetch('/api/contracts/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: formData
            });
            
            const data = await res.json();
            
            scannerLoader.classList.remove('active');
            if (res.ok && data.success) {
                closeModal();
                if (new URLSearchParams(window.location.search).get('from') === 'a3') {
                    window.location.assign(`/ai-extract/contracts/${encodeURIComponent(data.contract_id)}/`);
                    return;
                }
                loadContracts(data.contract_id);
                showContractDetail(data.contract_id);
            } else {
                alert("Upload failed: " + (data.error || "Unknown error"));
            }

        } catch (err) {
            scannerLoader.classList.remove('active');
            console.error(err);
            alert("Network error: Failed to upload contract.");
        }
    });

    // Master Risks Management Modal Interactions
    const openRisksBtn = document.getElementById('btn-open-risks-mgmt');
    const risksModal = document.getElementById('risks-modal');
    const closeRisksBtn = document.getElementById('btn-close-risks');
    const risksContainer = document.getElementById('modal-risks-container');
    const addRiskForm = document.getElementById('add-risk-form');

    async function loadMasterRisks() {
        risksContainer.innerHTML = `
            <div style="text-align: center; padding: 20px; color: var(--text-muted);">
                <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 20px; margin-bottom: 8px;"></i>
                <p style="font-size: 12px;">Loading risk categories...</p>
            </div>
        `;
        try {
            const res = await fetch('/api/risks/');
            if (!res.ok) throw new Error("Failed to load risks");
            const data = await res.json();
            
            if (data.length === 0) {
                risksContainer.innerHTML = `<p style="color: var(--text-muted); font-style: italic; text-align: center; padding: 20px;">No risk definitions found.</p>`;
                return;
            }
            
            let html = '';
            data.forEach(r => {
                let badgeClass = 'badge-medium';
                if (r.severity_level === 'HIGH' || r.severity_level === 'CRITICAL') {
                    badgeClass = 'badge-high';
                } else if (r.severity_level === 'LOW') {
                    badgeClass = 'badge-low';
                }
                
                let contractsHTML = '';
                if (r.contracts && r.contracts.length > 0) {
                    contractsHTML += `
                        <div style="margin-top: 8px; border-top: 1px dashed var(--panel-border); padding-top: 8px;">
                            <span style="font-size: 10.5px; color: var(--text-muted); display: block; margin-bottom: 6px;"><i class="fa-solid fa-link"></i> Associated Contracts:</span>
                            <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    `;
                    r.contracts.forEach(c => {
                        contractsHTML += `
                            <a href="#" onclick="event.preventDefault(); selectContractFromModal(${c.id})" style="font-size: 11px; background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25); color: #a5b4fc; padding: 3px 8px; border-radius: 4px; text-decoration: none; display: inline-flex; align-items: center; gap: 4px; transition: var(--transition);">
                                <i class="fa-solid fa-file-signature" style="font-size: 10px;"></i>
                                <span>${c.contract_code}</span>
                            </a>
                        `;
                    });
                    contractsHTML += `
                            </div>
                        </div>
                    `;
                } else {
                    contractsHTML += `
                        <div style="margin-top: 8px; border-top: 1px dashed var(--panel-border); padding-top: 6px; font-size: 10.5px; color: var(--text-muted); font-style: italic;">
                            No contracts currently flagged with this risk.
                        </div>
                    `;
                }
                
                html += `
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--panel-border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 6px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #ffffff; font-size: 13.5px;">${r.risk_name}</strong>
                            <span class="risk-badge ${badgeClass}" style="font-size: 10px; padding: 2px 6px;">${r.severity_level}</span>
                        </div>
                        <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.4; margin: 0;">${r.description || 'No description provided.'}</p>
                        ${contractsHTML}
                    </div>
                `;
            });
            risksContainer.innerHTML = html;
        } catch (err) {
            console.error(err);
            risksContainer.innerHTML = `<p style="color: var(--risk-high); font-size: 12px; text-align: center; padding: 20px;">Error loading risks.</p>`;
        }
    }

    if (openRisksBtn) {
        openRisksBtn.addEventListener('click', () => {
            addRiskForm.reset();
            risksModal.classList.add('active');
            loadMasterRisks();
        });
    }

    if (closeRisksBtn) {
        closeRisksBtn.addEventListener('click', () => {
            risksModal.classList.remove('active');
        });
    }

    if (addRiskForm) {
        addRiskForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const risk_name = addRiskForm.risk_name.value.trim();
            const severity_level = addRiskForm.severity_level.value;
            const description = addRiskForm.description.value.trim();
            
            try {
                const res = await fetch('/api/risks/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ risk_name, severity_level, description })
                });
                const data = await res.json();
                if (data.success) {
                    addRiskForm.reset();
                    loadMasterRisks();
                } else {
                    alert("Failed to add risk: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                console.error(err);
                alert("Network error: Failed to add new risk category.");
            }
        });
    }

    // Init load
    const urlParams = new URLSearchParams(window.location.search);
    const initialContractId = urlParams.get('contract_id');
    const initialVersionId = urlParams.get('version_id');
    if (urlParams.get('upload') === '1') {
        openUploadBtn.click();
        uploadForm.elements.contract_code.focus();
    }
    loadContracts(initialContractId);
    if (initialContractId) {
        showContractDetail(initialContractId, initialVersionId);
    }
});

// Toggle Clause finding details open/close (Global handler)
window.toggleFinding = function(header) {
    const item = header.parentElement;
    item.classList.toggle('active');
    
    const body = item.querySelector('.finding-body');
    const icon = header.querySelector('i');
    
    if (item.classList.contains('active')) {
        body.style.display = 'flex';
        icon.style.transform = 'rotate(180deg)';
    } else {
        body.style.display = 'none';
        icon.style.transform = 'rotate(0deg)';
    }
};

// Global handler to navigate to a contract from the risks modal
window.selectContractFromModal = function(id) {
    const risksModal = document.getElementById('risks-modal');
    if (risksModal) {
        risksModal.classList.remove('active');
    }
    
    // Find contract card in the registry list, highlight it and click it
    const card = document.querySelector(`.contract-card[data-id="${id}"]`);
    if (card) {
        card.click();
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
};

// Toast notification helper
window.showToast = function(message, type = 'error') {
    if (!document.getElementById('toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.innerHTML = `
            .toast-container {
                position: fixed;
                top: 24px;
                right: 24px;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                gap: 12px;
                pointer-events: none;
            }
            .toast-card {
                background: rgba(13, 17, 23, 0.95);
                border: 1px solid rgba(239, 68, 68, 0.2);
                border-left: 4px solid #ef4444;
                color: #ffffff;
                padding: 16px 20px;
                border-radius: 12px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(12px);
                display: flex;
                align-items: center;
                gap: 16px;
                min-width: 320px;
                max-width: 450px;
                transform: translateX(120%);
                transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), opacity 0.3s ease;
                pointer-events: auto;
                opacity: 0;
            }
            .toast-card.show {
                transform: translateX(0);
                opacity: 1;
            }
            .toast-card.warning {
                border-left-color: #f59e0b;
                border-color: rgba(245, 158, 11, 0.2);
            }
            .toast-icon {
                font-size: 20px;
                color: #ef4444;
                flex-shrink: 0;
            }
            .toast-card.warning .toast-icon {
                color: #f59e0b;
            }
            .toast-content {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            .toast-title {
                font-family: 'Outfit', sans-serif;
                font-weight: 600;
                font-size: 14.5px;
            }
            .toast-desc {
                font-size: 12.5px;
                color: rgba(255, 255, 255, 0.7);
                line-height: 1.4;
            }
            .toast-close {
                margin-left: auto;
                background: none;
                border: none;
                color: rgba(255, 255, 255, 0.4);
                cursor: pointer;
                padding: 4px;
                transition: color 0.2s;
            }
            .toast-close:hover {
                color: #ffffff;
            }
        `;
        document.head.appendChild(style);
    }

    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast-card ${type}`;
    
    const iconClass = type === 'warning' ? 'fa-solid fa-triangle-exclamation' : 'fa-solid fa-circle-exclamation';
    const titleText = type === 'warning' ? 'Cảnh Báo Hệ Thống' : 'Lỗi Hệ Thống';

    toast.innerHTML = `
        <i class="${iconClass} toast-icon"></i>
        <div class="toast-content">
            <div class="toast-title">${titleText}</div>
            <div class="toast-desc">${message}</div>
        </div>
        <button class="toast-close"><i class="fa-solid fa-xmark"></i></button>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    const closeBtn = toast.querySelector('.toast-close');
    const dismiss = () => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    };

    closeBtn.addEventListener('click', dismiss);
    setTimeout(dismiss, 8000);
};

// Helper function to extract CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

