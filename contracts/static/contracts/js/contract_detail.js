document.addEventListener('DOMContentLoaded', () => {
    // 0. Auto Initial AJAX Fetch on page load
    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const contractId = pathParts[1];
    const urlParams = new URLSearchParams(window.location.search);
    const versionId = urlParams.get('version_id');
    if (contractId && window.refreshContractDetailViaAjax) {
        window.refreshContractDetailViaAjax(contractId, versionId);
    }
    // 1. Gauge Animation
    const gauge = document.getElementById('gauge-val');
    const gaugeStatus = document.getElementById('gauge-status');
    
    if (gauge) {
        const score = parseFloat(gauge.dataset.score);
        const r = 36;
        const circumference = 2 * Math.PI * r;
        const offset = circumference - (score / 100) * circumference;
        
        // Trigger stroke offset transition
        setTimeout(() => {
            gauge.style.strokeDashoffset = offset;
        }, 100);
        
        // Color the circle and update status text
        if (score >= 80) {
            gauge.style.stroke = '#ef4444';
            gaugeStatus.innerHTML = 'High Risk';
            gaugeStatus.className = 'gauge-status score-high';
        } else if (score >= 50) {
            gauge.style.stroke = '#f97316';
            gaugeStatus.innerHTML = 'Medium Risk';
            gaugeStatus.className = 'gauge-status score-medium';
        } else {
            gauge.style.stroke = '#10b981';
            gaugeStatus.innerHTML = 'Low Risk';
            gaugeStatus.className = 'gauge-status score-low';
        }
    }

    // 2. Expert Review Form Handling
    const reviewForm = document.getElementById('detail-review-form');
    if (reviewForm) {
        reviewForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const analysisId = reviewForm.dataset.analysisId;
            const final_risk_level = reviewForm.final_risk_level.value;
            const comment = reviewForm.comment.value;
            
            try {
                const res = await fetch(`/api/analyses/${analysisId}/review/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ final_risk_level, comment })
                });
                
                const data = await res.json();
                if (data.success) {
                    showToast("Submit expert review success!", "success");
                    const contractId = window.location.pathname.split('/')[2];
                    if (window.refreshContractDetailViaAjax) {
                        window.refreshContractDetailViaAjax(contractId);
                    } else {
                        window.location.reload();
                    }
                } else {
                    alert("Error submitting review: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                console.error(err);
                alert("Failed to submit review request.");
            }
        });
    }

    // 3. Run/Re-run AI Analysis Trigger (Detailed Page)
    const triggerButtons = document.querySelectorAll('#btn-detail-trigger-analysis, #btn-detail-reanalyze');
    
    triggerButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const contractId = btn.dataset.contractId;
            const analysisTab = document.getElementById('analysis-tab');

            // 1. Ensure Risk Analysis tab is active
            const panelTabBtns = document.querySelectorAll('.panel-tab-btn');
            panelTabBtns.forEach(b => {
                b.classList.remove('active');
                b.style.color = 'var(--text-muted)';
                if (b.dataset.target === 'analysis-tab') {
                    b.classList.add('active');
                    b.style.color = '#ffffff';
                }
            });
            const panes = document.querySelectorAll('.tab-content-wrapper > .tab-pane');
            panes.forEach(pane => {
                if (pane.id === 'analysis-tab') {
                    pane.style.display = 'block';
                    pane.classList.add('active');
                } else {
                    pane.style.display = 'none';
                    pane.classList.remove('active');
                }
            });

            // 2. Disable trigger buttons
            triggerButtons.forEach(b => {
                b.disabled = true;
                b.style.opacity = '0.7';
                b.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...`;
            });

            // 3. Render inline loading indicator strictly inside Risk Assessment section
            if (analysisTab) {
                analysisTab.innerHTML = `
                    <div class="risk-loading-card" style="text-align: center; color: var(--text-muted); padding: 50px 24px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; background: rgba(99, 102, 241, 0.04); border: 1px dashed rgba(99, 102, 241, 0.3); border-radius: 16px; margin: 12px 0;">
                        <div style="position: relative; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center;">
                            <i class="fa-solid fa-wand-magic-sparkles" style="font-size: 28px; color: #a5b4fc;"></i>
                            <div style="position: absolute; top:0; left:0; width:64px; height:64px; border:3px solid rgba(99,102,241,0.15); border-top-color:#6366f1; border-radius:50%; animation:spin 1s linear infinite;"></div>
                        </div>
                        <div>
                            <h3 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 17px; margin: 0 0 6px 0; font-weight: 700;">AI Engine is Scanning Contract Risks...</h3>
                            <p style="max-width: 400px; font-size: 13px; color: var(--text-secondary); margin: 0; line-height: 1.5;">Extracting agreement clauses, verifying compliance against legal rules, and calculating risk score...</p>
                        </div>
                    </div>
                `;
            }

            try {
                const res = await fetch(`/api/contracts/${contractId}/analyze/`, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    }
                });
                const data = await res.json();
                
                if (data.success) {
                    showToast("AI analysis completed successfully!", "success");
                    if (window.refreshContractDetailViaAjax) {
                        window.refreshContractDetailViaAjax(contractId);
                    }
                } else {
                    const errorMsg = data.error || "Unknown error";
                    const isModelError = errorMsg.includes("503") || errorMsg.includes("communication failed") || errorMsg.includes("not loaded");
                    if (isModelError) {
                        showToast("Hệ thống chưa kết nối được với mô hình AI. Vui lòng tải mô hình hoặc kích hoạt GPU để tiến hành phân tích hợp đồng.", "warning");
                    } else {
                        showToast(errorMsg, "error");
                    }
                    if (window.refreshContractDetailViaAjax) {
                        window.refreshContractDetailViaAjax(contractId);
                    }
                }
            } catch (err) {
                console.error(err);
                showToast("Lỗi kết nối: Không thể gửi yêu cầu phân tích tới hệ thống.", "error");
            }
        });
    });

    // 4. Push to Workflow button
    const btnPushWorkflow = document.getElementById('btn-push-workflow');
    if (btnPushWorkflow) {
        btnPushWorkflow.addEventListener('click', async () => {
            const contractId = btnPushWorkflow.dataset.contractId;
            btnPushWorkflow.disabled = true;
            btnPushWorkflow.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Pushing...';

            try {
                const res = await fetch(`/api/contracts/${contractId}/workflow/`, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    }
                });
                const data = await res.json();

                if (data.success) {
                    showToast('Contract đã được đẩy lên workflow thành công! Đang tải lại...', 'success');
                    setTimeout(() => window.location.reload(), 1800);
                } else {
                    btnPushWorkflow.disabled = false;
                    btnPushWorkflow.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Push to Workflow';
                    const msg = data.error || 'Không thể đẩy lên workflow.';
                    showToast(msg.includes('already active') ? 'Workflow đã tồn tại cho contract này.' : msg, 'warning');
                }
            } catch (err) {
                btnPushWorkflow.disabled = false;
                btnPushWorkflow.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Push to Workflow';
                console.error(err);
                showToast('Lỗi kết nối: Không thể kết nối tới workflow service.', 'error');
            }
        });
    }

    // 4b. Re-push to Workflow button (Force recreate)
    const btnRepushWorkflow = document.getElementById('btn-repush-workflow');
    if (btnRepushWorkflow) {
        btnRepushWorkflow.addEventListener('click', async () => {
            if (!confirm('Bạn có chắc chắn muốn reset và tạo lại quy trình phê duyệt (Workflow) cho hợp đồng này không? Toàn bộ các bước ký kết và chữ ký cũ sẽ bị xóa.')) {
                return;
            }
            
            const contractId = btnRepushWorkflow.dataset.contractId;
            btnRepushWorkflow.disabled = true;
            btnRepushWorkflow.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Resetting...';

            try {
                const res = await fetch(`/api/contracts/${contractId}/workflow/`, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': getCookie('csrftoken')
                    }
                });
                const data = await res.json();

                if (data.success) {
                    showToast('Đã xóa và tái tạo workflow mới thành công! Đang tải lại...', 'success');
                    setTimeout(() => window.location.reload(), 1800);
                } else {
                    btnRepushWorkflow.disabled = false;
                    btnRepushWorkflow.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Re-push to Workflow';
                    showToast(data.error || 'Không thể tạo lại workflow.', 'warning');
                }
            } catch (err) {
                btnRepushWorkflow.disabled = false;
                btnRepushWorkflow.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Re-push to Workflow';
                console.error(err);
                showToast('Lỗi kết nối: Không thể kết nối tới workflow service.', 'error');
            }
        });
    }

    // 5. Auto-load workflow status
    const workflowContainer = document.getElementById('workflow-steps-container');
    if (workflowContainer) {
        const contractId = workflowContainer.dataset.contractId;
        (async () => {
            try {
                const res = await fetch(`/api/contracts/${contractId}/workflow/status/`);
                const data = await res.json();
                const wf = data.workflow;
                if (!wf) {
                    const pushContainer = document.getElementById('workflow-push-container');
                    const statusContainer = document.getElementById('workflow-status-container');
                    if (pushContainer && statusContainer) {
                        pushContainer.style.display = 'block';
                        statusContainer.style.display = 'none';
                    } else {
                        workflowContainer.innerHTML = '<p style="color:var(--text-muted);font-size:13px;">Chưa có workflow nào được tạo.</p>';
                    }
                    return;
                }
                const statusColor = { PENDING: '#6366f1', IN_PROGRESS: '#f59e0b', COMPLETED: '#10b981', REJECTED: '#ef4444' };
                const stepStatusIcon = { PENDING: 'fa-clock', APPROVED: 'fa-circle-check', REJECTED: 'fa-circle-xmark' };
                const stepStatusColor = { PENDING: '#6b7280', APPROVED: '#10b981', REJECTED: '#ef4444' };

                workflowContainer.innerHTML = `
                    <div style="margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;gap:8px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:12px;font-weight:600;background:${statusColor[wf.status]||'#6366f1'}22;color:${statusColor[wf.status]||'#6366f1'};padding:4px 10px;border-radius:20px;">${wf.status}</span>
                            <span style="font-size:12px;color:var(--text-muted);">${wf.workflow_name}</span>
                        </div>
                        <a href="/board/${wf.workflow_id}/" target="_blank" style="font-size:11.5px;color:#a5b4fc;text-decoration:none;font-weight:600;background:rgba(99,102,241,0.12);padding:4px 10px;border-radius:6px;border:1px solid rgba(99,102,241,0.25);display:inline-flex;align-items:center;gap:4px;" title="Open standalone Workflow Detail page">
                            <i class="fa-solid fa-route"></i> Workflow Detail <i class="fa-solid fa-up-right-from-square" style="font-size:10px;"></i>
                        </a>
                    </div>
                    <div style="display:flex;flex-direction:column;gap:8px;">
                        ${wf.steps.map(st => `
                        <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid rgba(255,255,255,0.07);">
                            <i class="fa-solid ${stepStatusIcon[st.status]||'fa-clock'}" style="color:${stepStatusColor[st.status]||'#6b7280'};font-size:15px;"></i>
                            <div>
                                <div style="font-size:13px;font-weight:600;color:#fff;">${st.step_order}. ${st.step_name}</div>
                                <div style="font-size:11.5px;color:var(--text-muted);">${st.status}${st.completed_at ? ' · ' + new Date(st.completed_at).toLocaleDateString('vi-VN') : ''}</div>
                            </div>
                        </div>`).join('')}
                    </div>
                `;
            } catch (e) {
                workflowContainer.innerHTML = '<p style="color:#ef4444;font-size:13px;">Không tải được trạng thái workflow.</p>';
            }
        })();
    }

    // 6. Version switcher dropdown event
    const versionSelect = document.getElementById('version-select');
    if (versionSelect) {
        versionSelect.addEventListener('change', (e) => {
            const versionId = e.target.value;
            const contractId = window.location.pathname.split('/')[2];
            if (window.refreshContractDetailViaAjax) {
                window.refreshContractDetailViaAjax(contractId, versionId);
            } else {
                const url = new URL(window.location.href);
                url.searchParams.set('version_id', versionId);
                window.location.href = url.toString();
            }
        });
    }

    // 5. Upload new version modal logic
    const btnUploadNew = document.getElementById('btn-upload-new-version');
    const uploadModal = document.getElementById('upload-version-modal');
    const btnCloseModal = document.getElementById('modal-close-btn');
    const btnCancelUpload = document.getElementById('btn-cancel-upload');
    
    if (btnUploadNew && uploadModal) {
        // Open Modal
        btnUploadNew.addEventListener('click', () => {
            uploadModal.classList.add('active');
        });
        
        // Close Modal helpers
        const closeModal = () => {
            uploadModal.classList.remove('active');
            document.getElementById('upload-version-form').reset();
            const selectedFile = document.getElementById('selected-file-name');
            if (selectedFile) {
                selectedFile.style.display = 'none';
                selectedFile.innerHTML = '';
            }
        };
        
        if (btnCloseModal) btnCloseModal.addEventListener('click', closeModal);
        if (btnCancelUpload) btnCancelUpload.addEventListener('click', closeModal);
        
        // Close on clicking overlay background
        uploadModal.addEventListener('click', (e) => {
            if (e.target === uploadModal) {
                closeModal();
            }
        });
    }

    // Tab switcher in upload modal
    const tabBtns = document.querySelectorAll('.modal-tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => {
                b.classList.remove('active');
                b.style.color = 'var(--text-muted)';
            });
            btn.classList.add('active');
            btn.style.color = '#ffffff';
            
            const targetTab = btn.dataset.tab;
            const uploadModal = document.getElementById('upload-version-modal');
            const panes = uploadModal ? uploadModal.querySelectorAll('.tab-pane') : document.querySelectorAll('#upload-version-modal .tab-pane');
            panes.forEach(pane => {
                if (pane.id === targetTab) {
                    pane.style.display = 'block';
                    pane.classList.add('active');
                } else {
                    pane.style.display = 'none';
                    pane.classList.remove('active');
                }
            });
        });
    });

    // File selection handling
    const fileZone = document.getElementById('file-upload-zone');
    const fileInput = document.getElementById('version_file_input');
    const selectedFileName = document.getElementById('selected-file-name');
    
    if (fileZone && fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                selectedFileName.style.display = 'block';
                selectedFileName.innerHTML = `<i class="fa-solid fa-file-circle-check"></i> Selected: ${fileInput.files[0].name}`;
            } else {
                selectedFileName.style.display = 'none';
                selectedFileName.innerHTML = '';
            }
        });
        
        // Drag and drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            fileZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                fileZone.style.borderColor = 'var(--accent-primary)';
                fileZone.style.background = 'rgba(99, 102, 241, 0.08)';
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                fileZone.style.borderColor = 'rgba(99, 102, 241, 0.3)';
                fileZone.style.background = 'rgba(99, 102, 241, 0.02)';
            }, false);
        });
        
        fileZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                selectedFileName.style.display = 'block';
                selectedFileName.innerHTML = `<i class="fa-solid fa-file-circle-check"></i> Selected: ${files[0].name}`;
            }
        });
    }

    // Submit new version form via AJAX
    const uploadForm = document.getElementById('upload-version-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const contractId = uploadForm.dataset.contractId || (document.getElementById('btn-detail-reanalyze') || document.getElementById('btn-detail-trigger-analysis') || {}).dataset?.contractId;
            if (!contractId) {
                showToast("Không tìm thấy ID hợp đồng trên trang.", "error");
                return;
            }

            const activeTabPane = uploadForm.querySelector('.tab-pane.active');
            const isFileTab = activeTabPane && activeTabPane.id === 'file-tab';
            
            const formData = new FormData(uploadForm);
            if (isFileTab) {
                formData.delete('raw_content');
                if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                    showToast("Vui lòng chọn một tập tin hợp đồng (.pdf, .docx, .txt).", "warning");
                    return;
                }
            } else {
                formData.delete('file');
                const rawContentVal = (uploadForm.raw_content ? uploadForm.raw_content.value : '').trim();
                if (!rawContentVal) {
                    showToast("Vui lòng nhập nội dung văn bản hợp đồng.", "warning");
                    return;
                }
            }
            
            if (uploadModal) uploadModal.classList.remove('active');
            
            if (scannerLoader) {
                scannerLoader.classList.add('active');
                const scannerText = scannerLoader.querySelector('.scanner-text');
                if (scannerText) scannerText.innerHTML = "UPLOADING & SCANNING NEW VERSION...";
            }
            
            try {
                const res = await fetch(`/api/contracts/${contractId}/versions/`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                
                setTimeout(() => {
                    if (scannerLoader) scannerLoader.classList.remove('active');
                    if (data.success) {
                        showToast("Phiên bản mới đã được tải lên thành công!", "success");
                        if (window.refreshContractDetailViaAjax) {
                            window.refreshContractDetailViaAjax(contractId, data.version_id);
                        }
                    } else {
                        const errorMsg = data.error || "Unknown error";
                        const isModelError = errorMsg.includes("503") || errorMsg.includes("communication failed") || errorMsg.includes("not loaded");
                        if (isModelError) {
                            showToast("Phiên bản mới đã được tải lên, nhưng hệ thống chưa kết nối được với mô hình AI để phân tích ngay.", "warning");
                            if (window.refreshContractDetailViaAjax) {
                                window.refreshContractDetailViaAjax(contractId, data.version_id);
                            }
                        } else {
                            showToast(errorMsg, "error");
                        }
                    }
                }, 3000);
            } catch (err) {
                if (scannerLoader) scannerLoader.classList.remove('active');
                console.error(err);
                showToast("Lỗi kết nối: Không thể tải lên phiên bản mới.", "error");
            }
        });
    }

    // Panel Tabs Switcher
    const panelTabBtns = document.querySelectorAll('.panel-tab-btn');
    panelTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            panelTabBtns.forEach(b => {
                b.classList.remove('active');
                b.style.color = 'var(--text-muted)';
            });
            btn.classList.add('active');
            btn.style.color = '#ffffff';
            
            const targetTab = btn.dataset.target;
            const panes = document.querySelectorAll('.tab-content-wrapper > .tab-pane');
            panes.forEach(pane => {
                if (pane.id === targetTab) {
                    pane.style.display = 'block';
                    pane.classList.add('active');
                } else {
                    pane.style.display = 'none';
                    pane.classList.remove('active');
                }
            });
        });
    });

    // Manual / AI Extraction API Handlers
    const btnManualExtract = document.getElementById('btn-manual-extract');
    const btnAiExtract = document.getElementById('btn-ai-extract');
    const vSelect = document.getElementById('version-select');
    
    if (btnManualExtract) {
        btnManualExtract.addEventListener('click', async () => {
            const contractId = btnManualExtract.dataset.contractId;
            const versionId = vSelect ? vSelect.value : null;
            
            if (scannerLoader) {
                scannerLoader.classList.add('active');
                const scannerText = scannerLoader.querySelector('.scanner-text');
                if (scannerText) scannerText.innerHTML = "RUNNING MANUAL EXTRACTION...";
            }
            
            try {
                const res = await fetch(`/api/contracts/${contractId}/manual-extract/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ version_id: versionId })
                });
                const data = await res.json();
                
                setTimeout(() => {
                     if (scannerLoader) scannerLoader.classList.remove('active');
                     if (data.success) {
                         showToast("Trích xuất thủ công hoàn tất!", "success");
                         if (window.refreshContractDetailViaAjax) window.refreshContractDetailViaAjax(contractId, versionId);
                     } else {
                         showToast(data.error || "Lỗi khi trích xuất thủ công.", "error");
                     }
                }, 1000);
            } catch (err) {
                if (scannerLoader) scannerLoader.classList.remove('active');
                console.error(err);
                showToast("Lỗi kết nối khi trích xuất thủ công.", "error");
            }
        });
    }
    
    if (btnAiExtract) {
        btnAiExtract.addEventListener('click', async () => {
            const contractId = btnAiExtract.dataset.contractId;
            const versionId = vSelect ? vSelect.value : null;
            
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
                    body: JSON.stringify({ version_id: versionId, re_extract: true })
                });
                const data = await res.json();
                
                setTimeout(() => {
                     if (scannerLoader) scannerLoader.classList.remove('active');
                     if (data.version_id) {
                         showToast("Trích xuất AI hoàn tất!", "success");
                         if (window.refreshContractDetailViaAjax) window.refreshContractDetailViaAjax(contractId, versionId);
                     } else {
                         showToast(data.error || "Lỗi khi trích xuất AI.", "error");
                     }
                }, 1000);
            } catch (err) {
                if (scannerLoader) scannerLoader.classList.remove('active');
                console.error(err);
                showToast("Lỗi kết nối khi trích xuất AI.", "error");
            }
        });
    }

    // 9. Generate AI Summary click handler
    const btnGenerateSummary = document.getElementById('btn-generate-summary');
    if (btnGenerateSummary) {
        btnGenerateSummary.addEventListener('click', async () => {
            const contractId = btnGenerateSummary.dataset.contractId;
            const versionSelect = document.getElementById('version-select');
            const versionId = versionSelect ? versionSelect.value : null;
            
            if (scannerLoader) {
                scannerLoader.classList.add('active');
                const scannerText = scannerLoader.querySelector('.scanner-text');
                if (scannerText) scannerText.innerHTML = "GENERATING AI EXECUTIVE SUMMARY...";
            }
            
            try {
                const res = await fetch(`/api/ai/contracts/${contractId}/summarize/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCookie('csrftoken')
                    },
                    body: JSON.stringify({ version_id: versionId })
                });
                const data = await res.json();
                
                setTimeout(() => {
                     if (scannerLoader) scannerLoader.classList.remove('active');
                     if (data.summary) {
                         showToast("Tóm tắt AI hoàn tất!", "success");
                         if (window.refreshContractDetailViaAjax) window.refreshContractDetailViaAjax(contractId, versionId);
                     } else {
                         showToast(data.error || "Lỗi khi tạo tóm tắt AI.", "error");
                     }
                }, 1000);
            } catch (err) {
                if (scannerLoader) scannerLoader.classList.remove('active');
                console.error(err);
                showToast("Lỗi kết nối khi tạo tóm tắt AI.", "error");
            }
        });
    }
});

// Toggle folding/unfolding of clauses
window.toggleClause = function(header) {
    const item = header.parentElement;
    item.classList.toggle('active');
    const body = item.querySelector('.clause-item-body');
    const icon = header.querySelector('i');
    if (item.classList.contains('active')) {
        body.style.display = 'block';
        icon.style.transform = 'rotate(180deg)';
    } else {
        body.style.display = 'none';
        icon.style.transform = 'rotate(0deg)';
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
            .toast-card.success {
                border-left-color: #10b981;
                border-color: rgba(16, 185, 129, 0.2);
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
    
    const iconClass = type === 'warning' ? 'fa-solid fa-triangle-exclamation' : type === 'success' ? 'fa-solid fa-circle-check' : 'fa-solid fa-circle-exclamation';
    const titleText = type === 'warning' ? 'Cảnh Báo Hệ Thống' : type === 'success' ? 'Thành Công' : 'Lỗi Hệ Thống';

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

// Toggle clause body (Global helper)
window.toggleClause = function(header) {
    const item = header.parentElement;
    const body = item.querySelector('.clause-item-body');
    const icon = header.querySelector('i.fa-chevron-down');
    if (body) {
        if (body.style.display === 'none' || !body.style.display) {
            body.style.display = 'block';
            if (icon) icon.style.transform = 'rotate(180deg)';
        } else {
            body.style.display = 'none';
            if (icon) icon.style.transform = 'rotate(0deg)';
        }
    }
};

// Pure AJAX refresh of Contract Detail page without full browser reload
window.refreshContractDetailViaAjax = async function(contractId, versionId = null) {
    const apiUrl = versionId ? `/api/contracts/${contractId}/?version_id=${versionId}` : `/api/contracts/${contractId}/`;
    try {
        const res = await fetch(apiUrl);
        if (!res.ok) return;
        const contract = await res.json();
        
        if (versionId) {
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('version_id', versionId);
            window.history.pushState({}, '', currentUrl.toString());
        }

        // 1. Update Version Select dropdown options
        const versionSelect = document.getElementById('version-select');
        if (versionSelect && contract.versions) {
            versionSelect.innerHTML = contract.versions.map(v => `
                <option value="${v.id}" ${v.id == contract.active_version_id ? 'selected' : ''}>
                    v${v.version_number} - ${(v.change_summary || '').substring(0, 20)} (${v.overall_score ? Math.round(v.overall_score) + '%' : 'Pending'})
                </option>
            `).join('');
        }

        // 2. Update Status badge
        const statusBadges = document.querySelectorAll('.status-badge');
        statusBadges.forEach(badge => {
            badge.className = `status-badge status-${(contract.status || '').toLowerCase()}`;
            badge.textContent = contract.status;
        });

        // 3. Update Document raw content
        const docPre = document.querySelector('.doc-content-pre');
        if (docPre) docPre.textContent = contract.raw_content || '';

        // 4. Update Active Version Info Card
        const activeVersionInfo = document.querySelector('.active-version-info-card');
        if (activeVersionInfo) {
            activeVersionInfo.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #ffffff; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                        <i class="fa-solid fa-code-branch" style="color: var(--accent-primary);"></i> Viewing Version ${contract.active_version_number || 1}
                    </span>
                    <span style="font-size: 11.5px; color: var(--text-muted);">
                        Status: ${contract.status}
                    </span>
                </div>
                ${contract.active_version_change_summary ? `
                <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 8px; padding-top: 8px;">
                    <strong style="color: var(--accent-primary);">Change Log:</strong> ${contract.active_version_change_summary}
                </div>` : ''}
            `;
        }

        // 5. Update Metadata cards (Code, Type, Value)
        const metaVals = document.querySelectorAll('.meta-val');
        if (metaVals.length >= 3) {
            metaVals[0].textContent = contract.contract_code || 'N/A';
            metaVals[1].textContent = contract.contract_type || 'N/A';
            metaVals[2].textContent = '$' + (contract.contract_value ? Number(contract.contract_value).toLocaleString('en-US', {minimumFractionDigits: 2}) : '0.00');
        }

        // 6. Update Gauge & Risk Score
        const gauge = document.getElementById('gauge-val');
        const gaugeText = document.querySelector('.gauge-text');
        const gaugeStatus = document.getElementById('gauge-status');
        if (gauge && contract.analysis && contract.analysis.overall_score !== undefined) {
            const score = parseFloat(contract.analysis.overall_score);
            gauge.dataset.score = score;
            if (gaugeText) gaugeText.textContent = Math.round(score) + '%';
            const r = 36;
            const circumference = 2 * Math.PI * r;
            const offset = circumference - (score / 100) * circumference;
            gauge.style.strokeDashoffset = offset;
            if (score >= 80) {
                gauge.style.stroke = '#ef4444';
                if (gaugeStatus) { gaugeStatus.innerHTML = 'High Risk'; gaugeStatus.className = 'gauge-status score-high'; }
            } else if (score >= 50) {
                gauge.style.stroke = '#f97316';
                if (gaugeStatus) { gaugeStatus.innerHTML = 'Medium Risk'; gaugeStatus.className = 'gauge-status score-medium'; }
            } else {
                gauge.style.stroke = '#10b981';
                if (gaugeStatus) { gaugeStatus.innerHTML = 'Low Risk'; gaugeStatus.className = 'gauge-status score-low'; }
            }
        }

        // 7. Update AI Summary Box
        const summaryBox = document.querySelector('.summary-box');
        if (summaryBox && contract.analysis && contract.analysis.summary) {
            summaryBox.textContent = contract.analysis.summary;
        }

        // 8. Update Findings List
        const findingsList = document.querySelector('.findings-list');
        if (findingsList) {
            if (contract.findings && contract.findings.length > 0) {
                findingsList.innerHTML = contract.findings.map(f => `
                    <div class="finding-item">
                        <div class="finding-header" onclick="toggleFinding(this)">
                            <div class="finding-left">
                                <span class="risk-badge badge-${(f.risk_level || '').toLowerCase()}">${f.risk_level}</span>
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
                `).join('');
            } else {
                findingsList.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">No specific risks identified.</p>';
            }
        }

        // 9. Update Clauses List
        const clausesList = document.querySelector('.clauses-list');
        if (clausesList) {
            if (contract.clauses && contract.clauses.length > 0) {
                clausesList.innerHTML = contract.clauses.map(cl => `
                    <div class="clause-item-container" style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; overflow: hidden; transition: all 0.2s;">
                        <div class="clause-item-header" style="padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleClause(this)">
                            <span style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 14px; color: #ffffff;">${cl.title}</span>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 11px; background: rgba(99,102,241,0.15); color: #818cf8; padding: 2px 8px; border-radius: 12px;">${cl.entities ? cl.entities.length : 0} entities</span>
                                <i class="fa-solid fa-chevron-down" style="color: var(--text-muted); font-size: 12px; transition: transform 0.2s;"></i>
                            </div>
                        </div>
                        <div class="clause-item-body" style="display: none; padding: 16px; border-top: 1px solid rgba(255,255,255,0.04); background: rgba(0,0,0,0.15);">
                            <p style="font-size: 13px; line-height: 1.6; color: var(--text-secondary); white-space: pre-wrap; margin-top: 0; margin-bottom: 16px;">${cl.content}</p>
                            ${cl.entities && cl.entities.length > 0 ? `
                            <div style="border-top: 1px dashed rgba(255,255,255,0.08); padding-top: 12px;">
                                <h5 style="margin: 0 0 10px 0; font-family: 'Outfit', sans-serif; font-size: 11px; text-transform: uppercase; color: var(--accent-primary); letter-spacing: 0.5px;">Extracted Entities</h5>
                                <div style="display: flex; flex-direction: column; gap: 8px;">
                                    ${cl.entities.map(ee => `
                                    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.04);">
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span class="entity-type-badge entity-type-${(ee.entity_type || '').toLowerCase()}">${ee.entity_type}</span>
                                            <span style="font-size: 12.5px; color: #ffffff; font-weight: 500;">${ee.entity_value}</span>
                                        </div>
                                        <span style="font-size: 11px; color: var(--text-muted);">Conf: ${ee.confidence_score ? Number(ee.confidence_score).toFixed(2) : '1.00'}</span>
                                    </div>
                                    `).join('')}
                                </div>
                            </div>` : '<p style="font-size: 12px; color: var(--text-muted); font-style: italic; margin: 0;">No entities extracted for this clause.</p>'}
                        </div>
                    </div>
                `).join('');
            } else {
                clausesList.innerHTML = '<p style="color: var(--text-muted); font-style: italic; text-align: center; padding: 20px 0;">No clauses extracted yet.</p>';
            }
        }

        // 10. Update Executive Summary Tab Text
        const aiSummaryText = document.getElementById('ai-summary-text');
        if (aiSummaryText && contract.ai_summary) {
            aiSummaryText.textContent = contract.ai_summary.summary || '';
        }
    } catch (err) {
        console.error("AJAX contract refresh error:", err);
    }
};

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
