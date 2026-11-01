/**
 * Workflow Board Logic - Contract Guard AI
 * Handles real-time loading, filtering, and step-by-step approvals of contract workflows.
 */

// Global State
let allWorkflows = [];
let currentFilter = 'ALL';
let activeStepId = null;
let currentUserId = window.currentUserId || 1; // Default fallback user ID
let currentUserRole = window.currentUserRole || '';
let currentUserIsSuperuser = window.currentUserIsSuperuser || false;
let stepIdToDelete = null;

const roleIdToNameMap = {
    1: 'ADMIN',
    2: 'EXPERT',
    3: 'VIEWER',
    4: 'LEGAL_EXPERT',
    5: 'MANAGER',
    6: 'FINANCE',
    7: 'TECHNICAL',
    8: 'SECURITY',
    9: 'COMPLIANCE',
    10: 'PROCUREMENT',
    11: 'EXECUTIVE'
};

document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI Event Listeners
    initEventListeners();

    // Initial fetch of workflows
    fetchWorkflows();
});

/**
 * Register DOM event listeners
 */
function initEventListeners() {
    // Refresh button
    const btnRefresh = document.getElementById('btn-refresh');
    if (btnRefresh) {
        btnRefresh.addEventListener('click', () => {
            fetchWorkflows();
        });
    }

    // Filter Buttons
    const filterButtons = document.querySelectorAll('.filter-bar .filter-btn');
    filterButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Remove active class from all
            filterButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked
            e.currentTarget.classList.add('active');
            // Update filter state and render
            currentFilter = e.currentTarget.getAttribute('data-filter');
            renderWorkflows();
        });
    });

    // Modal Close
    const modalClose = document.getElementById('modal-close');
    const modalOverlay = document.getElementById('approve-modal');
    if (modalClose) {
        modalClose.addEventListener('click', closeModal);
    }
    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) closeModal();
        });
    }

    // Approve Action
    const btnApprove = document.getElementById('btn-approve');
    if (btnApprove) {
        btnApprove.addEventListener('click', () => submitReview('APPROVED'));
    }

    // Reject Action
    const btnReject = document.getElementById('btn-reject');
    if (btnReject) {
        btnReject.addEventListener('click', () => submitReview('REJECTED'));
    }

    // Force Complete Action
    const btnForceComplete = document.getElementById('btn-force-complete');
    if (btnForceComplete) {
        btnForceComplete.addEventListener('click', () => submitReview('FORCE_COMPLETE'));
    }

    // Insert Step Modal Listeners
    const modalInsertClose = document.getElementById('modal-insert-close');
    const btnInsertCancel = document.getElementById('btn-insert-cancel');
    const btnInsertSubmit = document.getElementById('btn-insert-submit');
    const modalInsertOverlay = document.getElementById('insert-step-modal');

    if (modalInsertClose) {
        modalInsertClose.addEventListener('click', closeInsertStepModal);
    }
    if (btnInsertCancel) {
        btnInsertCancel.addEventListener('click', closeInsertStepModal);
    }
    if (btnInsertSubmit) {
        btnInsertSubmit.addEventListener('click', submitInsertStep);
    }
    if (modalInsertOverlay) {
        modalInsertOverlay.addEventListener('click', (e) => {
            if (e.target === modalInsertOverlay) closeInsertStepModal();
        });
    }
    // Delete Step Confirmation Modal Listeners
    const modalDeleteClose = document.getElementById('modal-delete-close');
    const btnDeleteCancel = document.getElementById('btn-delete-cancel');
    const btnDeleteSubmit = document.getElementById('btn-delete-submit');
    const modalDeleteOverlay = document.getElementById('delete-confirm-modal');

    if (modalDeleteClose) {
        modalDeleteClose.addEventListener('click', closeDeleteConfirmModal);
    }
    if (btnDeleteCancel) {
        btnDeleteCancel.addEventListener('click', closeDeleteConfirmModal);
    }
    if (btnDeleteSubmit) {
        btnDeleteSubmit.addEventListener('click', submitDeleteWorkflowStep);
    }
    if (modalDeleteOverlay) {
        modalDeleteOverlay.addEventListener('click', (e) => {
            if (e.target === modalDeleteOverlay) closeDeleteConfirmModal();
        });
    }
}

/**
 * Fetch all workflows via Django proxy endpoint
 */
async function fetchWorkflows() {
    showLoading(true);
    try {
        const response = await fetch('/workflows/all/');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        allWorkflows = data.workflows || [];
        updateStats();
        try {
            renderWorkflows();
        } catch (renderErr) {
            console.error('Error rendering workflows UI:', renderErr);
        }
    } catch (error) {
        console.error('Failed to fetch workflows:', error);
        showToast('Error loading workflows. Please try again.', 'error');
        renderErrorState();
    } finally {
        showLoading(false);
    }
}

/**
 * Show / Hide loading spinner
 */
function showLoading(isLoading) {
    const loadingState = document.getElementById('loading-state');
    const grid = document.getElementById('workflow-grid');

    if (isLoading) {
        if (!loadingState && grid) {
            grid.innerHTML = `
                <div class="loading-state" id="loading-state">
                    <div class="spinner"></div>
                    <p>Loading workflows...</p>
                </div>
            `;
        }
    } else {
        if (loadingState) {
            loadingState.remove();
        }
    }
}

/**
 * Update the Topbar statistic counts
 */
function updateStats() {
    const total = allWorkflows.length;
    const pending = allWorkflows.filter(w => w.status === 'PENDING').length;
    const inProgress = allWorkflows.filter(w => w.status === 'IN_PROGRESS').length;
    const completed = allWorkflows.filter(w => w.status === 'COMPLETED').length;

    updateStatElement('count-total', total);
    updateStatElement('count-pending', pending);
    updateStatElement('count-progress', inProgress);
    updateStatElement('count-done', completed);
}

function updateStatElement(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.innerText = value;
    }
}

/**
 * Render workflows on grid according to filter
 */
function renderWorkflows() {
    const grid = document.getElementById('workflow-grid');
    if (!grid) return;

    // Apply Filter
    let filtered = allWorkflows;
    if (currentFilter !== 'ALL') {
        filtered = allWorkflows.filter(w => w.status === currentFilter);
    }

    if (filtered.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-folder-open empty-icon"></i>
                <h3>No Workflows Found</h3>
                <p>There are no workflows matching the selected status filter.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = filtered.map(w => {
        const steps = Array.isArray(w.steps) ? w.steps : [];
        const sortedSteps = [...steps].sort((a, b) => a.step_order - b.step_order);
        const activeStep = sortedSteps.find(s => s.status === 'PENDING');
        
        // Helper function for hybrid parallel/sequential gating + custom step dependencies
        const isStepReviewable = (s) => {
            if (s.status !== 'PENDING') return false;
            if (w.status !== 'PENDING' && w.status !== 'IN_PROGRESS') return false;

            // Check custom step dependencies (Thắt nối bước phụ thuộc)
            const activeDeps = w.dependencies || [];
            const blockingDeps = activeDeps.filter(d => d.dependent_step_id === s.id && (d.is_blocking || d.prerequisite_step_status !== 'APPROVED'));
            if (blockingDeps.length > 0) {
                return false; // Phụ thuộc vào bước chưa APPROVED
            }

            if (sortedSteps.length === 0) return false;

            const firstStep = sortedSteps[0];

            // 1. Initial Gate (Step 1): Always reviewable if PENDING
            if (s.id === firstStep.id) return true;

            // Step 1 MUST be APPROVED before any other step can start
            if (firstStep.status !== 'APPROVED') return false;

            const nameLower = (s.step_name || '').toLowerCase();
            const isManagerStep = (s.role_id === 5 || s.role_id === 11 || nameLower.includes('phê duyệt cấp quản lý') || nameLower.includes('ban điều hành') || nameLower.includes('giám đốc phê duyệt'));
            const isFinalSigningOrArchive = nameLower.includes('ký kết hợp đồng') || nameLower.includes('lưu trữ kho hồ sơ') || nameLower.includes('bàn giao bản cứng');

            // 2. Manager Approval: Requires all middle verification steps to be APPROVED
            if (isManagerStep) {
                const middleSteps = sortedSteps.filter(st => 
                    st.id !== firstStep.id && 
                    st.id !== s.id && 
                    !((st.step_name || '').toLowerCase().includes('ký kết hợp đồng')) &&
                    !((st.step_name || '').toLowerCase().includes('lưu trữ kho hồ sơ'))
                );
                return middleSteps.every(st => st.status === 'APPROVED');
            }

            // 3. Final Signing / Archive: Requires all previous steps to be APPROVED
            if (isFinalSigningOrArchive) {
                const indexInSorted = sortedSteps.findIndex(st => st.id === s.id);
                const previousSteps = sortedSteps.slice(0, indexInSorted);
                return previousSteps.every(st => st.status === 'APPROVED');
            }

            // 4. Middle Verification Tasks (Security, Finance, Legal, Technical, Procurement, etc.): PARALLEL execution!
            return true;
        };

        const dateStr = w.started_at ? new Date(w.started_at).toLocaleString('vi-VN') : 'N/A';
        const completedStr = w.completed_at ? new Date(w.completed_at).toLocaleString('vi-VN') : '';

        // Compute completion progress percentage
        const totalSteps = steps.length;
        const approvedSteps = steps.filter(s => s.status === 'APPROVED').length;
        const progressPercent = totalSteps > 0 ? Math.round((approvedSteps / totalSteps) * 100) : 0;

        return `
            <div class="wf-card status-${w.status}">
                <div class="wf-card-header">
                    <div>
                        <div class="wf-name" title="${w.workflow_name}">
                            <a href="/board/${w.workflow_id}/" style="color: #ffffff; font-size: 15px; font-weight: 700; text-decoration: none; transition: var(--transition);" onmouseover="this.style.color='var(--accent)'" onmouseout="this.style.color='#ffffff'">
                                <i class="fa-solid fa-diagram-project" style="color: var(--accent); margin-right: 6px;"></i>${w.workflow_name}
                            </a>
                        </div>
                        ${w.contract_title ? `
                            <div class="contract-code-title" style="font-size: 12.5px; font-weight: 600; color: #818cf8; margin-top: 4px;">
                                <i class="fa-solid fa-file-contract"></i> Contract: 
                                <a href="http://localhost:8000/contracts/${w.contract_id}/" style="color: inherit; text-decoration: underline;" title="View Contract Details">
                                    ${w.contract_title} ${w.contract_code ? `(${w.contract_code})` : ''}
                                </a>
                            </div>
                        ` : ''}
                    </div>
                    <span class="wf-status-badge badge-${w.status}">${w.status}</span>
                </div>

                <div class="wf-type-row">
                    <span class="wf-type-badge type-${(w.workflow_type || 'WF_GENERAL').toLowerCase()}">
                        <i class="fa-solid fa-robot"></i> ${w.workflow_type || 'WF_GENERAL'}
                    </span>
                </div>

                ${w.reasons ? `
                    <div class="wf-reasons-box">
                        <div class="wf-reasons-header">
                            <i class="fa-solid fa-brain"></i> AI Reasoning
                        </div>
                        <div class="wf-reasons-body">
                            ${(typeof w.reasons === 'string' ? w.reasons.split('\n') : []).map(r => `<div class="reason-item">${r}</div>`).join('')}
                        </div>
                    </div>
                ` : ''}

                <div class="wf-progress-track">
                    <div class="progress-bar-wrap">
                        <div class="progress-bar-fill" style="width: ${progressPercent}%"></div>
                    </div>
                    <div class="wf-meta">
                        <span>Progress: ${progressPercent}% (${approvedSteps}/${totalSteps})</span>
                        <span><i class="fa-regular fa-clock"></i> ${dateStr}</span>
                    </div>
                </div>
                
                <div class="wf-steps-list">
                    ${sortedSteps.map((s, idx) => {
            const activeDeps = w.dependencies || [];
            const blockingDeps = activeDeps.filter(d => d.dependent_step_id === s.id && (d.is_blocking || d.prerequisite_step_status !== 'APPROVED'));
            const isBlockedByDependency = blockingDeps.length > 0;
            const canReview = isStepReviewable(s);
            let statusClass = s.status.toLowerCase();
            let icon = '<i class="fa-regular fa-circle"></i>';
            if (s.status === 'APPROVED') {
                icon = '<i class="fa-solid fa-circle-check"></i>';
            } else if (s.status === 'REJECTED') {
                icon = '<i class="fa-solid fa-circle-xmark"></i>';
            } else if (isBlockedByDependency) {
                icon = '<i class="fa-solid fa-lock" style="color: #ef4444;"></i>';
                statusClass += ' locked-step';
            } else if (canReview) {
                icon = '<i class="fa-solid fa-bolt" style="color: #6366f1;"></i>';
                statusClass += ' active-step';
            }

            const requiredRoleName = roleIdToNameMap[s.role_id] || '';
            const isManagerOrAdmin = currentUserIsSuperuser || (currentUserRole === 'ADMIN') || (currentUserRole === 'MANAGER');
            // Allow active reviewers to review open parallel steps
            const hasPermission = true;

            let stepRowHtml = `
                            <div class="wf-step-row status-${statusClass}">
                                <div class="step-dot dot-${s.status}">${icon}</div>
                                <div class="step-info">
                                    <div class="step-name">${s.step_name}</div>
                                    <div class="step-status-text">
                                        Step ${s.step_order} • ${s.status}
                                        ${isManagerOrAdmin ? `
                                            • <select class="step-role-select" onchange="updateStepRole(${s.id}, this.value)">
                                                ${Object.entries(roleIdToNameMap).map(([id, name]) => `
                                                    <option value="${id}" ${s.role_id == id ? 'selected' : ''}>${name}</option>
                                                `).join('')}
                                            </select>
                                        ` : `• Role: ${requiredRoleName}`}
                                    </div>
                                    ${isBlockedByDependency ? `
                                        <div style="font-size: 11px; color: #fca5a5; background: rgba(239,68,68,0.12); padding: 3px 8px; border-radius: 6px; display: inline-flex; align-items: center; gap: 4px; border: 1px solid rgba(239,68,68,0.25); margin-top: 4px;">
                                            <i class="fa-solid fa-lock" style="font-size: 10px;"></i>
                                            Khóa bởi: <strong>Step ${blockingDeps[0].prerequisite_step_order}: ${blockingDeps[0].prerequisite_step_name}</strong> (Chưa xong)
                                        </div>
                                    ` : ''}
                                    ${(s.approvals && s.approvals.length > 0) ? `
                                        <div class="step-approvals-list" style="margin-top: 4px; font-size: 11px; color: var(--text-sec);">
                                            <i class="fa-solid fa-signature" style="font-size: 10px; color: var(--green);"></i> Signed: 
                                            ${[...new Set(s.approvals.map(app => `${app.username}${app.role_name ? ` (${app.role_name})` : ''}`))].join(', ')}
                                        </div>
                                    ` : ''}
                                </div>
                                ${canReview ? `
                                    <button class="btn-step-action" onclick="openReviewModal(${s.id}, '${s.step_name}', '${w.workflow_name}')">
                                        Review
                                    </button>
                                ` : ''}
                                 ${isManagerOrAdmin ? `
                                     <button class="btn-delete-step" onclick="openDeleteConfirmModal(${s.id}, '${s.step_name}')" title="Delete Step">
                                         <i class="fa-solid fa-trash-can"></i>
                                     </button>
                                 ` : ''}
                             </div>
                        `;

            // Add manual step insertion dividers
            let preDivider = '';
            if (isManagerOrAdmin && idx === 0) {
                preDivider = `
                                <div class="insert-step-divider">
                                    <button class="btn-insert-step" onclick="openInsertStepModal(${w.workflow_id}, 1)">
                                        <i class="fa-solid fa-circle-plus"></i> Insert Step Here
                                    </button>
                                </div>
                            `;
            }
            let postDivider = '';
            if (isManagerOrAdmin) {
                postDivider = `
                                <div class="insert-step-divider">
                                    <button class="btn-insert-step" onclick="openInsertStepModal(${w.workflow_id}, ${s.step_order + 1})">
                                        <i class="fa-solid fa-circle-plus"></i> Insert Step Here
                                    </button>
                                </div>
                            `;
            }

            return preDivider + stepRowHtml + postDivider;
        }).join('')}
                </div>

                ${!activeStep ? `
                    <div class="wf-card-footer">
                        <div class="completion-info">
                            ${w.status === 'COMPLETED' ? `
                                <span class="text-success" style="font-size: 12px; font-weight: 600;"><i class="fa-solid fa-circle-check"></i> Fully Approved on ${completedStr}</span>
                            ` : w.status === 'REJECTED' ? `
                                <span class="text-danger" style="font-size: 12px; font-weight: 600;"><i class="fa-solid fa-circle-xmark"></i> Rejected on ${completedStr}</span>
                            ` : `
                                <span class="text-muted" style="font-size: 12px;">Workflow waiting for activation</span>
                            `}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

/**
 * Render Error/Fallback State
 */
function renderErrorState() {
    const grid = document.getElementById('workflow-grid');
    if (!grid) return;
    grid.innerHTML = `
        <div class="empty-state error">
            <i class="fa-solid fa-triangle-exclamation empty-icon text-danger"></i>
            <h3>Connection Failed</h3>
            <p>Could not communicate with the Workflow microservice. Please verify the service is running locally on port 8003.</p>
            <button class="btn" style="margin-top: 16px; background: var(--accent-primary);" onclick="fetchWorkflows()">
                <i class="fa-solid fa-arrows-rotate"></i> Retry Connection
            </button>
        </div>
    `;
}

/**
 * Open the Approve / Reject Modal
 */
window.openReviewModal = function (stepId, stepName, workflowName) {
    activeStepId = stepId;

    const modalStep = document.getElementById('modal-step-name');
    const modalWf = document.getElementById('modal-workflow-name');
    const commentArea = document.getElementById('modal-comment');
    const modalOverlay = document.getElementById('approve-modal');

    if (modalStep) modalStep.innerText = stepName;
    if (modalWf) modalWf.innerText = workflowName;
    if (commentArea) commentArea.value = '';


    const isManagerOrAdmin = currentUserIsSuperuser || (currentUserRole === 'ADMIN') || (currentUserRole === 'MANAGER');
    const btnForceComplete = document.getElementById('btn-force-complete');
    if (btnForceComplete) {
        btnForceComplete.style.display = isManagerOrAdmin ? 'flex' : 'none';
    }

    if (modalOverlay) {
        modalOverlay.style.display = 'flex';
        setTimeout(() => modalOverlay.classList.add('active'), 10);
    }
};

/**
 * Close Modal
 */
function closeModal() {
    const modalOverlay = document.getElementById('approve-modal');
    if (modalOverlay) {
        modalOverlay.classList.remove('active');
        setTimeout(() => {
            modalOverlay.style.display = 'none';
            activeStepId = null;
        }, 300);
    }
}

/**
 * Submit Review status (APPROVED or REJECTED)
 */
async function submitReview(action) {
    if (!activeStepId) return;

    const comment = document.getElementById('modal-comment')?.value || '';
    const btnApprove = document.getElementById('btn-approve');
    const btnReject = document.getElementById('btn-reject');
    const btnForceComplete = document.getElementById('btn-force-complete');

    // Disable buttons during submission
    if (btnApprove) btnApprove.disabled = true;
    if (btnReject) btnReject.disabled = true;
    if (btnForceComplete) btnForceComplete.disabled = true;

    try {
        const response = await fetch(`/workflows/steps/${activeStepId}/approve/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                action: action,
                comment: comment,
                user_id: currentUserId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            let msg = `Step successfully Approved!`;
            if (action === 'REJECTED') msg = `Step successfully Rejected!`;
            else if (action === 'FORCE_COMPLETE') msg = `Step accepted and completed!`;
            showToast(msg, 'success');
            closeModal();
            fetchWorkflows(); // Refresh list & stats
        } else {
            showToast(data.error || 'Failed to submit review.', 'error');
        }
    } catch (error) {
        console.error('Error submitting step review:', error);
        showToast('Error communicating with server.', 'error');
    } finally {
        if (btnApprove) btnApprove.disabled = false;
        if (btnReject) btnReject.disabled = false;
        if (btnForceComplete) btnForceComplete.disabled = false;
    }
}

/**
 * Show animated toast notifications
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = '<i class="fa-solid fa-info-circle"></i>';
    if (type === 'success') icon = '<i class="fa-solid fa-circle-check"></i>';
    if (type === 'error') icon = '<i class="fa-solid fa-triangle-exclamation"></i>';

    toast.innerHTML = `
        ${icon}
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Auto remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

/**
 * Helper to fetch CSRF token from cookies
 */
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

/**
 * Update step role for manager
 */
window.updateStepRole = async function (stepId, roleId) {
    try {
        const response = await fetch(`/workflows/steps/${stepId}/update_role/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                role_id: parseInt(roleId)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
            showToast('Step role updated successfully!', 'success');
            // Find the step in allWorkflows local cache and update it
            for (let wf of allWorkflows) {
                let st = wf.steps.find(s => s.id === stepId);
                if (st) {
                    st.role_id = parseInt(roleId);
                    break;
                }
            }
            renderWorkflows();
        } else {
            showToast(data.error || 'Failed to update step role.', 'error');
        }
    } catch (error) {
        console.error('Error updating step role:', error);
        showToast('Error communicating with server.', 'error');
    }
};

// State for inserting step
let activeInsertWorkflowId = null;
let activeInsertTargetOrder = null;

/**
 * Open Insert Step Modal
 */
window.openInsertStepModal = function (workflowId, targetOrder) {
    activeInsertWorkflowId = workflowId;
    activeInsertTargetOrder = targetOrder;

    const modal = document.getElementById('insert-step-modal');
    const nameInput = document.getElementById('insert-step-name');
    const roleSelect = document.getElementById('insert-step-role');
    const descTextarea = document.getElementById('insert-step-description');

    if (nameInput) nameInput.value = '';
    if (roleSelect) roleSelect.value = '4'; // default: LEGAL_EXPERT
    if (descTextarea) descTextarea.value = '';

    if (modal) {
        modal.style.display = 'flex';
        setTimeout(() => modal.classList.add('active'), 10);
    }
};

/**
 * Close Insert Step Modal
 */
window.closeInsertStepModal = function () {
    const modal = document.getElementById('insert-step-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => {
            modal.style.display = 'none';
            activeInsertWorkflowId = null;
            activeInsertTargetOrder = null;
        }, 300);
    }
};

/**
 * Submit Insert Step request
 */
window.submitInsertStep = async function () {
    if (!activeInsertWorkflowId || activeInsertTargetOrder === null) return;

    const nameInput = document.getElementById('insert-step-name');
    const roleSelect = document.getElementById('insert-step-role');
    const descTextarea = document.getElementById('insert-step-description');

    const stepName = nameInput ? nameInput.value.trim() : '';
    if (!stepName) {
        showToast('Step name is required!', 'error');
        return;
    }

    const roleId = roleSelect ? parseInt(roleSelect.value) : null;
    const description = descTextarea ? descTextarea.value.trim() : '';

    const btnSubmit = document.getElementById('btn-insert-submit');
    if (btnSubmit) btnSubmit.disabled = true;

    try {
        const response = await fetch(`/workflows/${activeInsertWorkflowId}/insert_step/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                step_order: activeInsertTargetOrder,
                step_name: stepName,
                role_id: roleId,
                description: description
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
            showToast('Workflow step inserted successfully!', 'success');
            closeInsertStepModal();
            fetchWorkflows(); // Reload workflows board
        } else {
            showToast(data.error || 'Failed to insert step.', 'error');
        }
    } catch (error) {
        console.error('Error inserting workflow step:', error);
        showToast('Error communicating with server.', 'error');
    } finally {
        if (btnSubmit) btnSubmit.disabled = false;
    }
};

/**
 * Open and close delete step confirmation modal
 */
window.openDeleteConfirmModal = function (stepId, stepName) {
    stepIdToDelete = stepId;
    const modalText = document.getElementById('delete-modal-text');
    if (modalText) {
        modalText.textContent = `Are you sure you want to delete the step "${stepName}"? This action cannot be undone.`;
    }
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.add('active');
};

window.closeDeleteConfirmModal = function () {
    stepIdToDelete = null;
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('active');
};

/**
 * Submit step deletion to the proxy endpoint
 */
async function submitDeleteWorkflowStep() {
    if (!stepIdToDelete) return;
    const btnSubmit = document.getElementById('btn-delete-submit');
    if (btnSubmit) {
        btnSubmit.disabled = true;
        btnSubmit.textContent = 'Deleting...';
    }

    try {
        const response = await fetch(`/workflows/steps/${stepIdToDelete}/delete/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.success) {
            showToast('Workflow step deleted successfully!', 'success');
            closeDeleteConfirmModal();
            fetchWorkflows(); // Reload workflows board
        } else {
            showToast(data.error || 'Failed to delete step.', 'error');
        }
    } catch (error) {
        console.error('Error deleting workflow step:', error);
        showToast('Error communicating with server.', 'error');
    } finally {
        if (btnSubmit) {
            btnSubmit.disabled = false;
            btnSubmit.textContent = 'Delete Step';
        }
    }
}
