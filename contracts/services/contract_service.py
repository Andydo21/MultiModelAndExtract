from contracts.repositories import contract_data_repository as data_repository
import os
from decimal import Decimal
from django.core.files.base import ContentFile
from contracts.repositories import ContractRepository, ContractFileRepository, ClauseRepository, RiskRepository, AIAnalysisRepository, RiskFindingRepository, ReviewRepository, AuditLogRepository

class ContractService:

    def __init__(self):
        self.contract_repo = ContractRepository()
        self.file_repo = ContractFileRepository()
        self.clause_repo = ClauseRepository()
        self.risk_repo = RiskRepository()
        self.analysis_repo = AIAnalysisRepository()
        self.finding_repo = RiskFindingRepository()
        self.review_repo = ReviewRepository()
        self.audit_repo = AuditLogRepository()

    def list_all_contracts(self, company=None):
        if company:
            contracts = self.contract_repo.get_contracts_by_company(company)
        else:
            contracts = self.contract_repo.get_all_contracts()
        data = []
        for c in contracts:
            latest_analysis = data_repository.first_ai_analyses_for_list_all_contracts(c)
            data.append({'id': c.id, 'contract_code': c.contract_code, 'title': c.title, 'contract_type': c.contract_type, 'start_date': c.start_date.isoformat() if c.start_date else None, 'end_date': c.end_date.isoformat() if c.end_date else None, 'contract_value': float(c.contract_value) if c.contract_value else None, 'status': c.status, 'risk_score': int(latest_analysis.overall_score) if latest_analysis and latest_analysis.overall_score is not None else None, 'created_at': c.created_at.isoformat()})
        return data

    def get_contract_details(self, contract_id, version_id=None):
        from django.conf import settings
        from contracts.services.workflow_service import WorkflowService
        c = self.contract_repo.get_contract_by_id(contract_id)
        if not c:
            return None
        from contracts.models import ContractVersion
        if version_id:
            try:
                version = data_repository.get_versions_for_get_contract_details(c, version_id)
            except Exception:
                version = c.latest_version
        else:
            version = c.latest_version
        if not version:
            version = data_repository.create_contract_version_for_get_contract_details(c)
        latest_file = data_repository.first_files_for_get_contract_details(version)
        latest_analysis = data_repository.first_ai_analyses_for_get_contract_details(version)
        if not latest_analysis:
            from decimal import Decimal
            latest_analysis = self.analysis_repo.create_analysis(contract_or_version=version, model_name='N/A', overall_score=Decimal('0.0'), summary='Chưa thực hiện phân tích AI.')
        clauses_data = []
        for cl in data_repository.all_ai_extract_clauses_for_get_contract_details(version):
            entities_list = [{'id': ee.id, 'entity_type': ee.entity_type, 'entity_value': ee.entity_value, 'confidence_score': float(ee.confidence_score)} for ee in data_repository.all_ai_extract_entities_for_get_contract_details(cl)]
            clauses_data.append({'id': cl.id, 'title': cl.clause_title, 'content': cl.clause_content, 'entities': entities_list})
        reviews_data = []
        for analysis in data_repository.all_ai_analyses_for_get_contract_details(version):
            for r in data_repository.all_reviews_for_get_contract_details(analysis):
                reviews_data.append({'id': r.id, 'comment': r.note, 'final_risk_level': r.decision, 'reviewer': r.user.username if r.user else 'Legal Expert', 'reviewed_at': r.reviewed_at.isoformat() if r.reviewed_at else ''})
        analysis_data = None
        findings_data = []
        if latest_analysis:
            analysis_data = {'id': latest_analysis.id, 'model_name': latest_analysis.model_name, 'overall_score': float(latest_analysis.overall_score), 'summary': latest_analysis.summary, 'created_at': latest_analysis.created_at.isoformat()}
            findings_data = [{'id': f.id, 'clause_id': f.clause.id if f.clause else None, 'clause_title': f.clause.clause_title if f.clause else '', 'risk_name': f.rule.rule_name, 'risk_level': f.risk_level, 'explanation': f.explanation, 'recommendation': f.recommendation, 'disadvantaged_party': f.disadvantaged_party} for f in data_repository.all_findings_for_get_contract_details(latest_analysis)]
        raw_content = ''
        from contracts.models import ContractContext
        contexts = data_repository.order_by_contexts_for_get_contract_details(version)
        if contexts.exists():
            raw_content = '\n\n'.join([ctx.content for ctx in contexts])
        if not raw_content and latest_file and latest_file.file_path:
            media_prefix = settings.MEDIA_URL
            import urllib.parse
            rel_path = urllib.parse.unquote(latest_file.file_path)
            if rel_path.startswith(media_prefix):
                rel_path = rel_path[len(media_prefix):]
            full_path = os.path.join(settings.MEDIA_ROOT, rel_path.replace('/', os.sep))
            if os.path.exists(full_path):
                try:
                    ext = os.path.splitext(latest_file.file_name)[1].lower() if latest_file.file_name else ''
                    if ext == '.txt':
                        from contracts.services.document_encryption_service import decrypt_pdf
                        with open(full_path, 'rb') as f:
                            encrypted_bytes = f.read()
                        decrypted_bytes = decrypt_pdf(encrypted_bytes)
                        raw_content = decrypted_bytes.decode('utf-8', errors='ignore')
                    elif ext == '.pdf':
                        from contracts.services.document_encryption_service import decrypt_pdf
                        import fitz
                        with open(full_path, 'rb') as f:
                            encrypted_bytes = f.read()
                        decrypted_bytes = decrypt_pdf(encrypted_bytes)
                        doc = fitz.open(stream=decrypted_bytes, filetype='pdf')
                        text_list = []
                        for page in doc:
                            text_list.append(page.get_text())
                        raw_content = '\n\n'.join(text_list)
                        doc.close()
                except Exception:
                    pass
        if not raw_content and clauses_data:
            raw_content = '\n\n'.join([f"--- {cl['title']} ---\n{cl['content']}" for cl in clauses_data])
        versions_list = []
        for v in data_repository.order_by_versions_for_get_contract_details(c):
            v_analysis = data_repository.latest_analysis_for_listed_version(v)
            versions_list.append({'id': v.id, 'version_number': v.version_number, 'change_summary': v.change_summary or 'Initial version', 'created_at': v.created_at.isoformat(), 'overall_score': float(v_analysis.overall_score) if v_analysis and v_analysis.overall_score is not None else None, 'risk_level': v_analysis.risk_level if v_analysis else 'NONE'})
        from ai_extract.models import ContractSummary
        summary_obj = data_repository.first_contract_summary_for_get_contract_details(version)
        summary_data = {'id': summary_obj.id, 'summary': summary_obj.summary, 'model_id': summary_obj.model_id, 'created_at': summary_obj.created_at.isoformat(), 'updated_at': summary_obj.updated_at.isoformat()} if summary_obj else None
        has_workflow = False
        workflow_id = None
        try:
            wf_status = WorkflowService().get_workflow_status(c.id, version_id=version.id)
            if wf_status and wf_status.get('workflow_id'):
                has_workflow = True
                workflow_id = wf_status.get('workflow_id')
        except Exception:
            pass
        return {'id': c.id, 'contract_code': c.contract_code, 'title': c.title, 'contract_type': c.contract_type, 'start_date': c.start_date.isoformat() if c.start_date else None, 'end_date': c.end_date.isoformat() if c.end_date else None, 'contract_value': float(c.contract_value) if c.contract_value else None, 'status': c.status, 'file_path': f'/api/contracts/{c.id}/download-pdf/?version_id={version.id}', 'raw_content': raw_content, 'clauses': clauses_data, 'analysis': analysis_data, 'findings': findings_data, 'reviews': reviews_data, 'active_version_id': version.id, 'active_version_number': version.version_number, 'active_version_change_summary': version.change_summary, 'versions': versions_list, 'ai_summary': summary_data, 'has_workflow': has_workflow, 'workflow_id': workflow_id}

    def create_and_analyze_contract(self, code, title, contract_type, start_date, end_date, contract_value, file_obj=None, raw_content=None, company=None):
        if not code or not title:
            raise ValueError('Contract code and title are required.')
        contract = self.contract_repo.create_contract(code=code, title=title, contract_type=contract_type, start_date=start_date, end_date=end_date, contract_value=contract_value, status='DRAFT', company=company)
        from contracts.models import ContractVersion
        version = data_repository.create_contract_version_for_create_and_analyze_contract(contract)
        saved_file_path = None
        from django.core.files.base import ContentFile
        from contracts.services.document_encryption_service import encrypt_pdf
        if file_obj:
            from django.core.files.storage import FileSystemStorage
            fs = FileSystemStorage()
            file_data = file_obj.read()
            encrypted_data = encrypt_pdf(file_data)
            filename = fs.save(f'contracts/{file_obj.name}', ContentFile(encrypted_data))
            saved_file_path = fs.url(filename)
        elif raw_content:
            from django.core.files.storage import FileSystemStorage
            fs = FileSystemStorage()
            safe_code = ''.join((x for x in code if x.isalnum() or x in '-_'))
            encrypted_data = encrypt_pdf(raw_content.encode('utf-8'))
            filename = fs.save(f'contracts/contract_{safe_code}.txt', ContentFile(encrypted_data))
            saved_file_path = fs.url(filename)
        if saved_file_path:
            self.file_repo.create_file_record(version, saved_file_path)
        return contract

    def create_new_version(self, contract_id, file_obj=None, raw_content=None, change_summary=''):
        contract = self.contract_repo.get_contract_by_id(contract_id)
        if not contract:
            raise ValueError('Contract not found.')
        latest = contract.latest_version
        next_num = latest.version_number + 1 if latest else 1
        from contracts.models import ContractVersion
        version = data_repository.create_contract_version_for_create_new_version(change_summary, contract, next_num)
        saved_file_path = None
        from django.core.files.base import ContentFile
        from contracts.services.document_encryption_service import encrypt_pdf
        if file_obj:
            from django.core.files.storage import FileSystemStorage
            fs = FileSystemStorage()
            file_data = file_obj.read()
            encrypted_data = encrypt_pdf(file_data)
            filename = fs.save(f'contracts/{file_obj.name}', ContentFile(encrypted_data))
            saved_file_path = fs.url(filename)
        elif raw_content:
            from django.core.files.storage import FileSystemStorage
            fs = FileSystemStorage()
            safe_code = ''.join((x for x in contract.contract_code if x.isalnum() or x in '-_'))
            encrypted_data = encrypt_pdf(raw_content.encode('utf-8'))
            filename = fs.save(f'contracts/contract_{safe_code}_v{next_num}.txt', ContentFile(encrypted_data))
            saved_file_path = fs.url(filename)
        if saved_file_path:
            self.file_repo.create_file_record(version, saved_file_path)
        contract.status = 'DRAFT'
        data_repository.save_contract_for_create_new_version(contract)
        return version

    def analyze_contract(self, contract_id, version_id=None):
        contract = self.contract_repo.get_contract_by_id(contract_id)
        if not contract:
            raise ValueError('Contract not found.')
        contract.status = 'ANALYZING'
        data_repository.save_contract_for_create_new_version(contract)
        from contracts.models import ContractVersion
        if version_id:
            try:
                version = data_repository.get_versions_for_analyze_contract(contract, version_id)
            except ContractVersion.DoesNotExist:
                raise ValueError('Version not found.')
        else:
            version = contract.latest_version
            if not version:
                version = data_repository.create_contract_version_for_create_and_analyze_contract(contract)
        data_repository.delete_ai_analyses_for_analyze_contract(version)
        self._run_ai_analysis_via_api(contract, version)
        return contract

    def manual_extract_contract(self, contract_id, version_id=None):
        contract = self.contract_repo.get_contract_by_id(contract_id)
        if not contract:
            raise ValueError('Contract not found.')
        from contracts.models import ContractVersion
        if version_id:
            try:
                version = data_repository.get_versions_for_analyze_contract(contract, version_id)
            except ContractVersion.DoesNotExist:
                raise ValueError('Version not found.')
        else:
            version = contract.latest_version
            if not version:
                version = data_repository.create_contract_version_for_create_and_analyze_contract(contract)
        from ai_extract.models import Clause
        data_repository.delete_clause_for_manual_extract_contract(version)
        self.extract_clauses_from_a3(version)
        for cl in data_repository.filter_clause_for_manual_extract_contract(version):
            self._extract_and_save_basic_entities(cl)
        contract.status = 'DRAFT'
        data_repository.save_contract_for_create_new_version(contract)
        return contract

    def submit_expert_review(self, analysis_id, comment, final_risk_level):
        analysis = self.analysis_repo.get_analysis_by_id(analysis_id)
        if not analysis:
            raise ValueError('Analysis not found.')
        if final_risk_level not in ['LOW', 'MEDIUM', 'HIGH']:
            raise ValueError('Invalid risk level.')
        admin_user = data_repository.first_user_for_submit_expert_review()
        review = self.review_repo.create_review(analysis, admin_user, comment, final_risk_level)
        contract = analysis.version.contract
        contract.status = 'APPROVED'
        data_repository.save_contract_for_create_new_version(contract)
        self.audit_repo.log_action(admin_user, 'REVIEW_SUBMITTED', 'Review', review.id)
        return review

    def extract_clauses_from_a3(self, version):
        """
        Delegates document processing, text extraction, page saving, and clause splitting
        to the unified ClauseExtractService.
        """
        from ai_extract.services import ClauseExtractService
        extractor = ClauseExtractService()
        extractor.extract_version(version, re_extract=True)

    def _run_ai_analysis_via_api(self, contract, version):
        import requests
        from django.conf import settings
        from decimal import Decimal
        if not data_repository.exists_ai_extract_clauses_for_run_ai_analysis_via_api(version):
            self.extract_clauses_from_a3(version)
        clauses = list(data_repository.all_ai_extract_clauses_for_get_contract_details(version))
        if not clauses:
            raise ValueError('No clauses found in contract to analyze.')
        for cl in clauses:
            if not data_repository.exists_ai_extract_entities_for_run_ai_analysis_via_api(cl):
                self._extract_and_save_basic_entities(cl)
        from ai_extract.models import ExtractedEntity
        db_entities = data_repository.filter_extracted_entity_for_run_ai_analysis_via_api(clauses)
        existing_rules = self.risk_repo.get_all_risks()
        payload = {'clauses': [{'title': cl.clause_title, 'content': cl.clause_content} for cl in clauses], 'extracted_entities': [{'clause_title': ee.clause.clause_title, 'entity_type': ee.entity_type, 'entity_value': ee.entity_value, 'normalized_value': ee.normalized_value or '', 'confidence_score': float(ee.confidence_score)} for ee in db_entities], 'risk_rules': [{'name': r.rule_name, 'description': r.description} for r in existing_rules]}
        try:
            from shared.jwt_middleware import get_system_auth_header
            response = requests.post(f'{settings.AI_SERVICE_URL}/api/v1/analyze', json=payload, headers=get_system_auth_header(), timeout=600)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            from django.db import connection
            connection.close()
            contract.status = 'DRAFT'
            data_repository.save_contract_for_create_new_version(contract)
            raise RuntimeError(f'AI Service communication failed: {e}')
        from django.db import connection
        connection.close()
        overall_score = Decimal(str(result.get('overall_score', 0)))
        summary = result.get('summary', 'Analysis completed.')
        analysis = self.analysis_repo.create_analysis(contract_or_version=version, model_name='Qwen2.5-3B-Instruct (Fine-tuned)', overall_score=overall_score, summary=summary)
        for cl in clauses:
            cl_title = (cl.clause_title or '').strip().lower()
            for finding in result.get('findings', []):
                api_title = (finding.get('clause_title') or '').strip().lower()
                if api_title == cl_title or (api_title and api_title in cl_title) or (cl_title and cl_title in api_title):
                    risk_name = finding.get('risk_category', 'Rủi ro chung')
                    risk_level = finding.get('risk_level', 'MEDIUM')
                    explanation = finding.get('explanation', '')
                    recommendation = finding.get('recommendation', '')
                    disadvantaged = finding.get('disadvantaged_party')
                    risk, _ = self.risk_repo.get_or_create_risk(risk_name, defaults={'description': f'Auto-created category for {risk_name}.', 'severity_level': risk_level})
                    self.finding_repo.create_finding(analysis=analysis, clause=cl, risk=risk, risk_level=risk_level, explanation=explanation, recommendation=recommendation, disadvantaged_party=disadvantaged)
            for entity in result.get('entities', []):
                api_title = (entity.get('clause_title') or '').strip().lower()
                if api_title == cl_title or (api_title and api_title in cl_title) or (cl_title and cl_title in api_title):
                    data_repository.get_or_create_extracted_entity_for_run_ai_analysis_via_api(Decimal, cl, entity)
        contract.status = 'ANALYZED'
        data_repository.save_contract_for_create_new_version(contract)

    def _extract_and_save_basic_entities(self, clause):
        from ai_extract.models import ExtractedEntity
        content_lower = clause.clause_content.lower()
        party_found = []
        if 'bên a' in content_lower:
            party_found.append('Bên A')
        if 'bên b' in content_lower:
            party_found.append('Bên B')
        if 'landlord' in content_lower or 'bên cho thuê' in content_lower:
            party_found.append('Bên cho thuê (Landlord)')
        if 'tenant' in content_lower or 'bên thuê' in content_lower:
            party_found.append('Bên thuê (Tenant)')
        if 'client' in content_lower or 'khách hàng' in content_lower:
            party_found.append('Khách hàng (Client)')
        if 'consultant' in content_lower or 'nhà tư vấn' in content_lower:
            party_found.append('Nhà tư vấn (Consultant)')
        if 'developer' in content_lower or 'nhà phát triển' in content_lower:
            party_found.append('Nhà phát triển (Developer)')
        if 'distributor' in content_lower or 'nhà phân phối' in content_lower:
            party_found.append('Nhà phân phối (Distributor)')
        if 'techvibe' in content_lower:
            party_found.append('Công ty TechVibe')
        if 'devcore' in content_lower:
            party_found.append('Công ty DevCore')
        if 'landmark' in content_lower:
            party_found.append('Landmark')
        if 'smartacademy' in content_lower:
            party_found.append('SmartAcademy')
        for p in party_found:
            data_repository.get_or_create_extracted_entity_for_extract_and_save_basic_entities(Decimal, clause, p)
        action_found = []
        if 'đơn phương chấm dứt' in content_lower or 'unilaterally terminate' in content_lower or 'terminate for convenience' in content_lower:
            action_found.append(('Đơn phương chấm dứt hợp đồng', 'TERMINATION'))
        if 'chậm thanh toán' in content_lower or 'chậm trả' in content_lower or 'late payment' in content_lower:
            action_found.append(('Chậm thanh toán nghĩa vụ tài chính', 'PAYMENT_DEFAULT'))
        if 'sở hữu trí tuệ' in content_lower or 'quyền tác giả' in content_lower or 'intellectual property' in content_lower or ('bản quyền' in content_lower):
            action_found.append(('Sở hữu trí tuệ và bản quyền tác giả', 'IP_OWNERSHIP'))
        if 'độc quyền' in content_lower or 'exclusivity' in content_lower or 'exclusive' in content_lower:
            action_found.append(('Cam kết độc quyền thương mại', 'EXCLUSIVITY'))
        if 'phạt vi phạm' in content_lower or 'penalty' in content_lower or 'forfeit' in content_lower or ('tiền cọc' in content_lower):
            action_found.append(('Áp dụng chế tài phạt vi phạm/tịch thu cọc', 'PENALTY'))
        if 'tranh chấp' in content_lower or 'dispute' in content_lower or 'arbitration' in content_lower or ('tòa án' in content_lower) or ('trọng tài' in content_lower):
            action_found.append(('Giải quyết tranh chấp phát sinh', 'DISPUTE_RESOLUTION'))
        for act_val, act_norm in action_found:
            data_repository.get_or_create_extracted_entity_for_extract_and_save_basic_entities_by_clause_and_entity_type_and_entity_value(Decimal, act_norm, act_val, clause)
