from contracts.repositories import document_download_data_repository as data_repository
from django.core.exceptions import ObjectDoesNotExist
from contracts.services import ContractService
from contracts.services.results import ServiceResult
from contracts.services.access_service import check_contract_permission as _check_contract_permission
contract_service = ContractService()

def download_file(file_id, user, content_type):
    from django.conf import settings
    import os
    from contracts.models import ContractFile
    from contracts.services.document_encryption_service import decrypt_pdf
    from contracts.services.pdf_rendering_service import generate_pdf_from_text
    try:
        cf = data_repository.get_contract_file_for_download_file(file_id)
        if not _check_contract_permission(user, cf.version.contract):
            return ServiceResult('Permission Denied: You do not have access to this file.', status=403, content_type='text/html')
    except ContractFile.DoesNotExist:
        raise ObjectDoesNotExist('File not found')
    rel_path = cf.file_path
    media_prefix = settings.MEDIA_URL
    if rel_path.startswith(media_prefix):
        rel_path = rel_path[len(media_prefix):]
    physical_path = os.path.join(settings.MEDIA_ROOT, rel_path.replace('/', os.sep))
    decrypted_data = None
    if os.path.exists(physical_path):
        try:
            with open(physical_path, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = decrypt_pdf(encrypted_data)
        except Exception:
            decrypted_data = None
    if decrypted_data is not None:
        if cf.file_name.lower().endswith('.pdf') and decrypted_data.startswith(b'%PDF'):
            response = ServiceResult(decrypted_data, content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{cf.file_name}"'
            return response
        else:
            import mimetypes
            content_type, _ = mimetypes.guess_type(cf.file_name)
            content_type = content_type or 'application/octet-stream'
            response = ServiceResult(decrypted_data, content_type=content_type)
            response['Content-Disposition'] = f'attachment; filename="{cf.file_name}"'
            return response
    else:
        details = contract_service.get_contract_details(cf.version.contract.id, version_id=cf.version.id) or {}
        text_str = details.get('raw_content') or cf.version.contract.title or ''
    pdf_bytes = generate_pdf_from_text(title=cf.version.contract.title or cf.file_name, contract_code=cf.version.contract.contract_code, text_content=text_str)
    pdf_filename = os.path.splitext(cf.file_name)[0] + '.pdf'
    if not pdf_filename.lower().endswith('.pdf'):
        pdf_filename += '.pdf'
    response = ServiceResult(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
    return response

def download_contract_pdf(contract_id, user, query):
    from django.conf import settings
    import os
    from contracts.models import Contract, ContractVersion
    from contracts.services.document_encryption_service import decrypt_pdf
    from contracts.services.pdf_rendering_service import generate_pdf_from_text
    try:
        c = data_repository.get_contract_for_download_contract_pdf(contract_id)
        if not _check_contract_permission(user, c):
            return ServiceResult('Permission Denied: You do not have access to this contract.', status=403, content_type='text/html')
    except Contract.DoesNotExist:
        raise ObjectDoesNotExist('Contract not found')
    version_id = query.get('version_id')
    if version_id:
        try:
            version = data_repository.get_versions_for_download_contract_pdf(c, version_id)
        except ContractVersion.DoesNotExist:
            version = c.latest_version
    else:
        version = c.latest_version
    if not version:
        raise ObjectDoesNotExist('Version not found')
    latest_file = data_repository.first_files_for_download_contract_pdf(version)
    if latest_file:
        rel_path = latest_file.file_path
        media_prefix = settings.MEDIA_URL
        if rel_path.startswith(media_prefix):
            rel_path = rel_path[len(media_prefix):]
        physical_path = os.path.join(settings.MEDIA_ROOT, rel_path.replace('/', os.sep))
        if os.path.exists(physical_path):
            try:
                with open(physical_path, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = decrypt_pdf(encrypted_data)
                if latest_file.file_name.lower().endswith('.pdf') and decrypted_data.startswith(b'%PDF'):
                    response = ServiceResult(decrypted_data, content_type='application/pdf')
                    response['Content-Disposition'] = f'attachment; filename="{latest_file.file_name}"'
                    return response
                else:
                    try:
                        text_str = decrypted_data.decode('utf-8', errors='ignore')
                    except Exception:
                        text_str = str(decrypted_data)
                    pdf_bytes = generate_pdf_from_text(title=c.title, contract_code=c.contract_code, text_content=text_str)
                    filename = f"{c.contract_code or 'contract'}_v{version.version_number}.pdf"
                    response = ServiceResult(pdf_bytes, content_type='application/pdf')
                    response['Content-Disposition'] = f'attachment; filename="{filename}"'
                    return response
            except Exception as e:
                return ServiceResult(f'Error reading file: {str(e)}', status=500, content_type='text/html')
    details = contract_service.get_contract_details(c.id, version_id=version.id) or {}
    raw_content = details.get('raw_content') or c.title or ''
    pdf_bytes = generate_pdf_from_text(title=c.title, contract_code=c.contract_code, text_content=raw_content)
    filename = f"{c.contract_code or 'contract'}_v{version.version_number}.pdf"
    response = ServiceResult(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response
