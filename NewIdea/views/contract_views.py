"""NewIdea search entry points over the web's tenant-scoped A3 pipeline."""
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from NewIdea.services import contract_search_service
from .knowledge_views import knowledge_api, json_body


def workspace(request, contract_id=None):
    from django.contrib.auth.views import redirect_to_login
    if not request.user.is_authenticated:
        return redirect_to_login(request.get_full_path())
    return render_workspace(request, contract_id)


@knowledge_api
def render_workspace(request, contract_id):
    from django.shortcuts import render
    from django.urls import reverse
    context=contract_search_service.workspace_context(request.user,contract_id,request.GET.get('version_id'))
    selected=context['selected_contract']
    context.update(newidea_mode=True,contract_workspace_url_name='newidea_contract',
        search_endpoint=reverse('newidea_search',args=[selected.pk]) if selected else '')
    return render(request,'contracts/intelligence.html',context)


@require_POST
@knowledge_api
def search_contract(request, contract_id):
    return search_response(request.user,contract_id,json_body(request))


def search_response(user, contract_id, data):
    result=contract_search_service.search_contract(user,contract_id,data)
    return JsonResponse(dict(result,search_engine='newidea_a3'))


@require_POST
@knowledge_api
def ask(request):
    """Body-based endpoint for NewIdea clients that already send contract_id."""
    data=json_body(request)
    contract_id=data.get('contract_id')
    if type(contract_id) is not int or contract_id < 1:
        raise ValueError('Chọn hợp đồng trước khi tìm tài liệu A3.')
    return search_response(request.user,contract_id,data)
