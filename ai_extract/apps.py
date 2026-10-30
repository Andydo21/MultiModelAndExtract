from django.apps import AppConfig


class AiExtractConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_extract"
    verbose_name = "AI Extract & Summarize"

    def ready(self):
        from django.db.models.signals import post_save, post_delete
        from contracts.models import ContractContext
        from .models import RetrievalSnapshot, LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock, Clause

        def invalidate(sender, instance, **kwargs):
            if kwargs.get('raw'):
                return
            if sender is LayoutExtractionBlock:
                lookup = dict(version__layout_extraction_runs__pages__pk=instance.page_id)
            elif sender is LayoutExtractionPage:
                lookup = dict(version__layout_extraction_runs__pk=instance.run_id)
            else:
                lookup = dict(version_id=instance.version_id)
            RetrievalSnapshot.objects.filter(**lookup).update(dirty=True)

        for sender in (LayoutExtractionRun, LayoutExtractionPage, LayoutExtractionBlock, Clause, ContractContext):
            for signal in (post_save, post_delete):
                signal.connect(invalidate, sender=sender, weak=False, dispatch_uid='retrieval-index:'+sender.__name__)
