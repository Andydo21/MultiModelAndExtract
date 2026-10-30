from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("ai_extract", "0002_clause_extractedentity"),
        ("contracts", "0009_auditlog_payload"),
    ]

    operations = [
        migrations.CreateModel(
            name="LayoutExtractionRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("pipeline_name", models.CharField(default="plan-a-surya-layout", max_length=100)),
                ("status", models.CharField(choices=[("RUNNING", "Running"), ("COMPLETED", "Completed"), ("FAILED", "Failed")], default="RUNNING", max_length=20)),
                ("total_pages", models.PositiveIntegerField(default=0)),
                ("total_blocks", models.PositiveIntegerField(default=0)),
                ("error_message", models.TextField(blank=True)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("version", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="layout_extraction_runs", to="contracts.contractversion")),
            ],
            options={"ordering": ["-started_at"]},
        ),
        migrations.CreateModel(
            name="LayoutExtractionPage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("page_number", models.PositiveIntegerField()),
                ("width", models.PositiveIntegerField(default=0)),
                ("height", models.PositiveIntegerField(default=0)),
                ("image_path", models.CharField(blank=True, max_length=500)),
                ("extracted_text", models.TextField(blank=True)),
                ("run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="pages", to="ai_extract.layoutextractionrun")),
            ],
            options={"ordering": ["page_number"]},
        ),
        migrations.CreateModel(
            name="LayoutExtractionBlock",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("block_order", models.PositiveIntegerField()),
                ("layout_type", models.CharField(default="paragraph", max_length=50)),
                ("text", models.TextField(blank=True)),
                ("bbox", models.JSONField(default=list)),
                ("confidence", models.DecimalField(decimal_places=4, default=0, max_digits=5)),
                ("page", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="blocks", to="ai_extract.layoutextractionpage")),
            ],
            options={"ordering": ["block_order"]},
        ),
        migrations.AddConstraint(
            model_name="layoutextractionpage",
            constraint=models.UniqueConstraint(fields=("run", "page_number"), name="unique_layout_page_per_run"),
        ),
        migrations.AddConstraint(
            model_name="layoutextractionblock",
            constraint=models.UniqueConstraint(fields=("page", "block_order"), name="unique_layout_block_order"),
        ),
    ]
