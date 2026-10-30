from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ai_extract", "0003_layout_extraction_models"),
    ]

    operations = [
        migrations.AlterField(
            model_name="layoutextractionrun",
            name="pipeline_name",
            field=models.CharField(default="plan-a-surya-colpali", max_length=100),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="dense_index_status",
            field=models.CharField(
                choices=[
                    ("PENDING", "Pending"),
                    ("RUNNING", "Running"),
                    ("COMPLETED", "Completed"),
                    ("PARTIAL", "Partial"),
                    ("FAILED", "Failed"),
                    ("SKIPPED", "Skipped"),
                ],
                default="PENDING",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="dense_points_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="dense_index_error",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="colpali_index_status",
            field=models.CharField(
                choices=[
                    ("PENDING", "Pending"),
                    ("RUNNING", "Running"),
                    ("COMPLETED", "Completed"),
                    ("PARTIAL", "Partial"),
                    ("FAILED", "Failed"),
                    ("SKIPPED", "Skipped"),
                ],
                default="PENDING",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="colpali_points_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="layoutextractionrun",
            name="colpali_index_error",
            field=models.TextField(blank=True),
        ),
    ]
