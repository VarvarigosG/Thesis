# Generated by Django 3.1.2 on 2021-01-18 07:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0002_auto_20201208_1525'),
    ]

    operations = [
        migrations.AlterField(
            model_name='choice',
            name='answer_text',
            field=models.CharField(blank=True, max_length=200),
        ),
        migrations.AlterField(
            model_name='question',
            name='question_text',
            field=models.CharField(max_length=1000),
        ),
    ]
