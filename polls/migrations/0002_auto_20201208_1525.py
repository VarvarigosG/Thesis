# Generated by Django 3.1.2 on 2020-12-08 13:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='choice',
            name='answer_text',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AlterModelTable(
            name='choice',
            table='polls_choice',
        ),
    ]
