# Generated by Django 3.1.2 on 2021-01-18 14:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mlpart', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='file',
            name='file',
            field=models.FileField(upload_to='research/files'),
        ),
    ]
