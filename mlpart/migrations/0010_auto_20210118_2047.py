# Generated by Django 3.1.2 on 2021-01-18 18:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mlpart', '0009_auto_20210118_2045'),
    ]

    operations = [
        migrations.AlterField(
            model_name='file',
            name='file',
            field=models.FileField(upload_to=''),
        ),
    ]