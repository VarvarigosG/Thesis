# Generated by Django 3.1.2 on 2021-02-19 15:48

from django.db import migrations, models
import mlpart.validators


class Migration(migrations.Migration):

    dependencies = [
        ('mlpart', '0021_auto_20210219_1729'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mlmodeldata',
            name='data',
            field=models.FileField(upload_to='', validators=[mlpart.validators.validate_file_extension1]),
        ),
    ]
