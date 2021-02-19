def validate_file_extension(value):
    from django.core.exceptions import ValidationError
    if not value.name.endswith('.pkl'):
        raise ValidationError('Unsupported file extension.')


def validate_file_extension1(value):
    from django.core.exceptions import ValidationError
    if not value.name.endswith('.csv'):
        raise ValidationError('Unsupported file extension.')
