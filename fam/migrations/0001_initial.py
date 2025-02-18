# Generated by Django 5.1.5 on 2025-02-01 12:27

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(max_length=100)),
                ('last_name', models.CharField(max_length=100)),
                ('maiden_name', models.CharField(blank=True, max_length=100, null=True)),
                ('birth_date', models.DateField(blank=True, null=True)),
                ('death_date', models.DateField(blank=True, null=True)),
                ('gender', models.CharField(blank=True, choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')], max_length=1, null=True)),
                ('birth_place', models.CharField(blank=True, max_length=200, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name_plural': 'People',
            },
        ),
        migrations.CreateModel(
            name='ParentChild',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relationship_type', models.CharField(choices=[('BIOLOGICAL', 'Biological'), ('ADOPTED', 'Adopted'), ('STEP', 'Step'), ('FOSTER', 'Foster'), ('GUARDIAN', 'Guardian')], default='BIOLOGICAL', max_length=20)),
                ('notes', models.TextField(blank=True, null=True)),
                ('child', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='parents_set', to='fam.person')),
                ('parent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='children_set', to='fam.person')),
            ],
            options={
                'verbose_name_plural': 'Parent-Child Relationships',
            },
        ),
        migrations.CreateModel(
            name='Relationship',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relationship_type', models.CharField(choices=[('MARRIAGE', 'Marriage'), ('PARTNERSHIP', 'Partnership'), ('ENGAGEMENT', 'Engagement'), ('DIVORCED', 'Divorced'), ('SEPARATED', 'Separated')], max_length=20)),
                ('start_date', models.DateField(blank=True, null=True)),
                ('end_date', models.DateField(blank=True, null=True)),
                ('is_current', models.BooleanField(default=True)),
                ('notes', models.TextField(blank=True, null=True)),
                ('person1', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='relationships_as_person1', to='fam.person')),
                ('person2', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='relationships_as_person2', to='fam.person')),
            ],
        ),
        migrations.AddIndex(
            model_name='person',
            index=models.Index(fields=['first_name', 'last_name'], name='fam_person_first_n_ea8f37_idx'),
        ),
        migrations.AddIndex(
            model_name='person',
            index=models.Index(fields=['birth_date', 'death_date'], name='fam_person_birth_d_0f9a74_idx'),
        ),
        migrations.AddIndex(
            model_name='person',
            index=models.Index(fields=['gender'], name='fam_person_gender_662fc6_idx'),
        ),
        migrations.AddIndex(
            model_name='parentchild',
            index=models.Index(fields=['parent'], name='fam_parentc_parent__05a797_idx'),
        ),
        migrations.AddIndex(
            model_name='parentchild',
            index=models.Index(fields=['child'], name='fam_parentc_child_i_62464b_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='parentchild',
            unique_together={('parent', 'child', 'relationship_type')},
        ),
        migrations.AddIndex(
            model_name='relationship',
            index=models.Index(fields=['person1', 'is_current'], name='fam_relatio_person1_7ee641_idx'),
        ),
        migrations.AddIndex(
            model_name='relationship',
            index=models.Index(fields=['person2', 'is_current'], name='fam_relatio_person2_980a12_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='relationship',
            unique_together={('person1', 'person2', 'start_date')},
        ),
    ]
