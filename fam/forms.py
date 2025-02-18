from django import forms
from .models import Person, Relationship, ParentChild
from datetime import date


class PersonForm(forms.ModelForm):
    """Form for creating and editing Person records."""

    class Meta:
        model = Person
        fields = [
            'first_name',
            'last_name',
            'maiden_name',
            'birth_date',
            'death_date',
            'gender',
            'birth_place'
        ]
        widgets = {
            'birth_date': forms.DateInput(attrs={'type': 'date'}),
            'death_date': forms.DateInput(attrs={'type': 'date'})
        }

    def clean(self):
        cleaned_data = super().clean()
        birth_date = cleaned_data.get('birth_date')
        death_date = cleaned_data.get('death_date')

        # Additional validation beyond model validation
        if birth_date:
            # Check if birth date is reasonable (e.g., not more than 130 years ago)
            max_age = date.today().year - 130
            if birth_date.year < max_age:
                self.add_error('birth_date', 'Birth date seems unreasonably old')

        return cleaned_data


class RelationshipForm(forms.ModelForm):
    """Form for creating and editing Relationship records."""

    class Meta:
        model = Relationship
        fields = [
            'person1',
            'person2',
            'relationship_type',
            'start_date',
            'end_date',
            'is_current',
            'notes'
        ]
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

    def clean(self):
        cleaned_data = super().clean()
        person1 = cleaned_data.get('person1')
        person2 = cleaned_data.get('person2')
        start_date = cleaned_data.get('start_date')
        is_current = cleaned_data.get('is_current')

        if person1 and person2 and start_date:
            # Check if both people were alive at relationship start
            if person1.death_date and person1.death_date < start_date:
                self.add_error('start_date', f'{person1.get_full_name()} was deceased before this date')
            if person2.death_date and person2.death_date < start_date:
                self.add_error('start_date', f'{person2.get_full_name()} was deceased before this date')

            # Check if both people were of reasonable age (e.g., at least 16)
            min_age = 16
            if person1.birth_date and (start_date.year - person1.birth_date.year) < min_age:
                self.add_error('start_date', f'{person1.get_full_name()} would have been under {min_age}')
            if person2.birth_date and (start_date.year - person2.birth_date.year) < min_age:
                self.add_error('start_date', f'{person2.get_full_name()} would have been under {min_age}')

        return cleaned_data


class ParentChildForm(forms.ModelForm):
    """Form for creating and editing ParentChild relationships."""

    class Meta:
        model = ParentChild
        fields = [
            'parent',
            'child',
            'relationship_type',
            'notes'
        ]
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

    def clean(self):
        cleaned_data = super().clean()
        parent = cleaned_data.get('parent')
        child = cleaned_data.get('child')
        relationship_type = cleaned_data.get('relationship_type')

        if parent and child:
            # For biological relationships, enforce stricter age differences
            if relationship_type == 'BIOLOGICAL':
                if parent.birth_date and child.birth_date:
                    age_difference = child.birth_date.year - parent.birth_date.year
                    if age_difference < 12:
                        self.add_error('parent', 'Parent seems too young for a biological relationship')
                    elif age_difference > 70:
                        self.add_error('parent', 'Parent seems too old for a biological relationship')

            # For adoptive relationships, ensure parent was at least 18 at adoption
            elif relationship_type in ['ADOPTED', 'FOSTER', 'GUARDIAN']:
                if parent.birth_date and child.birth_date:
                    age_difference = child.birth_date.year - parent.birth_date.year
                    if age_difference < 18:
                        self.add_error('parent',
                                       f'Parent must be at least 18 for {relationship_type.lower()} relationship')

        return cleaned_data
