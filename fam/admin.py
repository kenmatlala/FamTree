from django.contrib import admin
from django.utils.html import format_html
from .models import Person, Relationship, ParentChild


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = (
        'get_full_name', 'birth_date', 'death_date', 'gender', 'age', 'birth_place', 'created_by', 'created_at')
    list_filter = ('gender', 'created_at', 'birth_place')
    search_fields = ('first_name', 'last_name', 'maiden_name', 'birth_place')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'birth_date'

    fieldsets = (
        ('Basic Information', {
            'fields': ('first_name', 'last_name', 'maiden_name', 'gender')
        }),
        ('Dates', {
            'fields': ('birth_date', 'death_date')
        }),
        ('Additional Information', {
            'fields': ('birth_place', 'created_by')
        }),
        ('System Fields', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def get_queryset(self, request):
        """Prefetch related fields to avoid N+1 queries"""
        return super().get_queryset(request).select_related('created_by')


@admin.register(Relationship)
class RelationshipAdmin(admin.ModelAdmin):
    list_display = ('get_relationship_display', 'relationship_type', 'start_date', 'end_date', 'is_current')
    list_filter = ('relationship_type', 'is_current', 'start_date')
    search_fields = ('person1__first_name', 'person1__last_name', 'person2__first_name', 'person2__last_name')
    raw_id_fields = ('person1', 'person2')

    fieldsets = (
        ('People', {
            'fields': ('person1', 'person2')
        }),
        ('Relationship Details', {
            'fields': ('relationship_type', 'is_current', 'start_date', 'end_date')
        }),
        ('Additional Information', {
            'fields': ('notes',),
            'classes': ('collapse',)
        }),
    )

    def get_relationship_display(self, obj):
        """Display relationship as Person1 ↔ Person2"""
        return format_html(
            '{} ↔ {}',
            obj.person1.get_full_name(),
            obj.person2.get_full_name()
        )

    get_relationship_display.short_description = 'Relationship'

    def get_queryset(self, request):
        """Prefetch related fields to avoid N+1 queries"""
        return super().get_queryset(request).select_related('person1', 'person2')


@admin.register(ParentChild)
class ParentChildAdmin(admin.ModelAdmin):
    list_display = ('get_relationship_display', 'relationship_type', 'get_age_difference')
    list_filter = ('relationship_type',)
    search_fields = ('parent__first_name', 'parent__last_name', 'child__first_name', 'child__last_name')
    raw_id_fields = ('parent', 'child')

    fieldsets = (
        ('People', {
            'fields': ('parent', 'child')
        }),
        ('Relationship Details', {
            'fields': ('relationship_type',)
        }),
        ('Additional Information', {
            'fields': ('notes',),
            'classes': ('collapse',)
        }),
    )

    def get_relationship_display(self, obj):
        """Display relationship as Parent → Child"""
        return format_html(
            '{} → {}',
            obj.parent.get_full_name(),
            obj.child.get_full_name()
        )

    get_relationship_display.short_description = 'Relationship'

    def get_age_difference(self, obj):
        """Calculate and display age difference between parent and child"""
        if obj.parent.birth_date and obj.child.birth_date:
            years = (obj.child.birth_date - obj.parent.birth_date).days // 365
            return f"{years} years"
        return "Unknown"

    get_age_difference.short_description = 'Age Difference'

    def get_queryset(self, request):
        """Prefetch related fields to avoid N+1 queries"""
        return super().get_queryset(request).select_related('parent', 'child')
