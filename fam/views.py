from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.core.exceptions import ValidationError
from django.core.cache import cache
from .models import Person
from .forms import PersonForm, RelationshipForm, ParentChildForm


def index(request):
    # You can pass any context data to the template if needed
    context = {
        'message': 'Welcome to the Family Tree Application!',
    }
    return render(request, 'index.html', context)


@login_required
def add_person(request):
    if request.method == 'POST':
        form = PersonForm(request.POST)
        if form.is_valid():
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']

            # Check for similar names
            similar_people = Person.find_similar_names(first_name, last_name)

            if similar_people.exists() and not request.POST.get('confirmed'):
                return render(request, 'confirm_person.html', {
                    'similar_people': similar_people,
                    'form': form
                })

            try:
                person = form.save(commit=False)
                person.created_by = request.user
                person.save()
                messages.success(request, f"Successfully added {person.get_full_name()} to the family tree.")
                return redirect('person_detail', pk=person.pk)
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, 'add_person.html', {'form': form})
    else:
        form = PersonForm()
    return render(request, 'add_person.html', {'form': form})


@login_required
def person_detail(request, pk):
    person = get_object_or_404(Person, pk=pk)
    family_members = person.get_family_members(generations_up=2, generations_down=2)

    context = {
        'person': person,
        'family_members': family_members,
        'age': person.age,
        'current_partners': person.get_current_partners(),
        'past_relationships': person.get_relationships().filter(is_current=False)
    }
    return render(request, 'person_detail.html', context)


@login_required
def add_relationship(request):
    if request.method == 'POST':
        form = RelationshipForm(request.POST)
        if form.is_valid():
            try:
                relationship = form.save()
                messages.success(request, "Successfully added new relationship.")
                return redirect('person_detail', pk=relationship.person1.pk)
            except ValidationError as e:
                messages.error(request, str(e))
    else:
        form = RelationshipForm()
    return render(request, 'add_relationship.html', {'form': form})


@login_required
def add_parent_child(request):
    if request.method == 'POST':
        form = ParentChildForm(request.POST)
        if form.is_valid():
            try:
                parent_child = form.save()
                messages.success(request, "Successfully added parent-child relationship.")
                return redirect('person_detail', pk=parent_child.child.pk)
            except ValidationError as e:
                messages.error(request, str(e))
    else:
        form = ParentChildForm()
    return render(request, 'add_parent_child.html', {'form': form})


@login_required
def view_tree(request, person_id):
    person = get_object_or_404(Person, pk=person_id)
    family_members = person.get_family_members(
        generations_up=3,  # Show up to great-grandparents
        generations_down=3  # Show up to great-grandchildren
    )

    context = {
        'person': person,
        'family_members': family_members,
        'age': person.age
    }
    return render(request, 'view_tree.html', context)


@login_required
def find_relationship(request, person1_id, person2_id):
    person1 = get_object_or_404(Person, pk=person1_id)
    person2 = get_object_or_404(Person, pk=person2_id)

    # Get relationship degree
    relationship_type, details = person1.calculate_relationship_degree(person2)

    # Find common ancestors
    common_ancestors = person1.get_common_ancestors(person2)

    # Get the relationship path
    relationship_path = find_relationship_path(person1, person2)

    context = {
        'person1': person1,
        'person2': person2,
        'relationship_type': relationship_type,
        'relationship_details': details,
        'common_ancestors': common_ancestors,
        'relationship_path': format_relationship_path(relationship_path) if relationship_path else None
    }
    return render(request, 'relationship_result.html', context)


@login_required
def analyze_family(request, person_id):
    person = get_object_or_404(Person, pk=person_id)

    context = {
        'person': person,
        'direct_ancestors': person.get_ancestors(max_generations=4),
        'direct_descendants': person.get_descendants(max_generations=4),
        'siblings': person.get_siblings(include_half=True),
        'cousins': person.get_relatives_by_type(
            Person.RelationshipDegree.COUSIN,
            max_distance=3
        ),
        'aunts_uncles': person.get_relatives_by_type(
            Person.RelationshipDegree.AUNT_UNCLE
        ),
        'nieces_nephews': person.get_relatives_by_type(
            Person.RelationshipDegree.NIECE_NEPHEW
        )
    }
    return render(request, 'family_analysis.html', context)


@login_required
def search_people(request):
    query = request.GET.get('q', '')
    if query:
        results = Person.objects.filter(
            Q(first_name__icontains=query) |
            Q(last_name__icontains=query) |
            Q(maiden_name__icontains=query)
        ).distinct()

        paginator = Paginator(results, 20)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        context = {
            'query': query,
            'page_obj': page_obj,
            'total_results': results.count()
        }
    else:
        context = {'query': '', 'page_obj': None, 'total_results': 0}

    return render(request, 'search_results.html', context)


def format_relationship_path(path):
    """Helper function to format relationship path into human-readable description"""
    description = []
    for rel, rel_type in path:
        if rel_type == 'parent':
            description.append(f"parent ({rel.relationship_type.lower()})")
        elif rel_type == 'child':
            description.append(f"child ({rel.relationship_type.lower()})")
        elif rel_type == 'partner':
            status = 'current' if rel.is_current else 'former'
            description.append(f"{status} {rel.relationship_type.lower()}")

    return " → ".join(description)


def find_relationship_path(person1, person2, max_depth=5):
    """Enhanced version of find_relationship_path with caching and optimization"""
    cache_key = f"relationship_path_{person1.id}_{person2.id}"
    path = cache.get(cache_key)

    if path is not None:
        return path

    visited = set()

    def dfs(current, target, path, depth):
        if depth > max_depth:
            return None
        if current == target:
            return path

        visited.add(current.id)

        # Check direct relationships first (likely to be closer)
        for rel in current.get_relationships():
            partner = rel.person2 if rel.person1 == current else rel.person1
            if partner.id not in visited:
                new_path = path + [(rel, 'partner')]
                result = dfs(partner, target, new_path, depth + 1)
                if result:
                    return result

        # Check parent/child relationships
        for parent_rel in current.get_parents():
            if parent_rel.parent.id not in visited:
                new_path = path + [(parent_rel, 'parent')]
                result = dfs(parent_rel.parent, target, new_path, depth + 1)
                if result:
                    return result

        for child_rel in current.get_children():
            if child_rel.child.id not in visited:
                new_path = path + [(child_rel, 'child')]
                result = dfs(child_rel.child, target, new_path, depth + 1)
                if result:
                    return result

        return None

    path = dfs(person1, person2, [], 0)
    cache.set(cache_key, path, 3600)  # Cache for 1 hour
    return path
