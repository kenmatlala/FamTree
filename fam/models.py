# from django.db import models
from django.db import models
from django.contrib.auth.models import User
from django.db.models import Q
from django.core.exceptions import ValidationError
from datetime import date
import itertools
from django.core.cache import cache
from collections import defaultdict


# Create your models here.
class Person(models.Model):
    """
    Represents an individual in the family tree.
    Stores basic biographical information and links to user who created the record.
    """
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    maiden_name = models.CharField(max_length=100, blank=True, null=True)
    birth_date = models.DateField(null=True, blank=True)
    death_date = models.DateField(null=True, blank=True)
    gender = models.CharField(
        max_length=1,
        choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')],
        blank=True,
        null=True
    )
    birth_place = models.CharField(max_length=200, blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['first_name', 'last_name']),
            models.Index(fields=['birth_date', 'death_date']),
            models.Index(fields=['gender']),
        ]
        verbose_name_plural = "People"

    class RelationshipDegree:
        """Constants for relationship classifications"""
        DIRECT = 'direct'
        SIBLING = 'sibling'
        COUSIN = 'cousin'
        REMOVED_COUSIN = 'removed_cousin'
        AUNT_UNCLE = 'aunt_uncle'
        NIECE_NEPHEW = 'niece_nephew'
        STEP = 'step'
        IN_LAW = 'in_law'
        UNRELATED = 'unrelated'

    def clean(self):
        # Validate birth date is before death date if both exist
        if self.birth_date and self.death_date:
            if self.birth_date > self.death_date:
                raise ValidationError("Birth date must be before death date")

        # Validate birth date is not in the future
        if self.birth_date and self.birth_date > date.today():
            raise ValidationError("Birth date cannot be in the future")

    def get_full_name(self):
        """Returns the person's full name."""
        if self.maiden_name:
            return f"{self.first_name} {self.maiden_name} {self.last_name}"
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self):
        """Calculate age based on birth_date and death_date/current date."""
        if not self.birth_date:
            return None

        end_date = self.death_date if self.death_date else date.today()
        age = end_date.year - self.birth_date.year

        # Adjust age if birthday hasn't occurred this year
        if end_date.month < self.birth_date.month or \
                (end_date.month == self.birth_date.month and end_date.day < self.birth_date.day):
            age -= 1

        return age

    @classmethod
    def find_similar_names(cls, first_name, last_name):
        """
        Find people with similar names using case-insensitive matching.
        Also checks maiden names if available.
        """
        return cls.objects.filter(
            Q(first_name__iexact=first_name) &
            (Q(last_name__iexact=last_name) | Q(maiden_name__iexact=last_name))
        )

    def get_parents(self):
        """Returns all parent relationships for this person."""
        return self.parents_set.all()

    def get_children(self):
        """Returns all child relationships for this person."""
        return self.children_set.all()

    def get_relationships(self):
        """Returns all romantic relationships (current and past) for this person."""
        return Relationship.objects.filter(
            models.Q(person1=self) | models.Q(person2=self)
        )

    def get_current_partners(self):
        """Returns all current partners/spouses."""
        relationships = self.get_relationships().filter(is_current=True)
        return [
            rel.person2 if rel.person1 == self else rel.person1
            for rel in relationships
        ]

    def _get_cache_key(self, method_name, *args):
        """Generate a unique cache key for a method call."""
        args_str = '-'.join(str(arg) for arg in args)
        return f"person_{self.id}_{method_name}_{args_str}"

    def _cached_query(self, method_name, query_func, timeout=3600, *args):
        """Execute a query with caching."""
        cache_key = self._get_cache_key(method_name, *args)
        result = cache.get(cache_key)

        if result is None:
            result = query_func()
            cache.set(cache_key, result, timeout)

        return result

    def get_siblings(self, include_half=True):
        """Cached version of sibling retrieval."""

        def query_func():
            if include_half:
                parent_ids = self.get_parents().values_list('parent_id', flat=True)
                return list(Person.objects.filter(
                    parents_set__parent_id__in=parent_ids
                ).exclude(id=self.id).distinct())
            else:
                parent_ids = set(self.get_parents().values_list('parent_id', flat=True))
                if len(parent_ids) != 2:
                    return []
                return list(Person.objects.filter(
                    parents_set__parent_id__in=parent_ids
                ).annotate(
                    parent_count=models.Count('parents_set__parent_id')
                ).filter(parent_count=2).exclude(id=self.id))

        return self._cached_query('siblings', query_func, include_half)

    def validate_relationship_rules(self):
        """Enhanced validation rules for relationships"""
        errors = []

        # Basic validation (from previous version)
        self._validate_basic_rules(errors)

        # Advanced validation rules
        self._validate_age_rules(errors)
        self._validate_family_rules(errors)
        self._validate_temporal_rules(errors)
        self._validate_cultural_rules(errors)

        if errors:
            raise ValidationError(errors)

    def _validate_basic_rules(self, errors):
        """Basic validation rules"""
        # Circular ancestry check
        if self in self.get_ancestors():
            errors.append("Circular ancestry detected")

        # Biological parents limit
        bio_parents = self.get_parents().filter(relationship_type='BIOLOGICAL')
        if bio_parents.count() > 2:
            errors.append("Cannot have more than 2 biological parents")

    def _validate_age_rules(self, errors):
        """Age-related validation rules"""
        if self.birth_date:
            # Minimum parent age rule
            for parent_rel in self.get_parents():
                if parent_rel.parent.birth_date:
                    age_diff = (self.birth_date - parent_rel.parent.birth_date).days / 365
                    if parent_rel.relationship_type == 'BIOLOGICAL' and age_diff < 12:
                        errors.append(
                            f"Biological parent {parent_rel.parent.get_full_name()} would have been {age_diff} years "
                            f"old at birth")

            # Maximum age rule
            if self.birth_date < date(1800, 1, 1):
                errors.append("Birth date unreasonably old")

            # Future birth date
            if self.birth_date > date.today():
                errors.append("Birth date cannot be in the future")

    def _validate_family_rules(self, errors):
        """Family relationship validation rules"""
        # Marriage to relatives check
        for partner in self.get_current_partners():
            # Check close relatives
            if any([
                partner in self.get_siblings(include_half=True),
                partner in self.get_parents(),
                partner in self.get_children(),
                partner in self.get_first_cousins()
            ]):
                errors.append(f"Invalid marriage to close relative: {partner.get_full_name()}")

        # Multiple spouse check based on date ranges
        marriages = self.get_relationships().filter(relationship_type='MARRIAGE')
        for m1, m2 in itertools.combinations(marriages, 2):
            if self._date_ranges_overlap(m1, m2):
                errors.append("Overlapping marriage dates detected")

    def _validate_temporal_rules(self, errors):
        """Time-based validation rules"""
        # Death date validation
        if self.death_date:
            if self.death_date > date.today():
                errors.append("Death date cannot be in the future")
            if self.birth_date and self.death_date < self.birth_date:
                errors.append("Death date cannot be before birth date")

        # Relationship date validation
        for rel in self.get_relationships():
            if rel.start_date and rel.end_date and rel.start_date > rel.end_date:
                errors.append(f"Invalid relationship dates for {rel.relationship_type}")

    def _validate_cultural_rules(self, errors):
        """Cultural and logical validation rules"""
        # Gender-based parent validation (if using traditional family structures)
        if self.gender == 'M':
            maternal_count = self.get_parents().filter(
                parent__gender='F',
                relationship_type='BIOLOGICAL'
            ).count()
            if maternal_count > 1:
                errors.append("Multiple biological mothers specified")

    def get_ancestors(self, max_generations=None):
        """Get all ancestors up to specified generations."""

        def query_func():
            ancestors = []
            to_process = [(self, 0)]
            processed = set()

            while to_process:
                person, generation = to_process.pop(0)
                if max_generations and generation >= max_generations:
                    continue

                for parent_rel in person.get_parents():
                    parent = parent_rel.parent
                    if parent.id not in processed:
                        ancestors.append(parent)
                        processed.add(parent.id)
                        to_process.append((parent, generation + 1))

            return ancestors

        return self._cached_query('ancestors', query_func, max_generations)

    def get_common_ancestors(self, other_person):
        """Find common ancestors between two people."""

        def query_func():
            self_ancestors = set(self.get_ancestors())
            other_ancestors = set(other_person.get_ancestors())
            return list(self_ancestors.intersection(other_ancestors))

        cache_key = f"common_ancestors_{self.id}_{other_person.id}"
        return self._cached_query(cache_key, query_func)

    def calculate_relationship_degree(self, other_person):
        """
        Enhanced relationship calculator with consistent return types.
        Returns a tuple of (relationship_type: str, details: str)
        """

        def query_func():
            # Direct relationships
            if other_person in self.get_ancestors():
                generations = len(self.get_path_to_ancestor(other_person))
                return self.RelationshipDegree.DIRECT, f'ancestor_{generations}'

            if other_person in self.get_descendants():
                generations = len(self._get_path_to_descendant(other_person))
                return self.RelationshipDegree.DIRECT, f'descendant_{generations}'

            # Sibling relationships
            siblings = self.get_siblings(include_half=True)
            if other_person in siblings:
                is_half = other_person not in self.get_siblings(include_half=False)
                return self.RelationshipDegree.SIBLING, 'half' if is_half else 'full'

            # Find common ancestors for cousin relationships
            common_ancestors = self.get_common_ancestors(other_person)
            if common_ancestors:
                closest_ancestor = min(
                    common_ancestors,
                    key=lambda x: (
                            len(self.get_path_to_ancestor(x)) +
                            len(other_person.get_path_to_ancestor(x))
                    )
                )

                gen1 = len(self.get_path_to_ancestor(closest_ancestor))
                gen2 = len(other_person.get_path_to_ancestor(closest_ancestor))

                if gen1 == gen2:
                    degree = gen1 - 1
                    return self.RelationshipDegree.COUSIN, f'{degree}'
                else:
                    degree = min(gen1, gen2) - 1
                    removal = abs(gen1 - gen2)
                    return self.RelationshipDegree.REMOVED_COUSIN, f'{degree}_{removal}'

            # Check for in-law relationships
            if self._is_in_law_relationship(other_person):
                in_law_type = self._get_in_law_type(other_person)
                return self.RelationshipDegree.IN_LAW, in_law_type

            # Check for step relationships
            if self._is_step_relationship(other_person):
                step_type = self._get_step_type(other_person)
                return self.RelationshipDegree.STEP, step_type

            return self.RelationshipDegree.UNRELATED, 'none'

        return self._cached_query('relationship_degree', query_func)

    @staticmethod
    def _date_ranges_overlap(rel1, rel2):
        """Helper to check if two relationships overlap in time"""
        return (
                (not rel1.end_date or not rel2.start_date or rel1.end_date > rel2.start_date) and
                (not rel2.end_date or not rel1.start_date or rel2.end_date > rel1.start_date)
        )

    def get_relatives_by_type(self, relationship_type, max_distance=None):
        """
        Find all relatives of a specific type.
        Uses optimized queries and caching for large trees.
        """
        cache_key = f"relatives_{self.id}_{relationship_type}_{max_distance}"
        result = cache.get(cache_key)

        if result is None:
            if relationship_type == self.RelationshipDegree.COUSIN:
                result = self._find_cousins_optimized(max_distance)
            elif relationship_type == self.RelationshipDegree.SIBLING:
                result = self._find_siblings_optimized()
            elif relationship_type == self.RelationshipDegree.AUNT_UNCLE:
                result = self._find_aunts_uncles_optimized()
            elif relationship_type == self.RelationshipDegree.NIECE_NEPHEW:
                result = self._find_nieces_nephews_optimized()

            cache.set(cache_key, result, 3600)  # Cache for 1 hour

        return result

    def _find_cousins_optimized(self, max_degree=None):
        """
        Optimized method to find cousins up to a specific degree.
        Uses batch loading and query optimization.
        """
        cousins = defaultdict(list)  # degree -> list of cousins

        # Prefetch related objects to avoid N+1 queries
        parents = self.get_parents().select_related('parent')

        for parent in parents:
            # Get aunts/uncles (parent's siblings)
            aunts_uncles = Person.objects.filter(
                parents_set__parent_id__in=parent.parent.get_parents().values_list('parent_id', flat=True)
            ).exclude(
                id__in=[p.parent.id for p in parents]
            ).distinct()

            # Get their children (first cousins)
            first_cousins = Person.objects.filter(
                parents_set__parent_id__in=aunts_uncles.values_list('id', flat=True)
            ).exclude(id=self.id).distinct()

            cousins['1'].extend(first_cousins)

            # Continue for higher degrees if requested
            if max_degree and max_degree > 1:
                for degree in range(2, max_degree + 1):
                    previous_cousins = cousins[str(degree - 1)]
                    next_cousins = Person.objects.filter(
                        parents_set__parent_id__in=Person.objects.filter(
                            parents_set__child_id__in=[c.id for c in previous_cousins]
                        ).values_list('id', flat=True)
                    ).distinct()
                    cousins[str(degree)].extend(next_cousins)

        return dict(cousins)

    def _find_siblings_optimized(self):
        """Optimized sibling finding with relationship types"""
        return Person.objects.filter(
            parents_set__parent_id__in=self.get_parents().values_list('parent_id', flat=True)
        ).exclude(
            id=self.id
        ).annotate(
            common_parents=models.Count('parents_set__parent_id')
        ).select_related(
            'parents_set__relationship_type'
        ).distinct()

    def get_path_to_ancestor(self, ancestor):
        """Helper method to find the path to an ancestor."""
        """
            Returns a list of Person objects representing the path from this person to the given ancestor.
            Returns empty list if no path exists.

            Args:
                ancestor (Person): The ancestor to find path to

            Returns:
                list[Person]: List of people in path from self to ancestor, or empty list if no path
            """
        path = []
        current = self
        while current and current != ancestor:
            parents = current.get_parents()
            if not parents:
                return []
            parent_rel = parents[0]  # Take first parent path
            path.append(parent_rel.parent)
            current = parent_rel.parent

        # Only return path if we actually reached the ancestor
        return path if path and path[-1] == ancestor else []

    def _is_in_law_relationship(self, other_person):
        """Check for in-law relationships"""
        partners = self.get_current_partners()
        for partner in partners:
            if other_person in partner.get_relatives_by_type(self.RelationshipDegree.DIRECT):
                return True
        return False

    def _get_in_law_type(self, other_person):
        """
        Determine specific type of in-law relationship.
        Returns the type of in-law relationship as a string.

        Args:
            other_person (Person): The person to check relationship with

        Returns:
            str: Type of in-law relationship ('spouse_parent', 'spouse_sibling',
                 'sibling_spouse', 'child_spouse', or None if no in-law relationship)
        """
        # Check if other person is spouse's parent (parent-in-law)
        for partner in self.get_current_partners():
            if other_person in partner.get_parents():
                return 'spouse_parent'

            # Check if other person is spouse's sibling (sibling-in-law)
            if other_person in partner.get_siblings():
                return 'spouse_sibling'

        # Check if other person is sibling's spouse (also sibling-in-law)
        for sibling in self.get_siblings():
            if other_person in sibling.get_current_partners():
                return 'sibling_spouse'

        # Check if other person is child's spouse (child-in-law)
        for child_rel in self.get_children():
            child = child_rel.child
            if other_person in child.get_current_partners():
                return 'child_spouse'

        return None  # No in-law relationship found

    def _is_step_relationship(self, other_person):
        """Check for step relationships"""
        partners = self.get_current_partners()
        for partner in partners:
            if other_person in partner.get_children():
                return True
        return False

    def _get_path_to_descendant(self, descendant):
        """Helper method to find the path to a descendant."""
        return descendant.get_path_to_ancestor(self)

    def invalidate_cache(self):
        """Invalidate all cached data for this person."""
        cache_patterns = [
            f"person_{self.id}_*",
            f"common_ancestors_{self.id}_*",
            f"*_common_ancestors_*_{self.id}"
        ]
        for pattern in cache_patterns:
            cache.delete_pattern(pattern)

    def save(self, *args, **kwargs):
        """Override save to validate relationships and clear cache."""
        self.validate_relationship_rules()
        super().save(*args, **kwargs)
        self.invalidate_cache()

    def get_grandparents(self):
        """Returns all grandparents."""
        grandparents = []
        for parent_rel in self.get_parents():
            grandparents.extend(parent_rel.parent.get_parents())
        return grandparents

    def get_grandchildren(self):
        """Returns all grandchildren."""
        grandchildren = []
        for child_rel in self.get_children():
            grandchildren.extend(child_rel.child.get_children())
        return grandchildren

    def get_aunts_uncles(self):
        """Returns all aunts and uncles (siblings of parents)."""
        aunts_uncles = []
        for parent_rel in self.get_parents():
            aunts_uncles.extend(parent_rel.parent.get_siblings())
        return aunts_uncles

    def get_cousins(self):
        """Returns first cousins (children of aunts/uncles)."""
        cousins = []
        for aunt_uncle in self.get_aunts_uncles():
            cousins.extend(aunt_uncle.get_children())
        return cousins

    def get_family_members(self, generations_up=2, generations_down=2):
        """
        Returns a dictionary of all family members within specified generations.
        """
        family = {
            'parents': list(self.get_parents()),
            'siblings': list(self.get_siblings()),
            'partners': list(self.get_current_partners()),
            'children': list(self.get_children()),
            'grandparents': list(self.get_grandparents()) if generations_up >= 2 else [],
            'grandchildren': list(self.get_grandchildren()) if generations_down >= 2 else [],
            'aunts_uncles': list(self.get_aunts_uncles()) if generations_up >= 2 else [],
            'cousins': list(self.get_cousins()) if generations_up >= 2 else []
        }
        return family

    def add_parent(self, parent, relationship_type='BIOLOGICAL'):
        """
        Adds a parent relationship to this person.
        """
        return ParentChild.objects.create(
            parent=parent,
            child=self,
            relationship_type=relationship_type
        )

    def add_child(self, child, relationship_type='BIOLOGICAL'):
        """
        Adds a child relationship to this person.
        """
        return ParentChild.objects.create(
            parent=self,
            child=child,
            relationship_type=relationship_type
        )

    def add_partner(self, partner, relationship_type='MARRIAGE', start_date=None):
        """
        Adds a partner relationship to this person.
        """
        return Relationship.objects.create(
            person1=self,
            person2=partner,
            relationship_type=relationship_type,
            start_date=start_date,
            is_current=True
        )

    def get_first_cousins(self):
        """Returns all first cousins (children of aunts/uncles)."""
        first_cousins = []
        for parent in self.get_parents():
            for aunt_uncle in parent.get_siblings():
                first_cousins.extend(aunt_uncle.get_children())
        return list(set(first_cousins))  # Remove duplicates

    def get_descendants(self, max_generations=None):
        """Returns all descendants up to specified generations."""
        descendants = []
        to_process = [(self, 0)]
        processed = set()

        while to_process:
            person, generation = to_process.pop(0)
            if max_generations and generation >= max_generations:
                continue

            for child_rel in person.get_children():
                child = child_rel.child
                if child.id not in processed:
                    descendants.append(child)
                    processed.add(child.id)
                    to_process.append((child, generation + 1))

        return descendants

    def _get_step_type(self, other_person):
        """
        Determine specific type of step relationship.
        Returns 'child', 'parent', or 'sibling' depending on relationship.
        """
        # Check if other person is step-child
        for partner in self.get_current_partners():
            if other_person in partner.get_children():
                return 'child'

        # Check if other person is stepparent
        for parent in self.get_parents():
            if other_person in parent.get_current_partners():
                return 'parent'

        # Check if other person is step-sibling
        for parent in self.get_parents():
            for partner in parent.get_current_partners():
                if other_person in partner.get_children():
                    return 'sibling'

        return None

    def _find_aunts_uncles_optimized(self):
        """
        Optimized method to find all aunts and uncles.
        Uses a single query with prefetch_related.
        """
        parent_ids = self.get_parents().values_list('parent_id', flat=True)

        return Person.objects.filter(
            parents_set__parent_id__in=Person.objects.filter(
                children_set__child_id__in=parent_ids
            ).values_list('id', flat=True)
        ).exclude(
            id__in=parent_ids
        ).distinct()

    def _find_nieces_nephews_optimized(self):
        """
        Optimized method to find all nieces and nephews.
        Uses a single query with prefetch_related.
        """
        sibling_ids = self.get_siblings().values_list('id', flat=True)

        return Person.objects.filter(
            parents_set__parent_id__in=sibling_ids
        ).distinct()


class Relationship(models.Model):
    """
    Represents relationships between people (marriages, partnerships, etc.).
    Tracks both current and historical relationships.
    """
    RELATIONSHIP_TYPES = [
        ('MARRIAGE', 'Marriage'),
        ('PARTNERSHIP', 'Partnership'),
        ('ENGAGEMENT', 'Engagement'),
        ('DIVORCED', 'Divorced'),
        ('SEPARATED', 'Separated'),
    ]

    person1 = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='relationships_as_person1'
    )
    person2 = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='relationships_as_person2'
    )
    relationship_type = models.CharField(
        max_length=20,
        choices=RELATIONSHIP_TYPES
    )
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    is_current = models.BooleanField(default=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = [['person1', 'person2', 'start_date']]
        indexes = [
            models.Index(fields=['person1', 'is_current']),
            models.Index(fields=['person2', 'is_current']),
        ]

    def clean(self):
        # Prevent relationship with self
        if self.person1 == self.person2:
            raise ValidationError("A person cannot have a relationship with themselves")

        # Validate dates
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValidationError("Start date must be before end date")

        # Check for overlapping current relationships
        if self.is_current:
            overlapping = Relationship.objects.filter(
                (Q(person1=self.person1) | Q(person2=self.person1)),
                is_current=True
            ).exclude(pk=self.pk)

            if overlapping.exists():
                raise ValidationError("This person already has a current relationship")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)


class ParentChild(models.Model):
    """
    Represents parent-child relationships in the family tree.
    Allows tracking both biological and non-biological relationships.
    """
    RELATIONSHIP_TYPES = [
        ('BIOLOGICAL', 'Biological'),
        ('ADOPTED', 'Adopted'),
        ('STEP', 'Step'),
        ('FOSTER', 'Foster'),
        ('GUARDIAN', 'Guardian'),
    ]

    parent = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='children_set'
    )
    child = models.ForeignKey(
        Person,
        on_delete=models.CASCADE,
        related_name='parents_set'
    )
    relationship_type = models.CharField(
        max_length=20,
        choices=RELATIONSHIP_TYPES,
        default='BIOLOGICAL'
    )
    notes = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = [['parent', 'child', 'relationship_type']]
        verbose_name_plural = "Parent-Child Relationships"
        indexes = [
            models.Index(fields=['parent']),
            models.Index(fields=['child']),
        ]

    def clean(self):
        # Prevent self-parenting
        if self.parent == self.child:
            raise ValidationError("A person cannot be their own parent")

        # Validate parent is older than child
        if self.parent.birth_date and self.child.birth_date:
            if self.parent.birth_date >= self.child.birth_date:
                raise ValidationError("Parent must be born before child")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)
