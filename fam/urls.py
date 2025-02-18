from django.urls import path
from . import views

app_name = 'family_tree'

urlpatterns = [
    # Root URL
    path('', views.index, name='index'),  # Add this line for the root URL

    # Person management
    path('person/add/', views.add_person, name='add_person'),
    path('person/<int:pk>/', views.person_detail, name='person_detail'),
    path('person/search/', views.search_people, name='search_people'),

    # Relationship management
    path('relationship/add/', views.add_relationship, name='add_relationship'),
    path('relationship/parent-child/add/', views.add_parent_child, name='add_parent_child'),
    path('relationship/<int:person1_id>/<int:person2_id>/', views.find_relationship, name='find_relationship'),

    # Family tree visualization
    path('tree/<int:person_id>/', views.view_tree, name='view_tree'),

    # Family analysis
    path('analyze/<int:person_id>/', views.analyze_family, name='analyze_family'),
]