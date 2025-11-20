# File: detector/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # '/' - Our new homepage
    path('', views.home_view, name='home'),
    
    # '/detector/' - The upload page
    path('detector/', views.upload_and_predict, name='detector'),
    
    # '/about/' - The about page
    path('about/', views.about_view, name='about'),
]