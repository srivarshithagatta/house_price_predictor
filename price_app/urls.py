from django.urls import path
from . import views

urlpatterns = [
    path('result/', views.predict_price, name='predict_price'), 
     # Route for the main prediction form
    path('', views.property_form, name='property_form'),
]
