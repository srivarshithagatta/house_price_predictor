from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_price, name='predict_price'),  # Route for the main prediction form
]
