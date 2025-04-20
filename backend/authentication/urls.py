from django.urls import path
from . import views

app_name = "authentication"

urlpatterns = [
    path('api/signup/', views.user_signup, name='user_signup'),
    path('api/login/', views.user_login, name='user_login'),
    path('api/logout/', views.user_logout, name='user_logout'),
    path('api/profile/', views.user_profile, name='user_profile'),
    # TODO: path('api/restore_password/', views.user_restore_password, name='user_restore_password'),
]