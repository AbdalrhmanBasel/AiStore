from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Account
from django.utils.translation import gettext_lazy as _

@admin.register(Account)
class AccountAdmin(UserAdmin):
    # Fields to display in list view
    list_display = (
        'email', 'username', 'first_name', 'last_name', 
        'is_staff', 'is_active', 'date_joined'
    )
    
    # Make email clickable
    list_display_links = ('email',)
    
    # Filter options
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')
    
    # Fields to display in the form
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        (_('Personal Info'), {
            'fields': ('username', 'first_name', 'last_name', 'phone_number')
        }),
        (_('Permissions'), {
            'fields': ('is_active', 'is_staff', 'is_superuser')
        }),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    
    # Fields to display when adding a new user
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'username', 'first_name', 'last_name', 'password1', 'password2'),
        }),
        (_('Permissions'), {
            'fields': ('is_staff', 'is_active', 'is_superuser')
        }),
    )
    
    # Search functionality
    search_fields = ('email', 'username', 'first_name', 'last_name')
    
    # Ordering in admin
    ordering = ('-date_joined',)
    
    # Show read-only fields
    readonly_fields = ('date_joined', 'last_login')