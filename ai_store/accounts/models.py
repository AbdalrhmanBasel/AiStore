from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.core.validators import RegexValidator

phone_regex = RegexValidator(
    regex=r'^\+?1?\d{9,15}$',
    message=_("Phone number must be entered in the format: +999999999. Up to 15 digits allowed.")
)

class MyAccountManager(BaseUserManager):
    def _create_user(self, email, username, first_name, last_name, password, **extra_fields):
        if not email:
            raise ValueError(_('The Email must be set'))
        if not username:
            raise ValueError(_('The Username must be set'))
        
        email = self.normalize_email(email)
        user = self.model(
            email=email,
            username=username,
            first_name=first_name,
            last_name=last_name,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, username, first_name, last_name, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        extra_fields.setdefault('is_active', True)
        
        return self._create_user(email, username, first_name, last_name, password, **extra_fields)

    def create_superuser(self, email, username, first_name, last_name, password, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError(_('Superuser must have is_staff=True.'))
        if extra_fields.get('is_superuser') is not True:
            raise ValueError(_('Superuser must have is_superuser=True.'))
        
        return self._create_user(email, username, first_name, last_name, password, **extra_fields)

class Account(AbstractBaseUser, PermissionsMixin):
    first_name = models.CharField(_('first name'), max_length=50)
    last_name = models.CharField(_('last name'), max_length=50)
    username = models.CharField(_('username'), max_length=50, unique=True)
    email = models.EmailField(_('email address'), max_length=100, unique=True)
    phone_number = models.CharField(_('phone number'), validators=[phone_regex], max_length=17, blank=True, null=True)
    
    # Required fields
    date_joined = models.DateTimeField(_('date joined'), auto_now_add=True)
    last_login = models.DateTimeField(_('last login'), auto_now=True)
    
    # Permissions
    is_staff = models.BooleanField(_('is staff'), default=False)
    is_active = models.BooleanField(_('is active'), default=False)
    is_superuser = models.BooleanField(_('is superuser'), default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']

    objects = MyAccountManager()

    def __str__(self):
        return self.email

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"

    def get_short_name(self):
        return self.first_name

    def has_perm(self, perm, obj=None):
        """Return True if the user has the specified permission."""
        return self.is_superuser or self.is_staff and self.is_active

    def has_module_perms(self, app_label):
        """Return True if the user has permissions in the given app."""
        return self.is_superuser or self.is_staff and self.is_active

    class Meta:
        verbose_name = _('account')
        verbose_name_plural = _('accounts')
        ordering = ['-date_joined']