from django.db import models
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager

from django.db.models.signals import post_save
from django.dispatch import receiver

class CustomUserManager(BaseUserManager):
    def _create_user(self, email, password, **extra_fields):

        if not email:
            raise ValueError("Email must be written.")
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have `is_staff=True`.")
        
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have `is_superuser=True`.")
        
        return self._create_user(email, password, **extra_fields)
        

class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(("email address"), unique=True, null=False)
    is_staff = models.BooleanField("staff status", default=False)
    is_active = models.BooleanField(("active"), default=True)

    objects = CustomUserManager()
    USERNAME_FIELD = 'email'

    def get_full_name(self):
        """
        TODO: Fix this function to render user's full name correctly
        """
        return self.first_name + self.middle_name + self.last_name

    
    def __str__(self):
        return self.email
    

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    username = models.CharField(max_length=100, blank=True)


    first_name = models.CharField(max_length=100)
    middle_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    address_1 = models.CharField(max_length=300, blank=True)
    address_2 = models.CharField(max_length=300, blank=True)

    country = models.CharField(max_length=40, blank=True)
    city = models.CharField(max_length=50, blank=True)
    zipcode = models.CharField(max_length=10, blank=True)

    phone = models.CharField(max_length=20, blank=True)

    birth_date = models.DateField(blank=True)
    date_joined = models.DateTimeField(auto_now_add=True)


    def get_first_name(self):
        return self.first_name
    
    def get_middle_name(self):
        return self.middle_name
    
    def get_last_name(self):
        return self.last_name

    def __str__(self):
        return self.username

    def is_fully_filled(self):
        """
        Method to check if all the fields are filled
        """
        fields_names = [f.name for f in self._meta.get_fields()]

        for field_name in fields_names:
            value = getattr(self,field_name)
            if value is None or value == "":
                return False
            
        return True
    
    

@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    

@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()
