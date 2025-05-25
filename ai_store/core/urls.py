from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import home
from store.views import product_detail

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home, name="home"),
    path("product/<slug:slug>/", product_detail, name="product_detail"),
    path("store/", include("store.urls"))
]

# Serve media files in development (remove in production)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)