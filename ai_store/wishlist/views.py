from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from store.models import Product
from .models import WishlistItem
from cart.models import Cart, CartItem

# ---------- Helper function ----------

def get_user_cart(user):
    cart, created = Cart.objects.get_or_create(user=user)
    return cart

# ---------- Views ----------

@login_required
def wishlist(request):
    items = WishlistItem.objects.filter(user=request.user)
    return render(request, 'wishlist/wishlist.html', {'items': items})

@login_required
def add_to_wishlist(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    WishlistItem.objects.get_or_create(user=request.user, product=product)
    messages.success(request, f"Товар «{product.product_name}» добавлен в список желаемого.")
    return redirect('wishlist')

@login_required
def remove_from_wishlist(request, item_id):
    item = get_object_or_404(WishlistItem, id=item_id, user=request.user)
    item.delete()
    messages.info(request, "Товар удалён из списка желаемого.")
    return redirect('wishlist')

@login_required
def move_to_cart(request, item_id):
    wishlist_item = get_object_or_404(WishlistItem, id=item_id, user=request.user)
    product = wishlist_item.product

    if product.stock > 0:
        cart = get_user_cart(request.user)

        cart_item, created = CartItem.objects.get_or_create(
            cart=cart,
            product=product,
            defaults={'quantity': 1, 'is_active': True}
        )

        if not created:
            cart_item.is_active = True
            cart_item.quantity += 1

        cart_item.save()
        wishlist_item.delete()

        messages.success(request, f"Товар «{product.product_name}» перемещён в корзину.")
    else:
        messages.warning(request, f"Товар «{product.product_name}» нет в наличии.")

    return redirect('wishlist')

@login_required
def move_all_to_cart(request):
    wishlist_items = WishlistItem.objects.filter(user=request.user)
    cart = get_user_cart(request.user)

    moved_count = 0

    for item in wishlist_items:
        product = item.product

        if product.stock > 0:
            cart_item, created = CartItem.objects.get_or_create(
                cart=cart,
                product=product,
                defaults={'quantity': 1, 'is_active': True}
            )

            if not created:
                cart_item.is_active = True
                cart_item.quantity += 1

            cart_item.save()
            item.delete()

            moved_count += 1

    if moved_count > 0:
        messages.success(request, f"{moved_count} товар(ов) перемещено в корзину.")
    else:
        messages.warning(request, "Нет доступных товаров для перемещения в корзину.")

    return redirect('wishlist')
