# cart/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from decimal import Decimal, ROUND_HALF_UP

from .models import Cart, CartItem
from store.models import Product
from tracking.hooks import log_interaction

# ---------- Helpers ----------

def _cart_id(request):
    """
    Get or create session key.
    """
    cart = request.session.session_key
    if not cart:
        cart = request.session.create()
    return cart

def get_cart(request):
    """
    Get or create cart for authenticated user or anonymous session.
    Handles session → user merge.
    """
    if request.user.is_authenticated:
        user_cart, created = Cart.objects.get_or_create(user=request.user)

        # Migrate session cart if exists
        session_cart_id = request.session.get('cart_id')
        if session_cart_id:
            try:
                session_cart = Cart.objects.get(cart_id=session_cart_id, user__isnull=True)
                for item in session_cart.items.all():
                    existing_item = CartItem.objects.filter(cart=user_cart, product=item.product).first()
                    if existing_item:
                        existing_item.quantity += item.quantity
                        existing_item.save()
                    else:
                        item.cart = user_cart
                        item.save()
                session_cart.delete()
                del request.session['cart_id']
            except Cart.DoesNotExist:
                pass

        return user_cart

    else:
        # Anonymous user session cart
        cart_id = request.session.get('cart_id')
        if cart_id:
            cart = Cart.objects.filter(cart_id=cart_id, user__isnull=True).first()
            if not cart:
                cart = Cart.objects.create(cart_id=_cart_id(request))
                request.session['cart_id'] = cart.cart_id
        else:
            cart = Cart.objects.create(cart_id=_cart_id(request))
            request.session['cart_id'] = cart.cart_id

        return cart

# ---------- Views ----------

def cart(request):
    cart = get_cart(request)
    items = cart.items.filter(is_active=True).select_related('product')

    total_price = cart.total_price()
    tax_rate = Decimal('0.15')
    tax = (total_price * tax_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    final_total = (total_price + tax).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    context = {
        'cart_items': items,
        'cart_total_price': total_price,
        'cart_tax': tax,
        'cart_final_total': final_total,
    }
    return render(request, 'store/cart.html', context)

def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart = get_cart(request)

    if not product.is_available or product.stock < 1:
        messages.error(request, "Товар отсутствует на складе и не может быть добавлен в корзину.")
        return redirect('store')

    try:
        cart_item = CartItem.objects.get(product=product, cart=cart)
        if not cart_item.is_active:
            cart_item.is_active = True
            cart_item.quantity = 1
        else:
            if cart_item.quantity >= product.stock:
                messages.error(request, "Вы достигли максимального количества этого товара в корзине.")
                return redirect('cart')
            cart_item.quantity += 1
        cart_item.save()
    except CartItem.DoesNotExist:
        CartItem.objects.create(
            product=product,
            quantity=1,
            cart=cart,
            is_active=True,
        )

    # ⭐ Track cart interaction
    if request.user.is_authenticated:
        log_interaction(user_id=request.user.id, product_id=product.id, interaction_type='cart')

    messages.success(request, f"{product.product_name} добавлен в корзину.")
    return redirect('cart')

def remove_from_cart(request, item_id):
    cart = get_cart(request)
    item = get_object_or_404(CartItem, id=item_id, cart=cart)
    item.is_active = False
    item.save()
    return redirect('cart')

def update_quantity(request, item_id, action):
    cart = get_cart(request)
    item = get_object_or_404(CartItem, id=item_id, cart=cart)

    if action == 'increase':
        if item.quantity < item.product.stock:
            item.quantity += 1
        else:
            messages.error(request, "Достигнуто максимальное количество.")
            return redirect('cart')
    elif action == 'decrease' and item.quantity > 1:
        item.quantity -= 1

    item.save()

    # ⭐ Track cart interaction
    if request.user.is_authenticated:
        log_interaction(user_id=request.user.id, product_id=item.product.id, interaction_type='cart')

    return redirect('cart')
