from django.shortcuts import render, redirect, get_object_or_404
from .models import Cart, CartItem
from store.models import Product
from django.shortcuts import render, redirect
from django.contrib import messages

from .models import Cart, CartItem
from decimal import Decimal, ROUND_HALF_UP


def _cart_id(request):
    """
    Get or create the session key for the cart.
    """
    cart = request.session.session_key
    if not cart:
        cart = request.session.create()
    return cart

def get_cart(request):
    """
    Retrieve or create the Cart object associated with the current session.
    """
    cart_id = request.session.get('cart_id')
    if cart_id:
        cart = Cart.objects.filter(cart_id=cart_id).first()
        if cart is None:
            # Session cart_id is stale, create new cart
            cart = Cart.objects.create(cart_id=_cart_id(request))
            request.session['cart_id'] = cart.cart_id
    else:
        cart = Cart.objects.create(cart_id=_cart_id(request))
        request.session['cart_id'] = cart.cart_id
    return cart

from decimal import Decimal, ROUND_HALF_UP

def cart(request):
    cart = get_cart(request)
    items = cart.items.filter(is_active=True).select_related('product')

    total_price = cart.total_price()  # this should be Decimal
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

from django.contrib import messages

def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart = get_cart(request)

    # Check stock availability
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
    messages.success(request, f"{product.product_name} добавлен в корзину.")
    return redirect('cart')


def remove_from_cart(request, item_id):
    """
    Soft-remove the cart item by marking it inactive.
    """
    cart = get_cart(request)
    item = get_object_or_404(CartItem, id=item_id, cart=cart)
    item.is_active = False
    item.save()
    return redirect('cart')

def update_quantity(request, item_id, action):
    """
    Increase or decrease the quantity of a cart item.
    """
    cart = get_cart(request)
    item = get_object_or_404(CartItem, id=item_id, cart=cart)

    if action == 'increase':
        item.quantity += 1
    elif action == 'decrease' and item.quantity > 1:
        item.quantity -= 1

    item.save()
    return redirect('cart')




def get_cart(request):
    if request.user.is_authenticated:
        # Get or create user's cart
        user_cart, created = Cart.objects.get_or_create(user=request.user)

        # Migrate session cart items to user cart if session cart exists
        session_cart_id = request.session.get('cart_id')
        if session_cart_id:
            try:
                session_cart = Cart.objects.get(cart_id=session_cart_id, user__isnull=True)
                for item in session_cart.items.all():
                    # Merge or reassign cart items
                    existing_item = CartItem.objects.filter(cart=user_cart, product=item.product).first()
                    if existing_item:
                        existing_item.quantity += item.quantity
                        existing_item.save()
                    else:
                        item.cart = user_cart
                        item.save()
                session_cart.delete()  # Remove empty session cart
                del request.session['cart_id']  # Clear session cart id
            except Cart.DoesNotExist:
                pass

        return user_cart
    else:
        # Anonymous user: get or create cart by session
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


