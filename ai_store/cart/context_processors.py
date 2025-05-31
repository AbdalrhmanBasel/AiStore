from .models import Cart, CartItem

def cart_item_count(request):
    count = 0
    if request.user.is_authenticated:
        cart = Cart.objects.filter(user=request.user).first()
        if cart:
            count = cart.items.filter(is_active=True).count()
    else:
        cart_id = request.session.get('cart_id')
        if cart_id:
            cart = Cart.objects.filter(cart_id=cart_id, user__isnull=True).first()
            if cart:
                count = cart.items.filter(is_active=True).count()
    return {'cart_item_count': count}
