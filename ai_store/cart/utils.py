from cart.models import CartItem  # your cart app and model

def get_cart_items_and_totals(request):
    if request.user.is_authenticated:
        cart_items = CartItem.objects.filter(user=request.user)
    else:
        cart_items = CartItem.objects.none()  # Or handle anonymous cart
    
    total_price = sum(item.product.price * item.quantity for item in cart_items)
    tax = total_price * 0.1  # 10% tax example
    final_total = total_price + tax

    return cart_items, total_price, tax, final_total
