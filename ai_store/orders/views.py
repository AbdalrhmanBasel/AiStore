# orders/views.py

from django.shortcuts import render, redirect, get_object_or_404
from cart.models import Cart, CartItem
from store.models import Product
from orders.models import Order, OrderItem, Payment
from .forms import BillingForm
from django.contrib import messages
from django.db import transaction
from decimal import Decimal, ROUND_HALF_UP
from django.contrib.auth.decorators import login_required
from cart.views import get_cart
from tracking.hooks import log_interaction  # ⭐ Add this

@login_required(login_url='login')
def checkout(request):
    cart = get_cart(request)
    cart_items = CartItem.objects.filter(cart=cart, is_active=True).select_related('product')

    # Calculate totals
    total_price = cart.total_price()
    tax_rate = Decimal('0.15')
    tax = (total_price * tax_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    final_total = (total_price + tax).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    if request.method == "POST":
        form = BillingForm(request.POST)
        payment_method = request.POST.get('payment_method')

        if form.is_valid():
            if not cart_items.exists():
                messages.error(request, "Ваша корзина пуста.")
                return redirect('store')

            try:
                with transaction.atomic():
                    # Create Order
                    order = form.save(commit=False)
                    order.user = request.user
                    order.total_price = final_total
                    order.save()

                    # Create OrderItems + ⭐ Track purchase
                    for item in cart_items:
                        OrderItem.objects.create(
                            order=order,
                            product=item.product,
                            quantity=item.quantity,
                            price=item.product.price
                        )

                        # ⭐ Track purchase interaction
                        log_interaction(user_id=request.user.id, product_id=item.product.id, interaction_type='purchase')

                    # Create Payment
                    Payment.objects.create(
                        order=order,
                        method=payment_method,
                        status='pending',
                        amount_paid=Decimal('0.00')
                    )

                    # Clear Cart
                    cart_items.delete()

                    # Success message
                    messages.success(request, "Ваш заказ оформлен. Перенаправление на оплату...")

                    # Redirect
                    if payment_method == 'sbp':
                        return redirect('sbp_payment', order_id=order.id)
                    elif payment_method == 'sberpay':
                        return redirect('sberpay_payment', order_id=order.id)
                    else:
                        return redirect('order_success')

            except Exception as e:
                messages.error(request, f"Ошибка при обработке заказа: {e}")
        else:
            messages.error(request, "Пожалуйста, исправьте ошибки в форме.")
    else:
        form = BillingForm()

    context = {
        'billing_form': form,
        'cart_items': cart_items,
        'cart_total_price': total_price,
        'cart_tax': tax,
        'cart_final_total': final_total,
    }
    return render(request, 'orders/checkout.html', context)

# Order success page
def order_success(request):
    return render(request, 'orders/order_success.html')

# Order history
@login_required
def order_history(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'orders/order_history.html', {'orders': orders})

# Order detail
@login_required
def order_detail(request, order_id):
    order = get_object_or_404(Order, id=order_id, user=request.user)
    return render(request, 'orders/order_detail.html', {'order': order})

# 🚀 placeholders for payment pages:
@login_required
def sbp_payment(request, order_id):
    order = get_object_or_404(Order, id=order_id, user=request.user)
    return render(request, 'orders/sbp_payment.html', {'order': order})

@login_required
def sberpay_payment(request, order_id):
    order = get_object_or_404(Order, id=order_id, user=request.user)
    return render(request, 'orders/sberpay_payment.html', {'order': order})
