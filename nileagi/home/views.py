from django.shortcuts import render
from . forms import  ContactForm, SubscriptionForm
from . models import Messages, Subscribers
import string
import random

# Create your views here.
def home(request):
    return render(request, 'system/pages/index.html')

def about(request):
    return render(request, 'system/pages/about.html')

def services_products(request):
    return render(request, 'system/pages/services.html')

def blog(request):
    return render(request, 'system/pages/blog.html')

def contact(request):

    contact_form = ContactForm()

    context = {'contact_form': contact_form}

    if request.method == 'POST':

        contact_form = ContactForm(request.POST)

        if contact_form.is_valid():
            name = request.POST['name']
            email = request.POST['email']
            subject = request.POST['subject']
            message = request.POST['message']
            letters = string.ascii_uppercase
            message_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
            new_data = Messages(message_id=message_id,name=name, email=email, subject=subject,message=message)
            new_data.save()
            success_msg = 'Successfully submitted, Thank you!'
            context = {'success_msg': success_msg}
            return render(request, 'opticai/request_demo_response.html', context=context) 
        else:
            return render(request, 'system/pages/contact.html', context=context)  
    else:
        return render(request, 'system/pages/contact.html', context=context)

def subscribe(request):

    subscription_form = SubscriptionForm()

    context = {'subscription_form': subscription_form}

    if request.method == 'POST':

        subscription_form = SubscriptionForm(request.POST)

        if subscription_form.is_valid():
            email = request.POST['email']
            letters = string.ascii_uppercase
            email_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
            new_data = Subscribers(email_id=email_id, email=email)
            new_data.save()
            success_msg = 'You have successfully subscribed, Thank you!'
            context = {'success_msg': success_msg}
            return render(request, 'pages/subscription_response.html', context=context) 
        else:
            return render(request, 'pages/index.html', context=context)  
    else:
        return render(request, 'pages/index.html', context=context)