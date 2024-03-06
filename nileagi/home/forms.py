from django import forms

class ContactForm(forms.Form):
    name =  forms.CharField(max_length=200, widget=(forms.TextInput(attrs={'class': 'form-control','placeholder':'Your Name', 'id': 'name', 'required':'true'})))
    email = forms.EmailField(max_length=200, widget=(forms.EmailInput(attrs={'class': 'form-control', 'placeholder':'Your Email', 'id': 'email','required':'true'})))
    subject = forms.CharField(max_length=200, widget=(forms.TextInput(attrs={'class': 'form-control','placeholder':'Subject', 'id': 'subject','required':'true'})))
    message = forms.CharField(max_length=1000, widget=(forms.TextInput(attrs={'class': 'form-control','rows':'5', 'placeholder':'Message','required':'true'})))

class SubscriptionForm(forms.Form):
    email = forms.EmailField(max_length=20, widget=(forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your company email'})))