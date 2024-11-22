from django import forms

class PricePredictionForm(forms.Form):
    bedrooms = forms.IntegerField()
    bathrooms = forms.FloatField()
    sqft_living = forms.IntegerField()
    sqft_lot = forms.IntegerField()
    floors = forms.FloatField()
    waterfront = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    view = forms.IntegerField()
    condition = forms.IntegerField()
    sqft_above = forms.IntegerField()
    sqft_basement = forms.IntegerField()
    yr_built = forms.IntegerField()
    yr_renovated = forms.IntegerField()
