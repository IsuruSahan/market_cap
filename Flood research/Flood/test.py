import Weather.FBP

fb_obj = Weather.FBP.FB('pressure.csv')

x = fb_obj.get_week()

print(type(x))

for i in x:
    print(i)