print("Hello World")

user = int(input("How much your budget is in Philippine Peso?: "))

if user <= 2000:
    print("Mag trabaho ka bai")
elif user <= 10000:
   print("Teka grind kapa") 
elif user <= 100000:
    print("Lenovo, Asus dami kana choices")
elif user <= 100000000:
    print("Paldo kana bilhin mona ako")
else:
    print("Ayusin monga, sayang yung API ko")
    