print("Merhaba, ben Alperen Albayrak!")

# GÖREV 1
print(type({"asd", "asda", "asdas"}))

# GÖREV 2
text = "We can create complex number from two real numbers."
print(text.upper().split())

# GÖREV 3
text = "DATASCIENCE"
lst = list(text)

print(len(lst))

print(lst[0], lst[10])
print(lst[:4:])
lst.pop(8)
lst.insert(8, "Q")
lst[8] = "N"

print(lst)

# GÖREV 4
dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

print(dict.keys())
print(dict.values())

dict.update({"Daisy": ["England", 13]})
print(dict)

dict.setdefault("Ahmet", ["Turkey", 24])
print(dict)

dict.pop("Antonio")
print(dict)

# GÖREV 5
l = [2, 13, 18, 93, 22]


def func(num_lst):
    odd = []
    even = []
    for index, element in enumerate(num_lst):
        if index % 2 == 0:
            even.append(element)
        else:
            odd.append(element)
    return print(f"odd list: {odd}, even list: {even}")


func(l)

# GÖREV 6
import seaborn as sns
df = sns.load_dataset("car_crashes")
print(df.columns)

# GÖREV 7
print(["NUM_" + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns])
print([col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns])

# GÖREV 8
og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
print(new_cols)

new_df = df[new_cols]
print(new_df.head())
